# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
InfiniteTalk REST API Server

FastAPI-based REST API server for video generation tasks.
Runs on a separate port from the Gradio UI to allow simultaneous use.

Usage:
    python api_server.py --port 8419
    
    Or with uvicorn directly:
    uvicorn api_server:app --host 0.0.0.0 --port 8419 --reload
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
import warnings


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

API_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_results")
DEFAULT_PORT = 8419
DEFAULT_HOST = "0.0.0.0"

# GPU access lock — only one generation task runs at a time
gpu_lock = threading.Lock()

# Pipeline initialization lock — prevents double initialization on concurrent requests
_init_lock = threading.Lock()

# ============================================================
# Argument Parsing (mirrors generate_infinitetalk.py defaults)
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="InfiniteTalk REST API Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Server host")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Wan checkpoint directory")
    parser.add_argument("--infinitetalk_dir", type=str, default=None, help="InfiniteTalk checkpoint directory")
    parser.add_argument("--wav2vec_dir", type=str, default=None, help="Wav2Vec checkpoint directory")
    parser.add_argument("--quant_dir", type=str, default=None, help="Quant checkpoint directory")
    parser.add_argument("--dit_path", type=str, default=None, help="DiT checkpoint path")
    parser.add_argument("--lora_dir", type=str, nargs="+", default=None, help="LoRA checkpoint paths")
    parser.add_argument("--lora_scale", type=float, nargs="+", default=[1.2], help="LoRA scales")
    parser.add_argument("--offload_model", type=str2bool, default=None, help="Offload model to CPU")
    parser.add_argument("--task", type=str, default="infinitetalk-14B",
                        choices=["infinitetalk-14B"], help="The task to run.")
    parser.add_argument("--ulysses_size", type=int, default=1, help="Ulysses parallelism size")
    parser.add_argument("--ring_size", type=int, default=1, help="Ring attention parallelism size")
    parser.add_argument("--t5_fsdp", action="store_true", default=False, help="Use FSDP for T5")
    parser.add_argument("--t5_cpu", action="store_true", default=False, help="Place T5 on CPU")
    parser.add_argument("--dit_fsdp", action="store_true", default=False, help="Use FSDP for DiT")
    parser.add_argument("--motion_frame", type=int, default=9, help="Driven frame length for long video")
    parser.add_argument("--frame_num", type=int, default=81, help="Frames per clip (4n+1)")
    parser.add_argument("--num_persistent_param_in_dit", type=int, default=None, help="VRAM management param count")
    parser.add_argument("--use_teacache", action="store_true", default=False, help="Enable teacache")
    parser.add_argument("--teacache_thresh", type=float, default=0.2, help="Teacache threshold")
    parser.add_argument("--use_apg", action="store_true", default=False, help="Enable APG")
    parser.add_argument("--apg_momentum", type=float, default=-0.75, help="APG momentum")
    parser.add_argument("--apg_norm_threshold", type=float, default=55, help="APG norm threshold")
    parser.add_argument("--color_correction_strength", type=float, default=1.0, help="Color correction strength")
    parser.add_argument("--quant", type=str, default=None, help="Quantization type (int8/fp8)")
    parser.add_argument("--sample_shift", type=float, default=None, help="Sampling shift factor")
    parser.add_argument("--sample_steps", type=int, default=None, help="Sampling steps")
    return parser.parse_args()


# ============================================================
# Task Model
# ============================================================

class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskInfo:
    """Internal task tracking object."""
    
    def __init__(self, task_id: str, params: dict):
        self.task_id = task_id
        self.params = params
        self.status = TaskStatus.QUEUED
        self.progress = 0
        self.log = []
        self.error = None
        self.result_path = None
        self.cancel_event = threading.Event()
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": self.progress,
            "log": self.log[-100:],  # Keep last 100 log lines
            "error": self.error,
            "result_url": f"/api/download/{self.task_id}" if self.status == TaskStatus.DONE else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def add_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{timestamp}] {message}")
        self.updated_at = datetime.now().isoformat()

    def update_progress(self, progress: int):
        self.progress = min(progress, 100)
        self.updated_at = datetime.now().isoformat()


# ============================================================
# Task Manager
# ============================================================

class TaskManager:
    """Manages task queue, state persistence, and background execution."""

    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = threading.Lock()

        self._pipeline = None
        self._pipeline_args = None
        self._wav2vec_feature_extractor = None
        self._audio_encoder = None
        # Note: these will be initialized lazily on first task

    def set_pipeline(self, pipeline, args, wav2vec_fe, audio_encoder):
        self._pipeline = pipeline
        self._pipeline_args = args
        self._wav2vec_feature_extractor = wav2vec_fe
        self._audio_encoder = audio_encoder

    def create_task(self, params: dict) -> str:
        task_id = str(uuid.uuid4())
        task = TaskInfo(task_id, params)
        with self._lock:
            self._tasks[task_id] = task
        return task_id

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> list[dict]:
        with self._lock:
            return [t.to_dict() for t in self._tasks.values()]

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status in (TaskStatus.QUEUED, TaskStatus.RUNNING):
                task.cancel_event.set()
                if task.status == TaskStatus.QUEUED:
                    task.status = TaskStatus.CANCELLED
                    task.add_log("Task cancelled before execution.")
                else:
                    task.add_log("Cancellation requested...")
                return True
            return False

    def submit_task(self, task_id: str):
        """Submit a task for background execution."""
        task = self.get_task(task_id)
        if task is None:
            return
        thread = threading.Thread(target=self._run_task, args=(task_id,), daemon=True)
        thread.start()

    def _run_task(self, task_id: str):
        """Execute a generation task in a background thread."""
        task = self.get_task(task_id)
        if task is None:
            return

        acquired = gpu_lock.acquire(timeout=5)
        if not acquired:
            task.status = TaskStatus.FAILED
            task.error = "GPU is busy and could not acquire lock within timeout."
            task.add_log("ERROR: GPU is busy. Task queued timeout.")
            return

        try:
            # Check cancellation before starting
            if task.cancel_event.is_set():
                task.status = TaskStatus.CANCELLED
                task.add_log("Task was cancelled before execution started.")
                return

            task.status = TaskStatus.RUNNING
            task.add_log("Task started.")

            self._execute_generation(task)

            if not task.cancel_event.is_set() and task.status != TaskStatus.FAILED:
                task.status = TaskStatus.DONE
                task.progress = 100
                task.add_log("Task completed successfully.")
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.add_log(f"ERROR: {e}")
            import traceback
            task.add_log(traceback.format_exc())
        finally:
            if acquired:
                gpu_lock.release()
            task.updated_at = datetime.now().isoformat()

    def _execute_generation(self, task: TaskInfo):
        """Execute the actual video generation."""
        from generate_infinitetalk import (
            audio_prepare_multi,
            audio_prepare_single,
            custom_init,
            get_embedding,
            process_tts_single,
            process_tts_multi,
        )

        import torch
        import soundfile as sf

        args = task.params
        pipeline = self._pipeline
        pipeline_args = self._pipeline_args
        wav2vec_fe = self._wav2vec_feature_extractor
        audio_encoder = self._audio_encoder

        # Initialize wav2vec if not done yet
        if wav2vec_fe is None or audio_encoder is None:
            task.add_log("Initializing wav2vec...")
            wav2vec_dir = args.get("wav2vec_dir", pipeline_args.wav2vec_dir or "./weights/chinese-wav2vec2-base")
            wav2vec_fe, audio_encoder = custom_init("cpu", wav2vec_dir)
            # Store back
            self._wav2vec_feature_extractor = wav2vec_fe
            self._audio_encoder = audio_encoder

        task.update_progress(5)
        task.add_log("Preparing input data...")

        # Build input_data dict
        input_data = {}
        input_data["prompt"] = args.get("prompt", "")
        
        task_mode = args.get("task_mode", "VideoDubbing")
        mode_selector = args.get("mode_selector", "Single Person(Local File)")
        resolution = args.get("resolution", "infinitetalk-480")

        # Determine the cond_video path
        cond_video = args.get("cond_video", "")
        input_data["cond_video"] = cond_video

        # Build cond_audio dict
        cond_audio = {}
        audio_path_1 = args.get("audio_1", "")
        
        if "Local File" in mode_selector:
            if audio_path_1:
                cond_audio["person1"] = audio_path_1
            audio_path_2 = args.get("audio_2", "")
            if audio_path_2:
                cond_audio["person2"] = audio_path_2
                input_data["audio_type"] = "add" if "audio add" in mode_selector else "para"
        elif "TTS" in mode_selector:
            tts_text = args.get("tts_text", "")
            voice1 = args.get("voice_1", pipeline_args.lora_dir[0] if pipeline_args.lora_dir else "weights/Kokoro-82M/voices/am_adam.pt")
            voice2 = args.get("voice_2", "weights/Kokoro-82M/voices/af_heart.pt")
            input_data["tts_audio"] = {
                "text": tts_text,
                "human1_voice": voice1,
            }
            if "Multi" in mode_selector:
                input_data["tts_audio"]["human2_voice"] = voice2

        input_data["cond_audio"] = cond_audio

        task.update_progress(10)
        task.add_log(f"Mode: {mode_selector}, Resolution: {resolution}")

        # Check cancellation
        if task.cancel_event.is_set():
            task.add_log("Cancelled during preparation.")
            return

        # TTS processing or local file processing
        save_dir = os.path.join(API_RESULTS_DIR, task.task_id, "audio")
        os.makedirs(save_dir, exist_ok=True)

        if "TTS" in mode_selector:
            task.add_log("Generating TTS audio...")
            tts_data = input_data.get("tts_audio", {})
            if "human2_voice" in tts_data:
                new_human_speech1, new_human_speech2, sum_audio = process_tts_multi(
                    tts_data["text"], save_dir, tts_data["human1_voice"], tts_data["human2_voice"]
                )
                audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_fe, audio_encoder)
                audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_fe, audio_encoder)
                emb1_path = os.path.join(save_dir, "1.pt")
                emb2_path = os.path.join(save_dir, "2.pt")
                torch.save(audio_embedding_1, emb1_path)
                torch.save(audio_embedding_2, emb2_path)
                input_data["cond_audio"]["person1"] = emb1_path
                input_data["cond_audio"]["person2"] = emb2_path
                input_data["video_audio"] = sum_audio
            else:
                new_human_speech1, sum_audio = process_tts_single(
                    tts_data["text"], save_dir, tts_data["human1_voice"]
                )
                audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_fe, audio_encoder)
                emb1_path = os.path.join(save_dir, "1.pt")
                torch.save(audio_embedding_1, emb1_path)
                input_data["cond_audio"]["person1"] = emb1_path
                input_data["video_audio"] = sum_audio
        elif "Local File" in mode_selector:
            task.add_log("Processing audio files...")
            if len(cond_audio) == 2:
                new_human_speech1, new_human_speech2, sum_human_speechs = audio_prepare_multi(
                    cond_audio["person1"], cond_audio["person2"], input_data["audio_type"]
                )
                audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_fe, audio_encoder)
                audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_fe, audio_encoder)
                emb1_path = os.path.join(save_dir, "1.pt")
                emb2_path = os.path.join(save_dir, "2.pt")
                sum_audio = os.path.join(save_dir, "sum.wav")
                sf.write(sum_audio, sum_human_speechs, 16000)
                torch.save(audio_embedding_1, emb1_path)
                torch.save(audio_embedding_2, emb2_path)
                input_data["cond_audio"]["person1"] = emb1_path
                input_data["cond_audio"]["person2"] = emb2_path
                input_data["video_audio"] = sum_audio
            elif len(cond_audio) == 1:
                human_speech = audio_prepare_single(cond_audio["person1"])
                audio_embedding = get_embedding(human_speech, wav2vec_fe, audio_encoder)
                emb_path = os.path.join(save_dir, "1.pt")
                sum_audio = os.path.join(save_dir, "sum.wav")
                sf.write(sum_audio, human_speech, 16000)
                torch.save(audio_embedding, emb_path)
                input_data["cond_audio"]["person1"] = emb_path
                input_data["video_audio"] = sum_audio

        task.update_progress(20)
        task.add_log("Audio processing complete. Starting video generation...")

        # Check cancellation
        if task.cancel_event.is_set():
            task.add_log("Cancelled before generation.")
            return

        # Generate video
        task.add_log("Running InfiniteTalk pipeline...")

        # Add a progress callback mechanism by wrapping the progress reporting
        # We'll use a simple timer-based approach since the pipeline doesn't natively support callbacks
        # This reports approximate progress based on diffusion steps and segments
        sampling_steps = args.get("sampling_steps", pipeline_args.sample_steps or 8)
        mode = args.get("mode", pipeline_args.mode if hasattr(pipeline_args, "mode") else "streaming")
        max_frames = args.get("frame_num", 81) if mode == "clip" else args.get("max_frame_num", 1000)

        # Progress estimation:
        # 20-30%: TTS/audio prep
        # 30-90%: generation (this is the heavy part)
        # 90-100%: saving output

        # We'll run generation and check cancel periodically
        # Since generate_infinitetalk runs a single blocking call, 
        # we can't easily interrupt mid-generation.
        # The cancel_event will be checked only between generation chunks.

        resolution = args.get("resolution", "infinitetalk-480")
        default_shift = 7.0 if resolution == "infinitetalk-480" else 11.0

        try:
            video = pipeline.generate_infinitetalk(
                input_data,
                size_buckget=resolution,
                motion_frame=args.get("motion_frame", pipeline_args.motion_frame),
                frame_num=args.get("frame_num", pipeline_args.frame_num),
                shift=args.get("sample_shift", getattr(pipeline_args, 'sample_shift', None) or default_shift),
                sampling_steps=sampling_steps,
                text_guide_scale=args.get("text_guide_scale", pipeline_args.sample_text_guide_scale),
                audio_guide_scale=args.get("audio_guide_scale", pipeline_args.sample_audio_guide_scale),
                seed=args.get("seed", pipeline_args.base_seed),
                offload_model=pipeline_args.offload_model,
                max_frames_num=max_frames,
                color_correction_strength=args.get("color_correction_strength", pipeline_args.color_correction_strength),
                extra_args=pipeline_args,
            )
        except Exception as e:
            task.add_log(f"Generation error: {e}")
            raise

        # Check cancellation after generation
        if task.cancel_event.is_set():
            task.add_log("Task was cancelled (generation may have completed partially).")
            # Even if cancelled, we might have a partial result
            if video is None:
                return

        task.update_progress(85)
        task.add_log("Generation complete. Saving output video...")

        # Save the generated video
        from wan.utils.multitalk_utils import save_video_ffmpeg

        output_dir = os.path.join(API_RESULTS_DIR, task.task_id)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{task.task_id}.mp4")
        save_video_ffmpeg(video, output_path.replace(".mp4", ""), [input_data["video_audio"]], high_quality_save=False)

        task.result_path = output_path
        task.update_progress(100)
        task.add_log(f"Video saved to {output_path}")


# ============================================================
# Global state
# ============================================================

task_manager = TaskManager()
server_args = None


# ============================================================
# Lifespan handler — initialize pipeline on startup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global server_args
    logging.info("Starting InfiniteTalk API server...")
    os.makedirs(API_RESULTS_DIR, exist_ok=True)

    # Initialize the pipeline lazily on first request (not at startup)
    # to avoid blocking the startup. The pipeline is heavy (~14B params)
    # and will be initialized when the first generation task arrives.

    yield

    # Cleanup on shutdown
    logging.info("Shutting down InfiniteTalk API server...")
    # Clean up old/generated files
    for task_id in list(task_manager._tasks.keys()):
        task_dir = os.path.join(API_RESULTS_DIR, task_id)
        if os.path.exists(task_dir):
            shutil.rmtree(task_dir, ignore_errors=True)


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="InfiniteTalk REST API",
    description="REST API for audio-driven video generation using InfiniteTalk",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Helper functions
# ============================================================

def _init_pipeline_if_needed(task: TaskInfo):
    """Initialize the InfiniteTalk pipeline if it hasn't been initialized yet."""
    global server_args
    if task_manager._pipeline is not None:
        return
    with _init_lock:
        if task_manager._pipeline is not None:
            return

    task.add_log("Initializing InfiniteTalk pipeline (first run, this may take a while)...")

    args = server_args
    from generate_infinitetalk import custom_init, _init_logging

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank

    _init_logging(rank)

    # Set default values if not specified
    if args.offload_model is None:
        args.offload_model = True

    assert args.task == "infinitetalk-14B", "Only 'infinitetalk-14B' task is supported."

    # Locate ckpt_dir and weights
    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        # Try common locations
        for candidate in ["./weights/Wan2.1-I2V-14B-480P", "../weights/Wan2.1-I2V-14B-480P"]:
            if os.path.exists(candidate):
                ckpt_dir = candidate
                break
        if ckpt_dir is None:
            ckpt_dir = "./weights/Wan2.1-I2V-14B-480P"

    infinitetalk_dir = args.infinitetalk_dir
    if infinitetalk_dir is None:
        for candidate in ["./weights/InfiniteTalk/single/infinitetalk.safetensors",
                          "../weights/InfiniteTalk/single/infinitetalk.safetensors"]:
            if os.path.exists(candidate):
                infinitetalk_dir = candidate
                break
        if infinitetalk_dir is None:
            infinitetalk_dir = "weights/InfiniteTalk/single/infinitetalk.safetensors"

    wav2vec_dir = args.wav2vec_dir
    if wav2vec_dir is None:
        for candidate in ["./weights/chinese-wav2vec2-base", "../weights/chinese-wav2vec2-base"]:
            if os.path.exists(candidate):
                wav2vec_dir = candidate
                break
        if wav2vec_dir is None:
            wav2vec_dir = "./weights/chinese-wav2vec2-base"

    import wan
    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS["infinitetalk-14B"]
    task.add_log(f"Checkpoint dir: {ckpt_dir}")
    task.add_log(f"Infinitetalk dir: {infinitetalk_dir}")
    task.add_log(f"Wav2vec dir: {wav2vec_dir}")

    # Initialize wav2vec
    wav2vec_fe, audio_encoder = custom_init("cpu", wav2vec_dir)

    # Create pipeline
    task.add_log("Creating InfiniteTalk pipeline...")
    pipeline = wan.InfiniteTalkPipeline(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        quant_dir=args.quant_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
        lora_dir=args.lora_dir,
        lora_scales=args.lora_scale,
        quant=args.quant,
        dit_path=args.dit_path,
        infinitetalk_dir=infinitetalk_dir,
    )

    if args.num_persistent_param_in_dit is not None:
        pipeline.vram_management = True
        pipeline.enable_vram_management(
            num_persistent_param_in_dit=args.num_persistent_param_in_dit
        )

    task_manager.set_pipeline(pipeline, args, wav2vec_fe, audio_encoder)
    task.add_log("Pipeline initialized.")


def _save_uploaded_file(upload: UploadFile, task_dir: str, filename: str) -> str:
    """Save an uploaded file to the task directory."""
    os.makedirs(task_dir, exist_ok=True)
    file_path = os.path.join(task_dir, filename)
    with open(file_path, "wb") as f:
        f.write(upload.file.read())
    return file_path


# ============================================================
# API Endpoints
# ============================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "gpu_busy": gpu_lock.locked(),
        "active_tasks": sum(
            1 for t in task_manager._tasks.values()
            if t.status in (TaskStatus.QUEUED, TaskStatus.RUNNING)
        ),
    }


@app.post("/api/generate")
async def create_generation_task(
    # Task mode
    task_mode: str = Form("VideoDubbing", description="Task mode: 'SingleImageDriven' or 'VideoDubbing'"),
    mode_selector: str = Form("Single Person(Local File)", description="Audio mode selector"),
    prompt: str = Form("", description="Text prompt for generation"),
    
    # Files (optional — pass file paths for local files instead)
    image: Optional[UploadFile] = File(None, description="Input image (for SingleImageDriven)"),
    video: Optional[UploadFile] = File(None, description="Input video (for VideoDubbing)"),
    audio_1: Optional[UploadFile] = File(None, description="Audio for person 1"),
    audio_2: Optional[UploadFile] = File(None, description="Audio for person 2"),
    
    # Local file paths (alternative to file uploads)
    image_path: Optional[str] = Form(None, description="Local path to input image"),
    video_path: Optional[str] = Form(None, description="Local path to input video"),
    audio_1_path: Optional[str] = Form(None, description="Local path to audio for person 1"),
    audio_2_path: Optional[str] = Form(None, description="Local path to audio for person 2"),
    
    # TTS parameters
    tts_text: Optional[str] = Form(None, description="Text for TTS generation"),
    voice_1_path: Optional[str] = Form(None, description="Path to voice embedding for speaker 1"),
    voice_2_path: Optional[str] = Form(None, description="Path to voice embedding for speaker 2"),
    
    # Generation parameters
    resolution: str = Form("infinitetalk-480", description="Resolution: 'infinitetalk-480' or 'infinitetalk-720'"),
    sample_steps: int = Form(8, description="Number of diffusion sampling steps"),
    seed: int = Form(42, description="Random seed (-1 for random)"),
    text_guide_scale: float = Form(1.0, description="Text classifier-free guidance scale"),
    audio_guide_scale: float = Form(2.0, description="Audio classifier-free guidance scale"),
    frame_num: int = Form(81, description="Number of frames per clip (4n+1)"),
    motion_frame: int = Form(9, description="Driven frame length for long video generation"),
    max_frame_num: int = Form(1000, description="Maximum video length in frames"),
    mode: str = Form("streaming", description="Generation mode: 'clip' or 'streaming'"),
    n_prompt: Optional[str] = Form(None, description="Negative prompt"),
    color_correction_strength: float = Form(1.0, description="Color correction strength (0.0-1.0)"),
):
    """
    Submit a video generation task.
    
    Files can be uploaded directly OR local file paths can be provided.
    If both are provided, uploaded files take precedence.
    """
    # Create task first to get canonical task_id
    params = {}
    task_id = task_manager.create_task(params)
    task_dir = os.path.join(API_RESULTS_DIR, task_id, "input")
    os.makedirs(task_dir, exist_ok=True)

    # Save uploaded files and construct parameter dict
    params = {
        "task_mode": task_mode,
        "mode_selector": mode_selector,
        "prompt": prompt,
        "resolution": resolution,
        "sample_steps": sample_steps,
        "seed": seed,
        "text_guide_scale": text_guide_scale,
        "audio_guide_scale": audio_guide_scale,
        "frame_num": frame_num,
        "motion_frame": motion_frame,
        "max_frame_num": max_frame_num,
        "mode": mode,
        "n_prompt": n_prompt,
        "color_correction_strength": color_correction_strength,
        "tts_text": tts_text,
    }

    # Handle image/video input
    cond_video_path = None
    if image is not None:
        cond_video_path = _save_uploaded_file(image, task_dir, "input_image.png")
    elif image_path is not None:
        cond_video_path = image_path
    
    if task_mode == "VideoDubbing":
        if video is not None:
            cond_video_path = _save_uploaded_file(video, task_dir, "input_video.mp4")
        elif video_path is not None:
            cond_video_path = video_path
    
    params["cond_video"] = cond_video_path or ""

    # Handle audio files
    if audio_1 is not None:
        audio_1_path = _save_uploaded_file(audio_1, task_dir, "audio_1.wav")
        params["audio_1"] = audio_1_path
    elif audio_1_path is not None:
        params["audio_1"] = audio_1_path

    if audio_2 is not None:
        audio_2_path = _save_uploaded_file(audio_2, task_dir, "audio_2.wav")
        params["audio_2"] = audio_2_path
    elif audio_2_path is not None:
        params["audio_2"] = audio_2_path

    if voice_1_path:
        params["voice_1"] = voice_1_path
    if voice_2_path:
        params["voice_2"] = voice_2_path

    # Update task params and submit
    task = task_manager.get_task(task_id)
    task.params = params
    task.add_log(f"Task created. Mode: {task_mode}, Audio mode: {mode_selector}")

    # Initialize pipeline if needed (may take a while, so we do it in the background thread)
    try:
        _init_pipeline_if_needed(task)
    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error = f"Pipeline initialization failed: {e}"
        task.add_log(f"ERROR: Pipeline init failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Pipeline initialization failed: {e}", "task_id": task_id},
        )

    # Submit for background execution
    task_manager.submit_task(task_id)

    return {"task_id": task_id, "status": "queued"}


@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get the status and progress of a generation task."""
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return task.to_dict()


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a running or queued task."""
    success = task_manager.cancel_task(task_id)
    if not success:
        task = task_manager.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        raise HTTPException(status_code=400, detail=f"Task {task_id} cannot be cancelled (status: {task.status.value})")
    return {"task_id": task_id, "status": "cancelling"}


@app.get("/api/download/{task_id}")
async def download_result(task_id: str):
    """Download the generated video file."""
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    if task.status != TaskStatus.DONE:
        raise HTTPException(status_code=400, detail=f"Task {task_id} is not complete yet (status: {task.status.value})")
    if task.result_path is None or not os.path.exists(task.result_path):
        raise HTTPException(status_code=404, detail=f"Result file for task {task_id} not found")
    return FileResponse(
        task.result_path,
        media_type="video/mp4",
        filename=f"infinite_talk_{task_id}.mp4",
    )


@app.get("/api/tasks")
async def list_tasks(
    limit: int = Query(50, description="Maximum number of tasks to return"),
    offset: int = Query(0, description="Number of tasks to skip"),
    status_filter: Optional[str] = Query(None, description="Filter by status (queued/running/done/failed/cancelled)"),
):
    """List all tasks with optional filtering."""
    tasks = task_manager.list_tasks()
    if status_filter:
        tasks = [t for t in tasks if t["status"] == status_filter]
    tasks = tasks[offset:offset + limit]
    return {
        "total": len(task_manager._tasks),
        "limit": limit,
        "offset": offset,
        "tasks": tasks,
    }


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    server_args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )
    logging.info(f"Starting InfiniteTalk API server on {server_args.host}:{server_args.port}")
    logging.info(f"API results directory: {API_RESULTS_DIR}")
    uvicorn.run(
        "api_server:app",
        host=server_args.host,
        port=server_args.port,
        reload=False,
        log_level="info",
    )
