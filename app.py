# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
import sys
import json
import warnings
from datetime import datetime

import gradio as gr
warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image
import subprocess

import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.device_utils import (
    set_device, get_distributed_backend, is_cuda_device,
    is_mps_device, is_distributed as device_is_distributed,
)
from wan.utils.utils import cache_image, cache_video, str2bool
from wan.utils.multitalk_utils import save_video_ffmpeg
from kokoro import KPipeline
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model

import librosa
import pyloudnorm as pyln
import numpy as np
from einops import rearrange
import soundfile as sf
import re
import threading
import queue
import time
import shutil
import copy
from pathlib import Path

def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40

    if args.sample_shift is None:
        if args.size == 'infinitetalk-480':
            args.sample_shift = 7
        elif args.size == 'infinitetalk-720':
            args.sample_shift = 11
        else:
            raise NotImplementedError(f'Not supported size')

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, 99999999)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="infinitetalk-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="infinitetalk-480",
        choices=list(SIZE_CONFIGS.keys()),
        help="The buckget size of the generated video. The aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames to be generated in one clip. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default='./weights/Wan2.1-I2V-14B-480P',
        help="The path to the Wan checkpoint directory.")
    parser.add_argument(
        "--quant_dir",
        type=str,
        default=None,
        help="The path to the Wan quant checkpoint directory.")
    parser.add_argument(
        "--infinitetalk_dir",
        type=str,
        default='weights/InfiniteTalk/single/infinitetalk.safetensors',
        help="The path to the InfiniteTalk checkpoint directory.")
    parser.add_argument(
        "--wav2vec_dir",
        type=str,
        default='./weights/chinese-wav2vec2-base',
        help="The path to the wav2vec checkpoint directory.")
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="The path to the Wan checkpoint directory.")
    parser.add_argument(
        "--lora_dir",
        type=str,
        nargs='+',
        default=None,
        help="The path to the LoRA checkpoint directory.")
    parser.add_argument(
        "--lora_scale",
        type=float,
        nargs='+',
        default=[1.2],
        help="Controls how much to influence the outputs with the LoRA parameters. Accepts multiple float values."
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--audio_save_dir",
        type=str,
        default='save_audio/gradio',
        help="The path to save the audio embedding.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--input_json",
        type=str,
        default='examples.json',
        help="[meta file] The condition path to generate the video.")
    parser.add_argument(
        "--motion_frame",
        type=int,
        default=9,
        help="Driven frame length used in the mode of long video genration.")
    parser.add_argument(
        "--mode",
        type=str,
        default="streaming",
        choices=['clip', 'streaming'],
        help="clip: generate one video chunk, streaming: long video generation")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_text_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale for text control.")
    parser.add_argument(
        "--sample_audio_guide_scale",
        type=float,
        default=4.0,
        help="Classifier free guidance scale for audio control.")
    parser.add_argument(
        "--num_persistent_param_in_dit",
        type=int,
        default=None,
        required=False,
        help="Maximum parameter quantity retained in video memory, small number to reduce VRAM required",
    )
    parser.add_argument(
        "--use_teacache",
        action="store_true",
        default=False,
        help="Enable teacache for video generation."
    )
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="Threshold for teacache."
    )
    parser.add_argument(
        "--use_sparse_attention",
        action="store_true",
        default=False,
        help="Enable sparse attention for inference acceleration."
    )
    parser.add_argument(
        "--sparse_attention_ratio",
        type=float,
        default=0.5,
        help="Query token reduction ratio for sparse attention (0.0-1.0). Lower = faster."
    )
    parser.add_argument(
        "--use_apg",
        action="store_true",
        default=False,
        help="Enable adaptive projected guidance for video generation (APG)."
    )
    parser.add_argument(
        "--apg_momentum",
        type=float,
        default=-0.75,
        help="Momentum used in adaptive projected guidance (APG)."
    )
    parser.add_argument(
        "--apg_norm_threshold",
        type=float,
        default=55,
        help="Norm threshold used in adaptive projected guidance (APG)."
    )
    parser.add_argument(
        "--color_correction_strength",
        type=float,
        default=1.0,
        help="strength for color correction [0.0 -- 1.0]."
    )

    parser.add_argument(
        "--scene_seg",
        action="store_true",
        default=False,
        help="Enable scene segmentation for input video."
    )
    parser.add_argument(
        "--quant",
        type=str,
        default=None,
        help="Quantization type, must be 'int8' or 'fp8'."
    )
    args = parser.parse_args()
    _validate_args(args)
    return args


def custom_init(device, wav2vec):    
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True, attn_implementation="eager").to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
    return wav2vec_feature_extractor, audio_encoder

def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100 or np.isnan(loudness):
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio

def audio_prepare_multi(left_path, right_path, audio_type, sample_rate=16000):
    if left_path is None or right_path is None or left_path == 'None' or right_path == 'None':
        if left_path is None or left_path == 'None':
            human_speech_array2 = audio_prepare_single(right_path)
            human_speech_array1 = np.zeros(human_speech_array2.shape[0])
        elif right_path is None or right_path == 'None':
            human_speech_array1 = audio_prepare_single(left_path)
            human_speech_array2 = np.zeros(human_speech_array1.shape[0])
    else:
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = audio_prepare_single(right_path)

    if audio_type=='para':
        new_human_speech1 = human_speech_array1
        new_human_speech2 = human_speech_array2
    elif audio_type=='add':
        new_human_speech1 = np.concatenate([human_speech_array1[: human_speech_array1.shape[0]], np.zeros(human_speech_array2.shape[0])]) 
        new_human_speech2 = np.concatenate([np.zeros(human_speech_array1.shape[0]), human_speech_array2[:human_speech_array2.shape[0]]])
    sum_human_speechs = new_human_speech1 + new_human_speech2
    return new_human_speech1, new_human_speech2, sum_human_speechs

def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25 # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb

def extract_audio_from_video(filename, sample_rate):
    raw_audio_path = filename.split('/')[-1].split('.')[0]+'.wav'
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(filename),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "2",
        str(raw_audio_path),
    ]
    subprocess.run(ffmpeg_command, check=True)
    human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    os.remove(raw_audio_path)

    return human_speech_array

def audio_prepare_single(audio_path, sample_rate=16000):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ['.mp4', '.mov', '.avi', '.mkv']:
        human_speech_array = extract_audio_from_video(audio_path, sample_rate)
        return human_speech_array
    else:
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        return human_speech_array

def process_tts_single(text, save_dir, voice1):    
    s1_sentences = []

    pipeline = KPipeline(lang_code='a', repo_id='weights/Kokoro-82M')

    voice_tensor = torch.load(voice1, weights_only=True)
    generator = pipeline(
        text, voice=voice_tensor, # <= change voice here
        speed=1, split_pattern=r'\n+'
    )
    audios = []
    for i, (gs, ps, audio) in enumerate(generator):
        audios.append(audio)
    audios = torch.concat(audios, dim=0)
    s1_sentences.append(audios)
    s1_sentences = torch.concat(s1_sentences, dim=0)
    save_path1 =f'{save_dir}/s1.wav'
    sf.write(save_path1, s1_sentences, 24000) # save each audio file
    s1, _ = librosa.load(save_path1, sr=16000)
    return s1, save_path1
    
   

def process_tts_multi(text, save_dir, voice1, voice2):
    pattern = r'\(s(\d+)\)\s*(.*?)(?=\s*\(s\d+\)|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    s1_sentences = []
    s2_sentences = []

    pipeline = KPipeline(lang_code='a', repo_id='weights/Kokoro-82M')
    for idx, (speaker, content) in enumerate(matches):
        if speaker == '1':
            voice_tensor = torch.load(voice1, weights_only=True)
            generator = pipeline(
                content, voice=voice_tensor, # <= change voice here
                speed=1, split_pattern=r'\n+'
            )
            audios = []
            for i, (gs, ps, audio) in enumerate(generator):
                audios.append(audio)
            audios = torch.concat(audios, dim=0)
            s1_sentences.append(audios)
            s2_sentences.append(torch.zeros_like(audios))
        elif speaker == '2':
            voice_tensor = torch.load(voice2, weights_only=True)
            generator = pipeline(
                content, voice=voice_tensor, # <= change voice here
                speed=1, split_pattern=r'\n+'
            )
            audios = []
            for i, (gs, ps, audio) in enumerate(generator):
                audios.append(audio)
            audios = torch.concat(audios, dim=0)
            s2_sentences.append(audios)
            s1_sentences.append(torch.zeros_like(audios))
    
    s1_sentences = torch.concat(s1_sentences, dim=0)
    s2_sentences = torch.concat(s2_sentences, dim=0)
    sum_sentences = s1_sentences + s2_sentences
    save_path1 =f'{save_dir}/s1.wav'
    save_path2 =f'{save_dir}/s2.wav'
    save_path_sum = f'{save_dir}/sum.wav'
    sf.write(save_path1, s1_sentences, 24000) # save each audio file
    sf.write(save_path2, s2_sentences, 24000)
    sf.write(save_path_sum, sum_sentences, 24000)

    s1, _ = librosa.load(save_path1, sr=16000)
    s2, _ = librosa.load(save_path2, sr=16000)
    # sum, _ = librosa.load(save_path_sum, sr=16000)
    return s1, s2, save_path_sum


# ============================================================
# Enhanced Features: Task Queue, History, Presets, I18n
# ============================================================


class I18nManager:
    """Internationalization manager for Chinese/English UI"""

    _strings = {
        # Application
        "app.title": ["MeiGen-InfiniteTalk", "MeiGen-InfiniteTalk"],
        "app.subtitle": [
            "InfiniteTalk: 基于音频驱动的隔帧视频配音生成",
            "InfiniteTalk: Audio-driven Video Generation for Spare-Frame Video Dubbing."
        ],
        # Tabs
        "tab.generate": ["生成", "Generate"],
        "tab.queue": ["任务队列", "Task Queue"],
        "tab.history": ["历史记录", "History"],
        "tab.presets": ["参数预设", "Presets"],
        "tab.settings": ["设置", "Settings"],
        # Task mode
        "task_mode.label": ["选择任务模式", "Choose task mode"],
        "task_mode.single": ["单图驱动", "SingleImageDriven"],
        "task_mode.dubbing": ["视频配音", "VideoDubbing"],
        "vid_input": ["上传输入视频", "Upload Input Video"],
        "img_input": ["上传输入图片", "Upload Input Image"],
        "prompt": ["提示词", "Prompt"],
        "prompt.ph": ["描述你想要生成的视频内容", "Describe the video you want to generate"],
        # Audio
        "audio.title": ["音频选项", "Audio Options"],
        "mode.label": ["选择音频模式", "Select audio mode"],
        "mode.single_file": ["单人(本地文件)", "Single Person(Local File)"],
        "mode.single_tts": ["单人(TTS)", "Single Person(TTS)"],
        "mode.multi_file_add": ["多人(本地文件, 叠加)", "Multi Person(Local File, audio add)"],
        "mode.multi_file_para": ["多人(本地文件, 并行)", "Multi Person(Local File, audio parallel)"],
        "mode.multi_tts": ["多人(TTS)", "Multi Person(TTS)"],
        "resolution.label": ["选择分辨率", "Select resolution"],
        "audio.speaker1": ["说话人1的音频", "Conditioning Audio for speaker 1"],
        "audio.speaker2": ["说话人2的音频", "Conditioning Audio for speaker 2"],
        "tts.text": ["TTS文本", "Text for TTS"],
        "tts.ph": ["参考示例格式输入文本", "Refer to the format in the examples"],
        # Advanced
        "advanced.title": ["高级参数", "Advanced Options"],
        "advanced.steps": ["扩散步数", "Diffusion steps"],
        "advanced.seed": ["随机种子", "Seed"],
        "advanced.text_guide": ["文本引导系数", "Text Guide scale"],
        "advanced.audio_guide": ["音频引导系数", "Audio Guide scale"],
        "advanced.voice1": ["左声道说话人音色", "Voice for the left person"],
        "advanced.voice2": ["右声道说话人音色", "Voice for right person"],
        "advanced.n_prompt": ["负面提示词", "Negative Prompt"],
        "advanced.n_prompt.ph": ["描述你不想在视频中看到的内容", "Describe elements to exclude from generation"],
        # Buttons
        "btn.generate": ["添加到队列", "Add to Queue"],
        "btn.immediate": ["立即生成", "Generate Now"],
        "btn.refresh": ["刷新", "Refresh"],
        "btn.clear_completed": ["清除已完成", "Clear Completed"],
        "btn.cancel": ["取消", "Cancel"],
        "btn.view": ["查看", "View"],
        "btn.delete": ["删除", "Delete"],
        "btn.download": ["下载", "Download"],
        "btn.save_preset": ["保存预设", "Save Preset"],
        "btn.load_preset": ["加载预设", "Load Preset"],
        # Queue
        "queue.title": ["任务队列管理", "Task Queue Management"],
        "queue.id": ["任务ID", "Task ID"],
        "queue.status": ["状态", "Status"],
        "queue.progress": ["进度", "Progress"],
        "queue.stage": ["阶段", "Stage"],
        "queue.created": ["创建时间", "Created"],
        "queue.empty": ["暂无任务", "No tasks"],
        "queue.status.queued": ["排队中", "Queued"],
        "queue.status.running": ["运行中", "Running"],
        "queue.status.completed": ["已完成", "Completed"],
        "queue.status.failed": ["失败", "Failed"],
        # History
        "history.title": ["生成历史记录", "Generation History"],
        "history.prompt": ["提示词", "Prompt"],
        "history.resolution": ["分辨率", "Resolution"],
        "history.result": ["结果视频", "Result Video"],
        "history.created": ["完成时间", "Completed"],
        "history.empty": ["暂无历史记录", "No history yet"],
        # Presets
        "presets.title": ["参数预设管理", "Parameter Presets"],
        "presets.name": ["预设名称", "Preset Name"],
        "presets.name.ph": ["输入预设名称...", "Enter preset name..."],
        "presets.current": ["当前参数", "Current Parameters"],
        "presets.saved": ["已保存的预设", "Saved Presets"],
        "presets.empty": ["暂无预设", "No presets yet"],
        "presets.loaded": ["已加载预设: ", "Loaded preset: "],
        "presets.saved_msg": ["预设已保存: ", "Preset saved: "],
        "presets.deleted_msg": ["预设已删除: ", "Preset deleted: "],
        # Settings
        "settings.language": ["语言", "Language"],
        "settings.lang.zh": ["中文", "Chinese"],
        "settings.lang.en": ["English", "English"],
        # Status
        "status.processing": ["处理中...", "Processing..."],
        "status.ready": ["就绪", "Ready"],
        "status.queued_items": ["队列中的任务: ", "Queued tasks: "],
        # Progress stages
        "progress.prepare": ["准备音频...", "Preparing audio..."],
        "progress.extract": ["提取音频特征...", "Extracting audio features..."],
        "progress.generating": ["生成视频中...", "Generating video..."],
        "progress.saving": ["保存视频...", "Saving video..."],
        "progress.done": ["完成!", "Done!"],
        # Errors
        "error.no_image": ["请上传输入图片或视频", "Please upload an input image or video"],
        "error.no_audio": ["请上传条件音频", "Please upload conditioning audio"],
        "error.no_prompt": ["请输入提示词", "Please enter a prompt"],
        "error.generation": ["生成失败: ", "Generation failed: "],
        "error.unknown": ["未知错误", "Unknown error"],
        # Examples
        "examples.label": ["示例", "Examples"],
        # Result
        "result.label": ["生成结果", "Generated Video"],
        "result.latest": ["最新结果", "Latest Result"],
        "task_added": ["任务已添加到队列: ", "Task added to queue: "],
    }

    def __init__(self, lang="zh"):
        self._lang = lang if lang in ("zh", "en") else "zh"

    def t(self, key):
        """Get translated string for the current language"""
        vals = self._strings.get(key, [key, key])
        return vals[0] if self._lang == "zh" else vals[1]

    def set_lang(self, lang):
        self._lang = lang

    def get_lang(self):
        return self._lang


class HistoryManager:
    """Manages generation history using JSON file storage"""

    def __init__(self, workspace_dir):
        self.history_dir = os.path.join(workspace_dir, "history")
        self.videos_dir = os.path.join(self.history_dir, "videos")
        self.history_file = os.path.join(self.history_dir, "history.json")
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _load(self):
        if not os.path.exists(self.history_file):
            return []
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save(self, entries):
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    def add_entry(self, entry):
        """Add a history entry and return its index"""
        with self._lock:
            entries = self._load()
            entries.insert(0, entry)  # newest first
            # Keep max 200 entries
            if len(entries) > 200:
                # Remove old video files
                for old in entries[200:]:
                    old_path = old.get("result_path", "")
                    if old_path and os.path.exists(old_path):
                        try:
                            os.remove(old_path)
                        except OSError:
                            pass
                entries = entries[:200]
            self._save(entries)
        return 0

    def get_entries(self, limit=100):
        with self._lock:
            entries = self._load()
            return entries[:limit]

    def get_entry(self, index):
        with self._lock:
            entries = self._load()
            if 0 <= index < len(entries):
                return entries[index]
        return None

    def delete_entry(self, index):
        with self._lock:
            entries = self._load()
            if 0 <= index < len(entries):
                entry = entries.pop(index)
                # Remove the video file
                result_path = entry.get("result_path", "")
                if result_path and os.path.exists(result_path):
                    try:
                        os.remove(result_path)
                    except OSError:
                        pass
                self._save(entries)
                return True
        return False

    def get_video_path(self, index):
        entry = self.get_entry(index)
        if entry and entry.get("result_path"):
            path = entry["result_path"]
            if os.path.exists(path):
                return path
        return None


class PresetManager:
    """Manages parameter presets using JSON file storage"""

    def __init__(self, workspace_dir):
        self.presets_dir = os.path.join(workspace_dir, "presets")
        os.makedirs(self.presets_dir, exist_ok=True)

    def list_presets(self):
        """List all saved presets"""
        presets = []
        if not os.path.exists(self.presets_dir):
            return presets
        for fname in sorted(os.listdir(self.presets_dir)):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(self.presets_dir, fname), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        presets.append({
                            "name": fname[:-5],
                            "file": fname,
                            "data": data
                        })
                except (json.JSONDecodeError, IOError):
                    pass
        return presets

    def save_preset(self, name, params):
        """Save a parameter preset"""
        safe_name = name.replace(" ", "_").replace("/", "_")[:100]
        filepath = os.path.join(self.presets_dir, f"{safe_name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        return safe_name

    def load_preset(self, name):
        """Load a parameter preset"""
        safe_name = name.replace(" ", "_").replace("/", "_")[:100]
        filepath = os.path.join(self.presets_dir, f"{safe_name}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def delete_preset(self, name):
        """Delete a parameter preset"""
        safe_name = name.replace(" ", "_").replace("/", "_")[:100]
        filepath = os.path.join(self.presets_dir, f"{safe_name}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    def get_preset_names(self):
        """Get list of preset names"""
        return [p["name"] for p in self.list_presets()]


class TaskManager:
    """Thread-based task queue manager for sequential batch processing"""

    def __init__(self):
        self._queue = queue.Queue()
        self._tasks = {}  # task_id -> task_info dict
        self._lock = threading.RLock()  # 使用可重入锁，避免 _worker_loop 中调用 _set_status 时死锁
        self._counter = 0
        self._worker = None
        self._running = False

    def add_task(self, params):
        """Add a task to the queue. Returns the task ID."""
        with self._lock:
            self._counter += 1
            task_id = f"T{int(time.time())}-{self._counter}"
            self._tasks[task_id] = {
                "id": task_id,
                "status": "queued",
                "progress": 0,
                "stage": "等待中...",
                "params": params,
                "result_path": None,
                "error": None,
                "traceback": None,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "completed_at": None,
            }
            self._queue.put(task_id)
        return task_id

    def start_worker(self, target_func):
        """Start the background worker thread"""
        if self._worker and self._worker.is_alive():
            return
        self._running = True
        self._worker = threading.Thread(
            target=self._worker_loop,
            args=(target_func,),
            daemon=True,
            name="TaskQueueWorker"
        )
        self._worker.start()

    def stop_worker(self):
        """Stop the background worker"""
        self._running = False

    def _worker_loop(self, target_func):
        """Main worker loop - processes tasks one by one"""
        while self._running:
            try:
                task_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            with self._lock:
                if task_id not in self._tasks:
                    self._queue.task_done()
                    continue
                self._set_status(task_id, "running")
                self._tasks[task_id]["stage"] = "准备中..."

            try:
                # Run the actual generation
                result_path, history_entry = target_func(task_id, self._update_progress)

                with self._lock:
                    self._set_status(task_id, "completed")
                    self._tasks[task_id]["progress"] = 100
                    self._tasks[task_id]["stage"] = "完成"
                    self._tasks[task_id]["result_path"] = result_path
                    self._tasks[task_id]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self._tasks[task_id]["history_entry"] = history_entry
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                with self._lock:
                    self._set_status(task_id, "failed")
                    self._tasks[task_id]["error"] = str(e)
                    self._tasks[task_id]["traceback"] = tb
                    self._tasks[task_id]["stage"] = "失败"
                    self._tasks[task_id]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logging.error(f"Task {task_id} failed: {e}")
                logging.error(tb)
            finally:
                self._queue.task_done()

    # Valid state transitions
    _VALID_TRANSITIONS = {
        "queued": ["running", "failed"],
        "running": ["completed", "failed"],
        "completed": [],
        "failed": [],
    }

    def _update_progress(self, task_id, progress, stage=""):
        """Update progress for a task (called from worker thread)"""
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["progress"] = max(0, min(progress, 99))
                if stage:
                    self._tasks[task_id]["stage"] = stage

    def _set_status(self, task_id, new_status):
        """Set task status with validity check"""
        with self._lock:
            if task_id not in self._tasks:
                return False
            current = self._tasks[task_id]["status"]
            valid_next = self._VALID_TRANSITIONS.get(current, [])
            if new_status not in valid_next:
                logging.warning(f"Invalid status transition: {current} -> {new_status} for task {task_id}")
                return False
            self._tasks[task_id]["status"] = new_status
            return True

    def get_task(self, task_id):
        """Get task info by ID"""
        with self._lock:
            return copy.deepcopy(self._tasks.get(task_id))

    def get_all_tasks(self):
        """Get all tasks"""
        with self._lock:
            return [copy.deepcopy(t) for t in self._tasks.values()]

    def get_task_count(self, status=None):
        """Count tasks, optionally filtered by status"""
        with self._lock:
            if status:
                return sum(1 for t in self._tasks.values() if t["status"] == status)
            return len(self._tasks)

    def clear_completed(self):
        """Remove completed/failed tasks from the dict"""
        with self._lock:
            to_remove = [k for k, v in self._tasks.items() if v["status"] in ("completed", "failed")]
            for k in to_remove:
                del self._tasks[k]
            return len(to_remove)


def run_graio_demo(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        if is_cuda_device():
            set_device(local_rank)
        dist.init_process_group(
            backend=get_distributed_backend(),
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        # Gracefully handle CUDA-only features on non-CUDA devices
        if not is_cuda_device():
            if args.t5_fsdp or args.dit_fsdp:
                logging.warning("FSDP requires CUDA. Disabling FSDP for non-CUDA device.")
                args.t5_fsdp = False
                args.dit_fsdp = False
            if args.ulysses_size > 1 or args.ring_size > 1:
                logging.warning("Context parallelism requires CUDA/xfuser. Disabling for non-CUDA device.")
                args.ulysses_size = 1
                args.ring_size = 1
        else:
            assert not (
                args.t5_fsdp or args.dit_fsdp
            ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
            assert not (
                args.ulysses_size > 1 or args.ring_size > 1
            ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        try:
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )
            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size())
            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=args.ring_size,
                ulysses_degree=args.ulysses_size,
            )
        except ImportError:
            logging.warning("xfuser not available, running without context parallel")
            args.ulysses_size = 1
            args.ring_size = 1


    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_available() and dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    assert args.task == "infinitetalk-14B", 'You should choose infinitetalk in args.task.'

    wav2vec_feature_extractor, audio_encoder= custom_init('cpu', args.wav2vec_dir)
    os.makedirs(args.audio_save_dir,exist_ok=True)

    logging.info("Creating MultiTalk pipeline.")
    wan_i2v = wan.InfiniteTalkPipeline(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
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
        infinitetalk_dir=args.infinitetalk_dir
    )

    if args.num_persistent_param_in_dit is not None:
        wan_i2v.vram_management = True
        wan_i2v.enable_vram_management(
            num_persistent_param_in_dit=args.num_persistent_param_in_dit
        )

    # ========== Initialize enhanced features ==========
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    task_manager = TaskManager()
    history_manager = HistoryManager(workspace_dir)
    preset_manager = PresetManager(workspace_dir)
    i18n = I18nManager()

    # Load language preference
    lang_file = os.path.join(workspace_dir, ".lang_pref.json")
    if os.path.exists(lang_file):
        try:
            with open(lang_file, "r") as f:
                pref = json.load(f)
                i18n.set_lang(pref.get("lang", "zh"))
        except:
            pass

    latest_result_lock = threading.Lock()
    latest_result_data = {"path": None}

    # ========== Enhanced generate_video with progress & error handling ==========
    def generate_video(img2vid_image, vid2vid_vid, task_mode, img2vid_prompt, n_prompt,
                       img2vid_audio_1, img2vid_audio_2, sd_steps, seed, text_guide_scale,
                       audio_guide_scale, mode_selector, tts_text, resolution_select,
                       human1_voice, human2_voice, progress_callback=None, task_id=None):
        """
        Core generation function. Supports both direct Gradio calls and task queue.
        Returns: (video_path, history_entry_dict)
        """
        input_data = {}
        input_data["prompt"] = img2vid_prompt
        if task_mode == 'VideoDubbing':
            input_data["cond_video"] = vid2vid_vid
        else:
            input_data["cond_video"] = img2vid_image
        person = {}

        if mode_selector == "SingleFile":
            person['person1'] = img2vid_audio_1
        elif mode_selector == "SingleTTS":
            tts_audio = {}
            tts_audio['text'] = tts_text
            tts_audio['human1_voice'] = human1_voice
            input_data["tts_audio"] = tts_audio
        elif mode_selector in ("MultiFileAdd", "MultiFilePara"):
            person['person1'] = img2vid_audio_1
            person['person2'] = img2vid_audio_2
            input_data["audio_type"] = 'add' if mode_selector == "MultiFileAdd" else 'para'
        else:
            tts_audio = {}
            tts_audio['text'] = tts_text
            tts_audio['human1_voice'] = human1_voice
            tts_audio['human2_voice'] = human2_voice
            input_data["tts_audio"] = tts_audio

        input_data["cond_audio"] = person

        def _progress(stage_key, pct):
            if progress_callback:
                progress_callback(pct, stage_key)

        try:
            _progress("progress.prepare", 10)

            if mode_selector in ("SingleFile", "MultiFileAdd", "MultiFilePara"):
                if len(input_data['cond_audio']) == 2:
                    new_human_speech1, new_human_speech2, sum_human_speechs = audio_prepare_multi(input_data['cond_audio']['person1'], input_data['cond_audio']['person2'], input_data['audio_type'])
                    audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
                    audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder)
                    emb1_path = os.path.join(args.audio_save_dir, '1.pt')
                    emb2_path = os.path.join(args.audio_save_dir, '2.pt')
                    sum_audio = os.path.join(args.audio_save_dir, 'sum.wav')
                    sf.write(sum_audio, sum_human_speechs, 16000)
                    torch.save(audio_embedding_1, emb1_path)
                    torch.save(audio_embedding_2, emb2_path)
                    input_data['cond_audio']['person1'] = emb1_path
                    input_data['cond_audio']['person2'] = emb2_path
                    input_data['video_audio'] = sum_audio
                elif len(input_data['cond_audio']) == 1:
                    human_speech = audio_prepare_single(input_data['cond_audio']['person1'])
                    audio_embedding = get_embedding(human_speech, wav2vec_feature_extractor, audio_encoder)
                    emb_path = os.path.join(args.audio_save_dir, '1.pt')
                    sum_audio = os.path.join(args.audio_save_dir, 'sum.wav')
                    sf.write(sum_audio, human_speech, 16000)
                    torch.save(audio_embedding, emb_path)
                    input_data['cond_audio']['person1'] = emb_path
                    input_data['video_audio'] = sum_audio
            elif mode_selector in ("SingleTTS", "MultiTTS"):
                if 'human2_voice' not in input_data['tts_audio'].keys():
                    new_human_speech1, sum_audio = process_tts_single(input_data['tts_audio']['text'], args.audio_save_dir, input_data['tts_audio']['human1_voice'])
                    audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
                    emb1_path = os.path.join(args.audio_save_dir, '1.pt')
                    torch.save(audio_embedding_1, emb1_path)
                    input_data['cond_audio']['person1'] = emb1_path
                    input_data['video_audio'] = sum_audio
                else:
                    new_human_speech1, new_human_speech2, sum_audio = process_tts_multi(input_data['tts_audio']['text'], args.audio_save_dir, input_data['tts_audio']['human1_voice'], input_data['tts_audio']['human2_voice'])
                    audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
                    audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder)
                    emb1_path = os.path.join(args.audio_save_dir, '1.pt')
                    emb2_path = os.path.join(args.audio_save_dir, '2.pt')
                    torch.save(audio_embedding_1, emb1_path)
                    torch.save(audio_embedding_2, emb2_path)
                    input_data['cond_audio']['person1'] = emb1_path
                    input_data['cond_audio']['person2'] = emb2_path
                    input_data['video_audio'] = sum_audio

            _progress("progress.extract", 30)

            logging.info("Generating video ...")
            _progress("progress.generating", 40)
            video = wan_i2v.generate_infinitetalk(
                input_data,
                size_buckget=resolution_select,
                motion_frame=args.motion_frame,
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sampling_steps=sd_steps,
                text_guide_scale=text_guide_scale,
                audio_guide_scale=audio_guide_scale,
                seed=seed,
                n_prompt=n_prompt,
                offload_model=args.offload_model,
                max_frames_num=args.frame_num if args.mode == 'clip' else 1000,
                color_correction_strength=args.color_correction_strength,
                extra_args=args,
            )

            _progress("progress.saving", 90)
            save_file = args.save_file
            if save_file is None:
                formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                formatted_prompt = input_data['prompt'].replace(" ", "_").replace("/", "_")[:50]
                save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform == 'win32' else args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}"

            logging.info(f"Saving generated video to {save_file}.mp4")
            save_video_ffmpeg(video, save_file, [input_data['video_audio']], high_quality_save=False)
            logging.info("Finished.")

            video_path = save_file + '.mp4'
            history_entry = {
                "task_id": task_id or "",
                "prompt": img2vid_prompt,
                "task_mode": task_mode,
                "mode_selector": mode_selector,
                "resolution": resolution_select,
                "seed": seed,
                "sd_steps": sd_steps,
                "text_guide_scale": text_guide_scale,
                "audio_guide_scale": audio_guide_scale,
                "result_path": video_path,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            _progress("progress.done", 100)
            return video_path, history_entry

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logging.error(f"Generation failed [task={task_id}]: {e}")
            logging.error(tb)
            error_entry = {
                "task_id": task_id or "",
                "prompt": img2vid_prompt,
                "task_mode": task_mode,
                "mode_selector": mode_selector,
                "resolution": resolution_select,
                "seed": seed,
                "result_path": None,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
            }
            raise  # Re-raise to let the task manager handle it

    # ========== Task Queue Worker ==========
    def _task_runner(task_id, update_progress):
        """Called from TaskManager worker thread"""
        with task_manager._lock:
            if task_id not in task_manager._tasks:
                return None, None
            params = task_manager._tasks[task_id]["params"]

        def prog_cb(pct, stage):
            update_progress(task_id, pct, stage)

        video_path, history_entry = generate_video(
            params["img2vid_image"],
            params["vid2vid_vid"],
            params["task_mode"],
            params["img2vid_prompt"],
            params["n_prompt"],
            params["img2vid_audio_1"],
            params["img2vid_audio_2"],
            params["sd_steps"],
            params["seed"],
            params["text_guide_scale"],
            params["audio_guide_scale"],
            params["mode_selector"],
            params["tts_text"],
            params["resolution_select"],
            params["human1_voice"],
            params["human2_voice"],
            progress_callback=prog_cb,
            task_id=task_id,
        )

        # Save to history
        if history_entry:
            # Copy video to history folder
            if video_path and os.path.exists(video_path):
                import shutil
                safe_name = f"{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                history_video_path = os.path.join(history_manager.videos_dir, safe_name)
                try:
                    shutil.copy2(video_path, history_video_path)
                    history_entry["result_path"] = history_video_path
                except (shutil.Error, IOError):
                    history_entry["result_path"] = video_path
            history_manager.add_entry(history_entry)

        # Update latest result for the UI
        with latest_result_lock:
            latest_result_data["path"] = video_path

        return video_path, history_entry

    # ========== Gradio Handlers ==========
    def submit_to_queue(img2vid_image, vid2vid_vid, task_mode, img2vid_prompt, n_prompt,
                        img2vid_audio_1, img2vid_audio_2, sd_steps, seed, text_guide_scale,
                        audio_guide_scale, mode_selector, tts_text, resolution_select,
                        human1_voice, human2_voice):
        """Add task to queue and return immediately"""
        params = {
            "img2vid_image": img2vid_image,
            "vid2vid_vid": vid2vid_vid,
            "task_mode": task_mode,
            "img2vid_prompt": img2vid_prompt,
            "n_prompt": n_prompt,
            "img2vid_audio_1": img2vid_audio_1,
            "img2vid_audio_2": img2vid_audio_2,
            "sd_steps": sd_steps,
            "seed": seed,
            "text_guide_scale": text_guide_scale,
            "audio_guide_scale": audio_guide_scale,
            "mode_selector": mode_selector,
            "tts_text": tts_text,
            "resolution_select": resolution_select,
            "human1_voice": human1_voice,
            "human2_voice": human2_voice,
        }
        task_id = task_manager.add_task(params)
        task_manager.start_worker(_task_runner)
        return i18n.t("task_added") + task_id

    def generate_now(img2vid_image, vid2vid_vid, task_mode, img2vid_prompt, n_prompt,
                     img2vid_audio_1, img2vid_audio_2, sd_steps, seed, text_guide_scale,
                     audio_guide_scale, mode_selector, tts_text, resolution_select,
                     human1_voice, human2_voice, progress=gr.Progress()):
        """Immediate generation with Gradio progress bar"""
        def prog_cb(pct, stage_key):
            desc = i18n.t(stage_key) if hasattr(i18n, 't') else stage_key
            progress(pct / 100.0, desc=desc)

        video_path, history_entry = generate_video(
            img2vid_image, vid2vid_vid, task_mode, img2vid_prompt, n_prompt,
            img2vid_audio_1, img2vid_audio_2, sd_steps, seed, text_guide_scale,
            audio_guide_scale, mode_selector, tts_text, resolution_select,
            human1_voice, human2_voice,
            progress_callback=prog_cb, task_id="immediate"
        )

        # Save to history
        if history_entry:
            if video_path and os.path.exists(video_path):
                safe_name = f"immediate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                history_video_path = os.path.join(history_manager.videos_dir, safe_name)
                try:
                    shutil.copy2(video_path, history_video_path)
                    history_entry["result_path"] = history_video_path
                except (shutil.Error, IOError):
                    pass
            history_manager.add_entry(history_entry)

        with latest_result_lock:
            latest_result_data["path"] = video_path
        return video_path

    def get_latest_video():
        """Get the latest completed video path"""
        with latest_result_lock:
            path = latest_result_data["path"]
            if path and os.path.exists(path):
                return path
        return None

    def _build_table_html(headers, rows_data, empty_msg=None):
        """Build an HTML table string (replaces gr.Dataframe to avoid Gradio 6 Svelte bug)."""
        if empty_msg:
            return f'<div style="padding:16px;text-align:center;color:#888;font-size:14px;">{empty_msg}</div>'
        header_cells = "".join(
            f'<th style="padding:8px 12px;border-bottom:2px solid #e5e7eb;text-align:left;font-weight:600;font-size:13px;white-space:nowrap;">{h}</th>'
            for h in headers
        )
        row_html = ""
        for i, cells in enumerate(rows_data):
            bg = "background:#f9fafb;" if i % 2 == 1 else ""
            row_html += "<tr style='" + bg + "'>" + "".join(
                f'<td style="padding:6px 12px;border-bottom:1px solid #e5e7eb;font-size:13px;">{c}</td>'
                for c in cells
            ) + "</tr>"
        return f'''<div style="overflow-x:auto;border:1px solid #e5e7eb;border-radius:8px;">
<table style="width:100%;border-collapse:collapse;font-size:14px;">
<thead><tr>{header_cells}</tr></thead>
<tbody>{row_html}</tbody>
</table></div>'''

    def build_task_df():
        """Build HTML table from task list"""
        tasks = task_manager.get_all_tasks()
        if not tasks:
            return _build_table_html(
                [i18n.t("queue.id"), i18n.t("queue.status"), i18n.t("queue.progress"), i18n.t("queue.created")],
                [], empty_msg=i18n.t("queue.empty")
            )

        rows_data = []
        headers = [i18n.t("queue.id"), i18n.t("queue.status"), i18n.t("queue.progress"), i18n.t("queue.stage"), i18n.t("queue.created")]
        for t in reversed(tasks):
            if t["status"] == "queued":
                status_text = i18n.t("queue.status.queued")
            elif t["status"] == "running":
                status_text = i18n.t("queue.status.running")
            elif t["status"] == "completed":
                status_text = i18n.t("queue.status.completed")
            elif t["status"] == "failed":
                status_text = i18n.t("queue.status.failed")
            else:
                status_text = t["status"]
            prog = f"{t['progress']}%"
            rows_data.append([t["id"], status_text, prog, t["stage"], t["created_at"]])
        return _build_table_html(headers, rows_data)

    def build_history_df():
        """Build HTML table from history"""
        entries = history_manager.get_entries(50)
        if not entries:
            return _build_table_html(
                [i18n.t("history.prompt"), i18n.t("history.resolution"), i18n.t("history.created")],
                [], empty_msg=i18n.t("history.empty")
            )

        rows_data = []
        headers = ["#", i18n.t("history.prompt")[:30], i18n.t("history.resolution"), i18n.t("history.created")]
        for idx, e in enumerate(entries):
            prompt_short = e.get("prompt", "")[:40]
            if len(e.get("prompt", "")) > 40:
                prompt_short += "..."
            rows_data.append([str(idx), prompt_short, e.get("resolution", ""), e.get("created_at", "")])
        return _build_table_html(headers, rows_data)

    def get_history_video(idx):
        """Get video path from history entry"""
        try:
            idx = int(idx)
            path = history_manager.get_video_path(idx)
            return path if path else None
        except (ValueError, IndexError):
            return None

    def delete_history_entry(idx):
        try:
            idx = int(idx)
            history_manager.delete_entry(idx)
            return i18n.t("btn.delete") + " OK"
        except (ValueError, IndexError):
            return "Error"

    def save_preset_handler(name, sd_steps_val, seed_val, text_guide_val, audio_guide_val):
        """Save current params as a preset"""
        if not name.strip():
            return i18n.t("presets.save") + "... " + i18n.t("error.no_prompt")
        preset_data = {
            "sd_steps": sd_steps_val,
            "seed": seed_val,
            "text_guide_scale": text_guide_val,
            "audio_guide_scale": audio_guide_val,
        }
        preset_manager.save_preset(name.strip(), preset_data)
        return i18n.t("presets.saved_msg") + name.strip()

    def load_preset_handler(name):
        """Load a preset and return parameter updates"""
        if not name:
            return [gr.update()] * 8
        data = preset_manager.load_preset(name)
        if data is None:
            return [gr.update()] * 8
        return [
            gr.update(value=data.get("sd_steps", 8)),
            gr.update(value=data.get("seed", 42)),
            gr.update(value=data.get("text_guide_scale", 1.0)),
            gr.update(value=data.get("audio_guide_scale", 2.0)),
            gr.update(value=data.get("resolution_select", "infinitetalk-480")),
            gr.update(value=data.get("n_prompt", "")),
            gr.update(value=data.get("human1_voice", "weights/Kokoro-82M/voices/am_adam.pt")),
            gr.update(value=data.get("human2_voice", "weights/Kokoro-82M/voices/af_heart.pt")),
        ]

    def delete_preset_handler(name):
        if not name:
            return "No preset selected"
        preset_manager.delete_preset(name)
        return i18n.t("presets.deleted_msg") + name

    def refresh_preset_list():
        names = preset_manager.get_preset_names()
        if not names:
            return gr.Dropdown(choices=[], value=None, label=i18n.t("presets.saved"))
        return gr.Dropdown(choices=names, value=names[0] if names else None, label=i18n.t("presets.saved"))

    # ========== Language handler ==========
    def handle_lang_change(lang, task_val, mode_val):
        lang_code = "zh" if lang == "中文" else "en"
        i18n.set_lang(lang_code)
        try:
            with open(lang_file, "w") as f:
                json.dump({"lang": lang_code}, f)
        except (OSError, IOError):
            pass  # Non-critical; UI still updates correctly
        # 统一使用 gr.update() 返回，不使用 mix 模式
        return [
            gr.update(value=f"{i18n.t('settings.language')}: {lang}"),
            gr.update(value=f"**{i18n.t('settings.language')}**"),
            gr.update(value=f"**{i18n.t('task_mode.label')}**"),
            gr.update(
                choices=["SingleImageDriven", "VideoDubbing"],
                value=task_val,
            ),
            gr.update(value=f"**{i18n.t('mode.label')}**"),
            gr.update(
                choices=[
                    "SingleFile",
                    "SingleTTS",
                    "MultiFileAdd",
                    "MultiFilePara",
                    "MultiTTS",
                ],
                value=mode_val,
            ),
            gr.update(value=f"**{i18n.t('resolution.label')}**"),
        ]

    # ========== Existing UI helpers ==========
    def toggle_audio_mode(mode):
        if 'TTS' in mode:
            return [
                gr.Audio(visible=False, interactive=False),
                gr.Audio(visible=False, interactive=False),
                gr.Textbox(visible=True, interactive=True)
            ]
        elif 'Single' in mode:
            return [
                gr.Audio(visible=True, interactive=True),
                gr.Audio(visible=False, interactive=False),
                gr.Textbox(visible=False, interactive=False)
            ]
        else:
            return [
                gr.Audio(visible=True, interactive=True),
                gr.Audio(visible=True, interactive=True),
                gr.Textbox(visible=False, interactive=False)
            ]

    def show_upload(mode):
        if mode == "SingleImageDriven":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    # ========== UI Layout ==========
    with gr.Blocks(title="MeiGen-InfiniteTalk") as demo:

        gr.Markdown(f"""
                    <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                        MeiGen-InfiniteTalk
                    </div>
                    <div style="text-align: center; font-size: 16px; font-weight: normal; margin-bottom: 20px;">
                        {i18n.t("app.subtitle")}
                    </div>
                    <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
                        <a href=''><img src='https://img.shields.io/badge/Project-Page-blue'></a>
                        <a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
                        <a href=''><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
                    </div>
                    """)

        with gr.Row():
            lang_label = gr.Markdown(f"**{i18n.t('settings.language')}**")
            with gr.Column(scale=0, min_width=200):
                lang_radio = gr.Dropdown(
                    choices=["中文", "English"],
                    value="中文" if i18n.get_lang() == "zh" else "English",
                    show_label=False,
                    interactive=True,
                    scale=0,
                    min_width=120,
                )
            lang_status = gr.Textbox(
                label="",
                value="",
                visible=True,
                interactive=False,
                scale=2,
                show_label=False,
                container=False,
            )

        with gr.Tabs():
            # ======== TAB 1: GENERATE ========
            with gr.TabItem(i18n.t("tab.generate")):
                with gr.Row():
                    with gr.Column(scale=1):
                        task_mode_label = gr.Markdown(f"**{i18n.t('task_mode.label')}**")
                        task_mode = gr.Radio(
                            choices=["SingleImageDriven", "VideoDubbing"],
                            show_label=False,
                            value="VideoDubbing"
                        )
                        vid2vid_vid = gr.Video(
                            label=i18n.t("vid_input"),
                            visible=True)
                        img2vid_image = gr.Image(
                            type="filepath",
                            label=i18n.t("img_input"),
                            elem_id="image_upload",
                            visible=False
                        )
                        img2vid_prompt = gr.Textbox(
                            label=i18n.t("prompt"),
                            placeholder=i18n.t("prompt.ph"),
                        )
                        task_mode.change(
                            fn=show_upload,
                            inputs=task_mode,
                            outputs=[img2vid_image, vid2vid_vid]
                        )

                        with gr.Accordion(i18n.t("audio.title"), open=True):
                            mode_label = gr.Markdown(f"**{i18n.t('mode.label')}**")
                            mode_selector = gr.Radio(
                                choices=[
                                    "SingleFile",
                                    "SingleTTS",
                                    "MultiFileAdd",
                                    "MultiFilePara",
                                    "MultiTTS",
                                ],
                                show_label=False,
                                value="SingleFile"
                            )
                            resolution_label = gr.Markdown(f"**{i18n.t('resolution.label')}**")
                            resolution_select = gr.Radio(
                                choices=["infinitetalk-480", "infinitetalk-720"],
                                show_label=False,
                                value="infinitetalk-480"
                            )
                            img2vid_audio_1 = gr.Audio(
                                label=i18n.t("audio.speaker1"),
                                type="filepath",
                                visible=True
                            )
                            img2vid_audio_2 = gr.Audio(
                                label=i18n.t("audio.speaker2"),
                                type="filepath",
                                visible=False
                            )
                            tts_text = gr.Textbox(
                                label=i18n.t("tts.text"),
                                placeholder=i18n.t("tts.ph"),
                                visible=False,
                                interactive=False
                            )
                            mode_selector.change(
                                fn=toggle_audio_mode,
                                inputs=mode_selector,
                                outputs=[img2vid_audio_1, img2vid_audio_2, tts_text]
                            )

                        with gr.Accordion(i18n.t("advanced.title"), open=False):
                            with gr.Row():
                                sd_steps = gr.Slider(
                                    label=i18n.t("advanced.steps"),
                                    minimum=1,
                                    maximum=1000,
                                    value=8,
                                    step=1)
                                seed = gr.Slider(
                                    label=i18n.t("advanced.seed"),
                                    minimum=-1,
                                    maximum=2147483647,
                                    step=1,
                                    value=42)
                            with gr.Row():
                                text_guide_scale = gr.Slider(
                                    label=i18n.t("advanced.text_guide"),
                                    minimum=0,
                                    maximum=20,
                                    value=1.0,
                                    step=1)
                                audio_guide_scale = gr.Slider(
                                    label=i18n.t("advanced.audio_guide"),
                                    minimum=0,
                                    maximum=20,
                                    value=2.0,
                                    step=1)
                            with gr.Row():
                                human1_voice = gr.Textbox(
                                    label=i18n.t("advanced.voice1"),
                                    value="weights/Kokoro-82M/voices/am_adam.pt",
                                )
                                human2_voice = gr.Textbox(
                                    label=i18n.t("advanced.voice2"),
                                    value="weights/Kokoro-82M/voices/af_heart.pt"
                                )
                            n_prompt = gr.Textbox(
                                label=i18n.t("advanced.n_prompt"),
                                placeholder=i18n.t("advanced.n_prompt.ph"),
                                value="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
                            )

                        with gr.Row():
                            add_queue_btn = gr.Button(i18n.t("btn.generate"), variant="primary")
                            generate_now_btn = gr.Button(i18n.t("btn.immediate"))

                        queue_status_text = gr.Textbox(
                            label="",
                            value=i18n.t("status.ready"),
                            interactive=False,
                            show_label=False,
                            container=False,
                        )

                    with gr.Column(scale=2):
                        result_gallery = gr.Video(
                            label=i18n.t("result.label"),
                            interactive=False,
                            height=600,
                        )

                        gr.Examples(
                            examples=[
                                ['SingleImageDriven', 'examples/single/ref_image.png', None,
                                 "A woman is passionately singing into a professional microphone in a recording studio. She wears large black headphones and a dark cardigan over a gray top. Her long, wavy brown hair frames her face as she looks slightly upwards, her mouth open mid-song. The studio is equipped with various audio equipment, including a mixing console and a keyboard, with soundproofing panels on the walls. The lighting is warm and focused on her, creating a professional and intimate atmosphere. A close-up shot captures her expressive performance.",
                                 "SingleFile", "examples/single/1.wav", None, None],
                                ['VideoDubbing', None, 'examples/single/ref_video.mp4',
                                 "A man is talking", "SingleFile", "examples/single/1.wav", None, None],
                            ],
                            inputs=[task_mode, img2vid_image, vid2vid_vid, img2vid_prompt,
                                    mode_selector, img2vid_audio_1, img2vid_audio_2, tts_text],
                        )

                # Button handlers
                add_queue_btn.click(
                    fn=submit_to_queue,
                    inputs=[img2vid_image, vid2vid_vid, task_mode, img2vid_prompt, n_prompt,
                            img2vid_audio_1, img2vid_audio_2, sd_steps, seed, text_guide_scale,
                            audio_guide_scale, mode_selector, tts_text, resolution_select,
                            human1_voice, human2_voice],
                    outputs=[queue_status_text],
                )

                generate_now_btn.click(
                    fn=generate_now,
                    inputs=[img2vid_image, vid2vid_vid, task_mode, img2vid_prompt, n_prompt,
                            img2vid_audio_1, img2vid_audio_2, sd_steps, seed, text_guide_scale,
                            audio_guide_scale, mode_selector, tts_text, resolution_select,
                            human1_voice, human2_voice],
                    outputs=[result_gallery],
                )

            # ======== TAB 2: TASK QUEUE ========
            with gr.TabItem(i18n.t("tab.queue")):
                gr.Markdown(f"### {i18n.t('queue.title')}")
                with gr.Row():
                    refresh_queue_btn = gr.Button(i18n.t("btn.refresh"))
                    clear_completed_btn = gr.Button(i18n.t("btn.clear_completed"))
                queue_df = gr.HTML(
                    value=_build_table_html(
                        [i18n.t("queue.id"), i18n.t("queue.status"), i18n.t("queue.progress"), i18n.t("queue.stage"), i18n.t("queue.created")],
                        [], empty_msg=i18n.t("queue.empty")
                    ),
                )

                refresh_queue_btn.click(
                    fn=build_task_df,
                    inputs=[],
                    outputs=[queue_df],
                )
                clear_completed_btn.click(
                    fn=lambda: task_manager.clear_completed() or build_task_df(),
                    inputs=[],
                    outputs=[queue_df],
                )

            # ======== TAB 3: HISTORY ========
            with gr.TabItem(i18n.t("tab.history")):
                gr.Markdown(f"### {i18n.t('history.title')}")
                with gr.Row():
                    refresh_history_btn = gr.Button(i18n.t("btn.refresh"))
                history_df = gr.HTML(
                    value=_build_table_html(
                        ["#", i18n.t("history.prompt")[:30], i18n.t("history.resolution"), i18n.t("history.created")],
                        [], empty_msg=i18n.t("history.empty")
                    ),
                )
                with gr.Row():
                    history_idx = gr.Textbox(
                        label="#",
                        placeholder="0",
                        scale=0,
                        min_width=80,
                    )
                    view_history_btn = gr.Button(i18n.t("btn.view"))
                    delete_history_btn = gr.Button(i18n.t("btn.delete"))
                history_video = gr.Video(
                    label=i18n.t("history.result"),
                    interactive=False,
                    height=400,
                )

                refresh_history_btn.click(
                    fn=build_history_df,
                    inputs=[],
                    outputs=[history_df],
                )
                view_history_btn.click(
                    fn=get_history_video,
                    inputs=[history_idx],
                    outputs=[history_video],
                )
                def _delete_and_refresh(idx):
                    delete_history_entry(idx)
                    return build_history_df()

                delete_history_btn.click(
                    fn=_delete_and_refresh,
                    inputs=[history_idx],
                    outputs=[history_df],
                )

            # ======== TAB 4: PRESETS ========
            with gr.TabItem(i18n.t("tab.presets")):
                gr.Markdown(f"### {i18n.t('presets.title')}")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"**{i18n.t('presets.current')}**")
                        gr.Markdown(f"{i18n.t('advanced.steps')}: 8 | {i18n.t('advanced.seed')}: 42 | {i18n.t('advanced.text_guide')}: 1.0 | {i18n.t('advanced.audio_guide')}: 2.0")
                        with gr.Row():
                            preset_name = gr.Textbox(
                                label=i18n.t("presets.name"),
                                placeholder=i18n.t("presets.name.ph"),
                                scale=2,
                            )
                            save_preset_btn = gr.Button(i18n.t("btn.save_preset"), scale=0)
                        preset_save_status = gr.Textbox(
                            label="",
                            value="",
                            interactive=False,
                            show_label=False,
                            container=False,
                        )
                    with gr.Column():
                        preset_dropdown = gr.Dropdown(
                            choices=preset_manager.get_preset_names(),
                            label=i18n.t("presets.saved"),
                            interactive=True,
                        )
                        with gr.Row():
                            load_preset_btn = gr.Button(i18n.t("btn.load_preset"))
                            delete_preset_btn = gr.Button(i18n.t("btn.delete"))
                        preset_load_status = gr.Textbox(
                            label="",
                            value="",
                            interactive=False,
                            show_label=False,
                            container=False,
                        )

                save_preset_btn.click(
                    fn=save_preset_handler,
                    inputs=[preset_name, sd_steps, seed, text_guide_scale, audio_guide_scale],
                    outputs=[preset_save_status],
                ).then(
                    fn=refresh_preset_list,
                    inputs=[],
                    outputs=[preset_dropdown],
                )

                load_preset_btn.click(
                    fn=load_preset_handler,
                    inputs=[preset_dropdown],
                    outputs=[sd_steps, seed, text_guide_scale, audio_guide_scale,
                             resolution_select, n_prompt, human1_voice, human2_voice],
                )

                delete_preset_btn.click(
                    fn=delete_preset_handler,
                    inputs=[preset_dropdown],
                    outputs=[preset_load_status],
                ).then(
                    fn=refresh_preset_list,
                    inputs=[],
                    outputs=[preset_dropdown],
                )

        # Refresh on page load (no periodic polling to avoid overwriting instant results)
        demo.load(
            fn=get_latest_video,
            inputs=[],
            outputs=[result_gallery],
        )

        # Language switch handler (bound after all UI components are created)
        # 使用 .input() 事件而非 .change()，避免 Gradio 6 的 Radio change 事件级联 bug
        lang_radio.input(
            fn=handle_lang_change,
            inputs=[lang_radio, task_mode, mode_selector],
            outputs=[lang_status, lang_label, task_mode_label, task_mode, mode_label, mode_selector, resolution_label],
            concurrency_limit=None,
        )

    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", debug=True, server_port=8418, show_error=True)

        


if __name__ == "__main__":
    args = _parse_args()
    run_graio_demo(args)
    
