"""
Tests for api_server.py — InfiniteTalk REST API server.

Requires: pytest, fastapi, uvicorn, pydantic

Strategy:
- uvicorn (module-level import in api_server.py) is mocked before import.
- All GPU/ML dependencies (torch, PIL, librosa, wan) are lazy-imported inside
  functions, so they are mocked only in tests that exercise those code paths.
- parse_args() reads sys.argv → a global fixture isolates argv per test.
- API_RESULTS_DIR is redirected to a temporary directory.
"""

import json
import os
import sys
import threading
import time
import tempfile
from unittest.mock import ANY, MagicMock, PropertyMock, patch
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Mock module-level imports of api_server.py BEFORE importing it.
# uvicorn is imported at the top of api_server.py; torch/PIL/wan etc. are
# all lazy and do NOT need pre-mocking.
# ---------------------------------------------------------------------------
sys.modules["uvicorn"] = MagicMock()

# Now safe to import (no GPU deps triggered).
from api_server import (
    API_RESULTS_DIR as _ORIG_API_RESULTS_DIR,
    TaskInfo,
    TaskManager,
    TaskStatus,
    app,
    gpu_lock,
    parse_args,
    task_manager as global_task_manager,
)
from fastapi.testclient import TestClient


# ===================================================================
#  Fixtures
# ===================================================================


@pytest.fixture(autouse=True)
def isolate_argv():
    """Prevent parse_args() from reading leftover argv from other tests."""
    with patch.object(sys, "argv", ["api_server.py"]):
        yield


@pytest.fixture(autouse=True)
def temp_api_results(monkeypatch):
    """Redirect API_RESULTS_DIR to a temporary, isolated directory."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr("api_server.API_RESULTS_DIR", tmp)
        yield tmp


@pytest.fixture(autouse=True)
def clear_global_tasks():
    """Clear the module-level TaskManager between tests."""
    global_task_manager._tasks = {}
    yield
    global_task_manager._tasks = {}


@pytest.fixture
def fresh_tm():
    """Return a brand-new, isolated TaskManager instance."""
    return TaskManager()


@pytest.fixture
def client():
    """
    TestClient that mocks the GPU-heavy pipeline initialisation and
    task submission so tests never touch real ML dependencies.
    """
    with (
        patch("api_server._init_pipeline_if_needed") as mock_init,
        patch.object(global_task_manager, "submit_task") as mock_submit,
    ):
        mock_init.return_value = None
        mock_submit.return_value = None
        yield TestClient(app)


# ===================================================================
#  1 — 工具函数测试 (Utility function tests)
# ===================================================================


class TestParseArgs:
    """parse_args() — default and custom argument handling."""

    def test_defaults(self):
        """Defaults should match the module-level constants."""
        with patch.object(sys, "argv", ["api_server.py"]):
            args = parse_args()
        assert args.port == 8419
        assert args.host == "0.0.0.0"
        assert args.ulysses_size == 1
        assert args.ring_size == 1
        assert args.t5_fsdp is False
        assert args.t5_cpu is False
        assert args.dit_fsdp is False
        assert args.use_teacache is False
        assert args.use_apg is False
        assert args.frame_num == 81
        assert args.motion_frame == 9
        assert args.offload_model is None
        assert args.ckpt_dir is None
        assert args.lora_dir is None
        assert args.lora_scale == [1.2]
        assert args.motion_frame == 9
        assert args.frame_num == 81
        assert args.quant is None

    def test_custom_port_and_host(self):
        """Explicit --port and --host should override defaults."""
        with patch.object(sys, "argv", ["api_server.py", "--port", "9999", "--host", "127.0.0.1"]):
            args = parse_args()
        assert args.port == 9999
        assert args.host == "127.0.0.1"

    def test_boolean_flags(self):
        """Flags like --t5_fsdp should be True when present."""
        with patch.object(
            sys, "argv", ["api_server.py", "--t5_fsdp", "--dit_fsdp", "--t5_cpu"]
        ):
            args = parse_args()
        assert args.t5_fsdp is True
        assert args.dit_fsdp is True
        assert args.t5_cpu is True
        assert args.ulysses_size == 1

    def test_lora_list(self):
        """--lora_dir should accept multiple paths."""
        with patch.object(
            sys, "argv", ["api_server.py", "--lora_dir", "a.sft", "b.sft", "--lora_scale", "0.5", "1.5"]
        ):
            args = parse_args()
        assert args.lora_dir == ["a.sft", "b.sft"]
        assert args.lora_scale == [0.5, 1.5]

    def test_advanced_flags(self):
        """Teacache and APG flags."""
        with patch.object(
            sys,
            "argv",
            [
                "api_server.py",
                "--use_teacache",
                "--teacache_thresh",
                "0.3",
                "--use_apg",
                "--apg_momentum",
                "-0.5",
            ],
        ):
            args = parse_args()
        assert args.use_teacache is True
        assert args.teacache_thresh == 0.3
        assert args.use_apg is True
        assert args.apg_momentum == -0.5


class TestTaskInfoToDict:
    """TaskInfo.to_dict() serialisation."""

    def test_queued_task(self):
        """A queued task should serialise with status='queued' and null result_url."""
        info = TaskInfo("my-id", {"prompt": "hello"})
        d = info.to_dict()
        assert d["task_id"] == "my-id"
        assert d["status"] == "queued"
        assert d["progress"] == 0
        assert d["error"] is None
        assert d["result_url"] is None
        assert "created_at" in d
        assert "updated_at" in d
        assert isinstance(d["log"], list)

    def test_completed_task(self):
        """A completed task should have 100 progress and a download URL."""
        info = TaskInfo("done-id", {})
        info.status = TaskStatus.DONE
        info.progress = 100
        info.result_path = "/tmp/out.mp4"
        d = info.to_dict()
        assert d["status"] == "done"
        assert d["progress"] == 100
        assert d["result_url"] == f"/api/download/{info.task_id}"

    def test_log_truncation(self):
        """Only the last 100 log lines should be kept."""
        info = TaskInfo("log-id", {})
        for i in range(150):
            info.add_log(f"line {i}")
        d = info.to_dict()
        assert len(d["log"]) == 100
        assert d["log"][0].endswith("line 50")
        assert d["log"][-1].endswith("line 149")

    def test_error_state(self):
        """A failed task should carry error message."""
        info = TaskInfo("err-id", {})
        info.status = TaskStatus.FAILED
        info.error = "Out of memory"
        d = info.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "Out of memory"


# ===================================================================
#  2 — TaskManager tests
# ===================================================================


class TestTaskManager:
    """Unit tests for TaskManager operations."""

    def test_create_returns_unique_id(self, fresh_tm):
        """create_task should return a unique, non-empty string."""
        tid = fresh_tm.create_task({"prompt": "a"})
        assert isinstance(tid, str)
        assert len(tid) > 0

        # Consecutive calls must differ
        tid2 = fresh_tm.create_task({"prompt": "b"})
        assert tid != tid2

    def test_create_queued_status(self, fresh_tm):
        """A newly created task must have queued status."""
        tid = fresh_tm.create_task({"prompt": "x"})
        task = fresh_tm.get_task(tid)
        assert task is not None
        assert task.status == TaskStatus.QUEUED
        assert task.params == {"prompt": "x"}

    def test_get_nonexistent_returns_none(self, fresh_tm):
        """Querying an unknown id should return None."""
        assert fresh_tm.get_task("no-such-task") is None

    def test_cancel_queued_task(self, fresh_tm):
        """Cancelling a queued task sets cancel_event and marks CANCELLED."""
        tid = fresh_tm.create_task({})
        assert fresh_tm.cancel_task(tid) is True

        task = fresh_tm.get_task(tid)
        assert task.cancel_event.is_set()
        assert task.status == TaskStatus.CANCELLED

    def test_cancel_nonexistent_returns_false(self, fresh_tm):
        """Cancelling a non-existent task must return False."""
        assert fresh_tm.cancel_task("ghost") is False

    def test_cancel_twice_returns_false(self, fresh_tm):
        """Already-cancelled tasks cannot be cancelled again."""
        tid = fresh_tm.create_task({})
        fresh_tm.cancel_task(tid)
        assert fresh_tm.cancel_task(tid) is False

    def test_cancel_running_does_not_change_status_immediately(self, fresh_tm):
        """
        Cancelling a RUNNING task sets the event but does NOT change
        status immediately (status changes only after _run_task checks it).
        """
        tid = fresh_tm.create_task({})
        task = fresh_tm.get_task(tid)
        task.status = TaskStatus.RUNNING  # simulate mid-run

        assert fresh_tm.cancel_task(tid) is True
        # Event is set, but status remains RUNNING
        assert task.cancel_event.is_set()
        # It's still RUNNING until _run_task checks the event
        # (but in the API the status could also stay RUNNING until it checks)
        assert task.status == TaskStatus.RUNNING

    def test_list_empty(self, fresh_tm):
        """An empty manager should return an empty list."""
        assert fresh_tm.list_tasks() == []

    def test_list_returns_all(self, fresh_tm):
        """list_tasks should return a list of dicts with correct keys."""
        fresh_tm.create_task({"a": 1})
        fresh_tm.create_task({"b": 2})
        tasks = fresh_tm.list_tasks()
        assert len(tasks) == 2
        for t in tasks:
            assert isinstance(t, dict)
            assert "task_id" in t
            assert "status" in t
            assert "progress" in t
            assert "created_at" in t
            assert "updated_at" in t
            assert "result_url" in t

    def test_list_ordered_by_insertion(self, fresh_tm):
        """Tasks should appear in insertion order."""
        id_a = fresh_tm.create_task({"o": 1})
        id_b = fresh_tm.create_task({"o": 2})
        tasks = fresh_tm.list_tasks()
        assert tasks[0]["task_id"] == id_a
        assert tasks[1]["task_id"] == id_b

    def test_submit_and_status_transitions(self, fresh_tm):
        """
        After submit_task, status transitions through
        queued → running → completed.
        """
        tid = fresh_tm.create_task({"prompt": "transition"})
        assert fresh_tm.get_task(tid).status == TaskStatus.QUEUED

        # Mock _run_task to simulate the real lifecycle
        def mock_run(task_id):
            task = fresh_tm.get_task(task_id)
            if task is None:
                return
            task.status = TaskStatus.RUNNING
            task.add_log("Task started.")
            time.sleep(0.03)
            task.status = TaskStatus.DONE
            task.progress = 100
            task.add_log("Task completed.")

        with patch.object(fresh_tm, "_run_task", side_effect=mock_run):
            fresh_tm.submit_task(tid)
            # Brief pause for the daemon thread to execute
            time.sleep(0.15)

        task = fresh_tm.get_task(tid)
        assert task.status == TaskStatus.DONE, f"Expected DONE, got {task.status}"
        assert task.progress == 100, f"Expected progress 100, got {task.progress}"
        assert any("completed" in line for line in task.log)

    def test_submit_task_non_existent(self, fresh_tm):
        """submit_task on a non-existent id should not crash."""
        # Should just return without error
        fresh_tm.submit_task("does_not_exist")

    def test_set_pipeline(self, fresh_tm):
        """set_pipeline should store the pipeline reference."""
        assert fresh_tm._pipeline is None
        fresh_tm.set_pipeline("p", "a", "w", "e")
        assert fresh_tm._pipeline == "p"
        assert fresh_tm._pipeline_args == "a"
        assert fresh_tm._wav2vec_feature_extractor == "w"
        assert fresh_tm._audio_encoder == "e"


# ===================================================================
#  3 — FastAPI 端点测试 (Endpoint tests via TestClient)
# ===================================================================


class TestAPIEndpoints:
    """Integration tests for FastAPI endpoints using TestClient."""

    # ------------------------------------------------------------------
    # GET /api/health
    # ------------------------------------------------------------------

    def test_health_endpoint(self, client):
        """GET /api/health should return 200 with ok status."""
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "gpu_busy" in data
        assert "active_tasks" in data

    # ------------------------------------------------------------------
    # POST /api/generate
    # ------------------------------------------------------------------

    def test_create_task_success(self, client):
        """
        A valid POST /api/generate should return 200/202 with task_id
        and queued status.
        """
        resp = client.post(
            "/api/generate",
            data={
                "task_mode": "VideoDubbing",
                "mode_selector": "Single Person(Local File)",
                "prompt": "Test prompt",
                "resolution": "infinitetalk-480",
            },
        )
        assert resp.status_code in (200, 201, 202), f"Unexpected status: {resp.status_code}"
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        assert len(data["task_id"]) > 0

    def test_create_task_without_optional_params(self, client):
        """
        POST /api/generate with only required defaults should still
        create a task successfully.
        """
        resp = client.post("/api/generate")
        # All Form() params have defaults → should succeed.
        assert resp.status_code in (200, 201, 202), f"Unexpected status: {resp.status_code}"
        data = resp.json()
        assert "task_id" in data

    def test_create_task_with_tts_mode(self, client):
        """TTS mode should be accepted as a valid parameter combination."""
        resp = client.post(
            "/api/generate",
            data={
                "task_mode": "VideoDubbing",
                "mode_selector": "Single Person(TTS)",
                "prompt": "Hello world",
                "tts_text": "Testing TTS generation",
            },
        )
        assert resp.status_code in (200, 201, 202)
        data = resp.json()
        assert "task_id" in data

    def test_create_task_with_custom_parameters(self, client):
        """Custom generation parameters should be passed through."""
        resp = client.post(
            "/api/generate",
            data={
                "task_mode": "SingleImageDriven",
                "mode_selector": "Single Person(Local File)",
                "prompt": "Custom test",
                "resolution": "infinitetalk-720",
                "sample_steps": 25,
                "seed": 12345,
                "text_guide_scale": 5.0,
                "audio_guide_scale": 3.0,
                "frame_num": 161,
                "motion_frame": 17,
            },
        )
        assert resp.status_code in (200, 201, 202)
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "queued"

    # ------------------------------------------------------------------
    # GET /api/tasks/{task_id}
    # ------------------------------------------------------------------

    def test_get_task_not_found(self, client):
        """Querying a non-existent task should return 404."""
        resp = client.get("/api/tasks/non-existent-id-12345")
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data

    def test_get_task_status(self, client):
        """A known task should return its status and progress details."""
        # Create a task via the API first
        create_resp = client.post("/api/generate", data={"prompt": "status test"})
        assert create_resp.status_code in (200, 201, 202)
        task_id = create_resp.json()["task_id"]

        # Retrieve its status
        resp = client.get(f"/api/tasks/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == task_id
        assert data["status"] == "queued"
        assert data["progress"] == 0
        assert "created_at" in data
        assert "updated_at" in data

    def test_get_task_status_with_created_via_manager(self, client):
        """Tasks created through TaskManager directly should be visible."""
        tid = global_task_manager.create_task({"direct": True})
        resp = client.get(f"/api/tasks/{tid}")
        assert resp.status_code == 200
        assert resp.json()["task_id"] == tid

    # ------------------------------------------------------------------
    # GET /api/tasks
    # ------------------------------------------------------------------

    def test_list_tasks_empty(self, client):
        """GET /api/tasks on empty manager should return total 0."""
        resp = client.get("/api/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert data["total"] >= 0
        assert isinstance(data["tasks"], list)

    def test_list_tasks_with_filter(self, client):
        """Status filter should work correctly."""
        # Create a cancelled task to test filter
        tid = global_task_manager.create_task({})
        global_task_manager.get_task(tid).status = TaskStatus.CANCELLED

        tid2 = global_task_manager.create_task({})
        # Leave as queued

        resp = client.get("/api/tasks", params={"status_filter": "queued"})
        assert resp.status_code == 200
        data = resp.json()
        assert all(t["status"] == "queued" for t in data["tasks"])

    def test_list_tasks_limit_and_offset(self, client):
        """Pagination parameters should be respected."""
        for i in range(10):
            global_task_manager.create_task({"idx": i})

        resp = client.get("/api/tasks", params={"limit": 3, "offset": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 10
        assert len(data["tasks"]) == 3
        assert data["limit"] == 3
        assert data["offset"] == 2

    # ------------------------------------------------------------------
    # POST /api/tasks/{id}/cancel
    # ------------------------------------------------------------------

    def test_cancel_existing_queued_task(self, client):
        """Cancelling a queued task should succeed."""
        tid = global_task_manager.create_task({"test": True})
        resp = client.post(f"/api/tasks/{tid}/cancel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == tid
        assert data["status"] == "cancelling"

        # Verify the task is now cancelled
        task = global_task_manager.get_task(tid)
        assert task.status == TaskStatus.CANCELLED

    def test_cancel_non_existent_task(self, client):
        """Cancelling a non-existent task should return 404."""
        resp = client.post("/api/tasks/some-random-id/cancel")
        assert resp.status_code == 404

    def test_cancel_completed_task(self, client):
        """Cancelling a completed task should return 400."""
        tid = global_task_manager.create_task({})
        task = global_task_manager.get_task(tid)
        task.status = TaskStatus.DONE
        resp = client.post(f"/api/tasks/{tid}/cancel")
        assert resp.status_code == 400
        data = resp.json()
        assert "detail" in data

    # ------------------------------------------------------------------
    # GET /api/download/{task_id}
    # ------------------------------------------------------------------

    def test_download_not_found(self, client):
        """Downloading a non-existent task should return 404."""
        resp = client.get("/api/download/missing-task")
        assert resp.status_code == 404

    def test_download_not_complete(self, client):
        """Downloading a task that is not DONE should return 400."""
        tid = global_task_manager.create_task({})  # status=queued
        resp = client.get(f"/api/download/{tid}")
        assert resp.status_code == 400

    def test_download_missing_file(self, client):
        """
        Downloading a DONE task where the result file is missing
        should return 404.
        """
        tid = global_task_manager.create_task({})
        task = global_task_manager.get_task(tid)
        task.status = TaskStatus.DONE
        task.progress = 100
        # Do NOT set result_path (simulate missing file)
        resp = client.get(f"/api/download/{tid}")
        assert resp.status_code == 404

    def test_download_success(self, client, monkeypatch):
        """
        Downloading a completed task with a real result file should
        return the video file.
        """
        tid = global_task_manager.create_task({})
        task = global_task_manager.get_task(tid)
        task.status = TaskStatus.DONE
        task.progress = 100

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00\x00\x00 ftypmp42 mock video content")
            result_path = f.name

        task.result_path = result_path
        try:
            resp = client.get(f"/api/download/{tid}")
            assert resp.status_code == 200
            assert resp.headers.get("content-type", "").startswith("video/")
            assert "infinite_talk_" in resp.headers.get("content-disposition", "")
        finally:
            os.unlink(result_path)
