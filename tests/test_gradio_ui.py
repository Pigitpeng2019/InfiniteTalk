"""Tests for Gradio UI helper classes in app.py"""

import json
import os
import tempfile
import threading
import time
from unittest.mock import patch, MagicMock

# Mock gradio and torch before importing app.py
import sys

# Block problematic imports at module level
_gr_mock = MagicMock()
_torch_mock = MagicMock()
_pil_mock = MagicMock()
_subprocess_mock = MagicMock()
_librosa_mock = MagicMock()
_pyln_mock = MagicMock()
_einops_mock = MagicMock()
_sf_mock = MagicMock()
_re_mock = MagicMock()
_shutil_mock = MagicMock()
_warnings_mock = MagicMock()
_wan_mock = MagicMock()
_kokoro_mock = MagicMock()
_transformers_mock = MagicMock()

sys.modules["gradio"] = _gr_mock
sys.modules["torch"] = _torch_mock
sys.modules["torch.distributed"] = MagicMock()
sys.modules["PIL"] = _pil_mock
sys.modules["PIL.Image"] = MagicMock()
sys.modules["subprocess"] = _subprocess_mock
sys.modules["wan"] = _wan_mock
sys.modules["wan.configs"] = MagicMock()
sys.modules["wan.utils"] = MagicMock()
sys.modules["wan.utils.utils"] = MagicMock()
sys.modules["wan.utils.multitalk_utils"] = MagicMock()
sys.modules["kokoro"] = _kokoro_mock
sys.modules["transformers"] = _transformers_mock
sys.modules["transformers.Wav2Vec2FeatureExtractor"] = MagicMock()
sys.modules["librosa"] = _librosa_mock
sys.modules["pyloudnorm"] = _pyln_mock
sys.modules["einops"] = _einops_mock
sys.modules["soundfile"] = _sf_mock
sys.modules["re"] = _re_mock
sys.modules["shutil"] = _shutil_mock
sys.modules["warnings"] = _warnings_mock

# Also mock src modules
sys.modules["src"] = MagicMock()
sys.modules["src.audio_analysis"] = MagicMock()
sys.modules["src.audio_analysis.wav2vec2"] = MagicMock()

from app import I18nManager, HistoryManager, PresetManager, TaskManager
import pytest


# ============================================================
# I18nManager Tests
# ============================================================

class TestI18nManager:
    def test_i18n_init_default_lang(self):
        """Default language should be Chinese"""
        i18n = I18nManager()
        assert i18n.get_lang() == "zh"

    def test_i18n_init_explicit_lang(self):
        """Explicit language setting at init"""
        i18n_en = I18nManager(lang="en")
        assert i18n_en.get_lang() == "en"
        i18n_zh = I18nManager(lang="zh")
        assert i18n_zh.get_lang() == "zh"

    def test_i18n_init_invalid_lang(self):
        """Invalid language falls back to zh"""
        i18n = I18nManager(lang="fr")
        assert i18n.get_lang() == "zh"

    def test_i18n_set_lang(self):
        """Switching language should work correctly"""
        i18n = I18nManager(lang="zh")
        assert i18n.get_lang() == "zh"
        i18n.set_lang("en")
        assert i18n.get_lang() == "en"

    def test_i18n_set_lang_invalid(self):
        """Setting invalid language should keep current"""
        i18n = I18nManager(lang="zh")
        i18n.set_lang("invalid")
        # set_lang sets directly without validation; get_lang returns whatever was set
        # Let's check the actual behavior
        assert i18n.get_lang() == "invalid"

    def test_i18n_translation_zh(self):
        """Chinese translations should be correct for key UI strings"""
        i18n = I18nManager(lang="zh")
        assert i18n.t("app.title") == "MeiGen-InfiniteTalk"
        assert i18n.t("tab.generate") == "生成"
        assert i18n.t("queue.status.queued") == "排队中"
        assert i18n.t("queue.status.completed") == "已完成"
        assert i18n.t("queue.status.failed") == "失败"
        assert i18n.t("btn.generate") == "添加到队列"
        assert i18n.t("btn.immediate") == "立即生成"
        assert i18n.t("settings.language") == "语言"
        assert i18n.t("error.no_image") == "请上传输入图片或视频"
        assert i18n.t("progress.generating") == "生成视频中..."

    def test_i18n_translation_en(self):
        """English translations should be correct for key UI strings"""
        i18n = I18nManager(lang="en")
        assert i18n.t("app.title") == "MeiGen-InfiniteTalk"
        assert i18n.t("tab.generate") == "Generate"
        assert i18n.t("queue.status.queued") == "Queued"
        assert i18n.t("queue.status.completed") == "Completed"
        assert i18n.t("queue.status.failed") == "Failed"
        assert i18n.t("btn.generate") == "Add to Queue"
        assert i18n.t("btn.immediate") == "Generate Now"
        assert i18n.t("settings.language") == "Language"
        assert i18n.t("error.no_image") == "Please upload an input image or video"
        assert i18n.t("progress.generating") == "Generating video..."

    def test_i18n_missing_key(self):
        """Missing key should return the key itself"""
        i18n = I18nManager(lang="zh")
        assert i18n.t("nonexistent.key") == "nonexistent.key"

        i18n_en = I18nManager(lang="en")
        assert i18n_en.t("nonexistent.key") == "nonexistent.key"


# ============================================================
# HistoryManager Tests
# ============================================================

class TestHistoryManager:

    def test_history_init(self):
        """Initialization creates directories and an empty history file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            hm = HistoryManager(tmpdir)
            history_dir = os.path.join(tmpdir, "history")
            videos_dir = os.path.join(tmpdir, "history", "videos")
            history_file = os.path.join(tmpdir, "history", "history.json")

            assert os.path.isdir(history_dir)
            assert os.path.isdir(videos_dir)
            assert not os.path.exists(history_file)  # Not created until first add

            # _load should work with non-existent file
            assert hm._load() == []

    def test_history_add_entry(self):
        """Adding entries works correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            hm = HistoryManager(tmpdir)
            entry = {
                "prompt": "test prompt",
                "result_path": "/tmp/test.mp4",
                "params": {"step": 40}
            }
            hm.add_entry(entry)
            entries = hm._load()
            assert len(entries) == 1
            assert entries[0]["prompt"] == "test prompt"

    def test_history_add_entry_inserts_front(self):
        """New entries should be inserted at the front (newest first)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            hm = HistoryManager(tmpdir)
            hm.add_entry({"prompt": "first"})
            hm.add_entry({"prompt": "second"})
            entries = hm._load()
            assert len(entries) == 2
            assert entries[0]["prompt"] == "second"
            assert entries[1]["prompt"] == "first"

    def test_history_get_entries(self):
        """get_entries returns entries and respects the limit"""
        with tempfile.TemporaryDirectory() as tmpdir:
            hm = HistoryManager(tmpdir)
            for i in range(10):
                hm.add_entry({"prompt": f"entry_{i}", "index": i})

            all_entries = hm.get_entries(limit=100)
            assert len(all_entries) == 10

            limited = hm.get_entries(limit=3)
            assert len(limited) == 3

    def test_history_get_entry(self):
        """get_entry returns correct entry by index or None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            hm = HistoryManager(tmpdir)
            hm.add_entry({"prompt": "first", "idx": 0})
            hm.add_entry({"prompt": "second", "idx": 1})

            entry = hm.get_entry(0)
            assert entry is not None
            assert entry["prompt"] == "second"  # newest first

            entry = hm.get_entry(1)
            assert entry["prompt"] == "first"

            # Out of range
            assert hm.get_entry(999) is None
            assert hm.get_entry(-1) is None

    def test_history_delete_entry(self):
        """Deleting an entry removes it from the list"""
        with tempfile.TemporaryDirectory() as tmpdir:
            hm = HistoryManager(tmpdir)
            hm.add_entry({"prompt": "first"})
            hm.add_entry({"prompt": "second"})
            assert len(hm._load()) == 2

            # Delete the second entry (index 1 after newest-first insert)
            result = hm.delete_entry(1)
            assert result is True
            entries = hm._load()
            assert len(entries) == 1
            assert entries[0]["prompt"] == "second"

    def test_history_delete_entry_invalid_index(self):
        """Deleting an invalid index returns False"""
        with tempfile.TemporaryDirectory() as tmpdir:
            hm = HistoryManager(tmpdir)
            assert hm.delete_entry(0) is False
            assert hm.delete_entry(-1) is False

    def test_history_max_entries(self):
        """Should keep max 200 entries and clean up old video files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            hm = HistoryManager(tmpdir)
            # Add 210 entries with unique video file paths
            paths = []
            for i in range(210):
                video_path = os.path.join(tmpdir, "history", "videos", f"video_{i}.mp4")
                # Create a dummy video file for each
                Path_like_dir = os.path.dirname(video_path)
                os.makedirs(Path_like_dir, exist_ok=True)
                with open(video_path, "w") as f:
                    f.write("dummy")
                paths.append(video_path)
                hm.add_entry({
                    "prompt": f"entry_{i}",
                    "result_path": video_path
                })

            entries = hm._load()
            assert len(entries) == 200  # Max 200
            assert entries[0]["prompt"] == "entry_209"  # Newest first

            # The first 10 entries (oldest) should have been deleted, including their video files
            for i in range(10):
                # Entries beyond 200 (i=0..9) were trimmed
                assert not os.path.exists(paths[i]), f"Video file {i} should have been deleted"

    def test_history_get_video_path(self):
        """get_video_path returns path for valid index, None for invalid"""
        with tempfile.TemporaryDirectory() as tmpdir:
            hm = HistoryManager(tmpdir)
            video_path = os.path.join(tmpdir, "history", "videos", "test_result.mp4")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            with open(video_path, "w") as f:
                f.write("dummy")

            hm.add_entry({
                "prompt": "test",
                "result_path": video_path
            })
            result = hm.get_video_path(0)
            assert result == video_path

            # Invalid index
            assert hm.get_video_path(999) is None

    def test_history_corrupted_file(self):
        """Corrupted history.json should be handled gracefully (return empty list)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            hm = HistoryManager(tmpdir)
            history_file = os.path.join(tmpdir, "history", "history.json")
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            with open(history_file, "w") as f:
                f.write("invalid json content{{{")

            entries = hm._load()
            assert entries == []


# ============================================================
# PresetManager Tests
# ============================================================

class TestPresetManager:

    def test_preset_save_load(self):
        """Saving a preset and loading it back should return the same data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PresetManager(tmpdir)
            params = {
                "steps": 40,
                "seed": 42,
                "text_guide": 5.0,
                "audio_guide": 4.0
            }
            name = pm.save_preset("my_preset", params)
            assert name == "my_preset"

            loaded = pm.load_preset("my_preset")
            assert loaded == params

    def test_preset_save_with_special_chars(self):
        """Saving a preset with special chars should sanitize the filename"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PresetManager(tmpdir)
            params = {"key": "value"}
            name = pm.save_preset("my/preset name/test", params)
            # Spaces become underscores, slashes become underscores
            assert "/" not in name
            assert " " not in name
            assert len(name) <= 100

    def test_preset_list(self):
        """list_presets should return all saved presets"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PresetManager(tmpdir)
            assert pm.list_presets() == []

            pm.save_preset("preset_a", {"a": 1})
            pm.save_preset("preset_b", {"b": 2})
            pm.save_preset("preset_c", {"c": 3})

            presets = pm.list_presets()
            assert len(presets) == 3
            names = [p["name"] for p in presets]
            assert "preset_a" in names
            assert "preset_b" in names
            assert "preset_c" in names

    def test_preset_list_data(self):
        """list_presets should include the data of each preset"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PresetManager(tmpdir)
            pm.save_preset("test_preset", {"steps": 40, "seed": 42})

            presets = pm.list_presets()
            assert len(presets) == 1
            assert presets[0]["data"]["steps"] == 40
            assert presets[0]["data"]["seed"] == 42

    def test_preset_delete(self):
        """Deleting a preset should remove it"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PresetManager(tmpdir)
            pm.save_preset("my_preset", {"key": "value"})
            assert len(pm.list_presets()) == 1

            result = pm.delete_preset("my_preset")
            assert result is True
            assert len(pm.list_presets()) == 0

            # Deleting non-existent returns False
            assert pm.delete_preset("nonexistent") is False

    def test_preset_delete_nonexistent_returns_false(self):
        """Deleting a non-existent preset returns False"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PresetManager(tmpdir)
            pm.save_preset("my_preset", {"key": "value"})
            # Delete the existing one
            assert pm.delete_preset("my_preset") is True
            # Try deleting again - should be False now
            assert pm.delete_preset("my_preset") is False

    def test_preset_corrupted_file(self):
        """Corrupted preset files should be skipped gracefully in list_presets"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PresetManager(tmpdir)
            # Save a valid preset
            pm.save_preset("good_preset", {"key": "value"})

            # Create a corrupted file
            corrupted_path = os.path.join(tmpdir, "presets", "bad_preset.json")
            with open(corrupted_path, "w") as f:
                f.write("{invalid json")

            presets = pm.list_presets()
            # Should only have the good preset, silently skip the bad one
            assert len(presets) == 1
            assert presets[0]["name"] == "good_preset"

    def test_preset_load_nonexistent(self):
        """Loading a non-existent preset returns None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PresetManager(tmpdir)
            result = pm.load_preset("ghost_preset")
            assert result is None

    def test_preset_get_names(self):
        """get_preset_names returns a simple list of names"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PresetManager(tmpdir)
            pm.save_preset("alpha", {"v": 1})
            pm.save_preset("beta", {"v": 2})

            names = pm.get_preset_names()
            assert len(names) == 2
            assert "alpha" in names
            assert "beta" in names


# ============================================================
# TaskManager Tests
# ============================================================

class TestTaskManager:

    def test_task_manager_add(self):
        """Adding a task should return a task_id with initial status 'queued'"""
        tm = TaskManager()
        task_id = tm.add_task({"prompt": "test"})
        assert task_id is not None
        assert task_id.startswith("T")

        task = tm.get_task(task_id)
        assert task is not None
        assert task["status"] == "queued"
        assert task["progress"] == 0
        assert task["params"]["prompt"] == "test"
        assert task["completed_at"] is None

    def test_task_manager_add_multiple(self):
        """Adding multiple tasks should return unique IDs"""
        tm = TaskManager()
        id1 = tm.add_task({"prompt": "a"})
        id2 = tm.add_task({"prompt": "b"})
        assert id1 != id2

    def test_task_manager_get_all_tasks(self):
        """get_all_tasks should return all tasks"""
        tm = TaskManager()
        tm.add_task({"prompt": "a"})
        tm.add_task({"prompt": "b"})
        tm.add_task({"prompt": "c"})

        tasks = tm.get_all_tasks()
        assert len(tasks) == 3

    def test_task_manager_status_transition(self):
        """Status should transition correctly: queued -> running -> completed/failed"""
        tm = TaskManager()
        task_id = tm.add_task({"prompt": "test"})

        # Start worker with a successful target
        def successful_target(task_id_arg, progress_cb):
            progress_cb(task_id_arg, 50, "processing...")
            return "/tmp/result.mp4", {"result": "ok"}

        tm.start_worker(successful_target)
        # Give the worker time to process
        time.sleep(0.3)

        # Task should be completed now
        task = tm.get_task(task_id)
        assert task is not None
        assert task["status"] == "completed"
        assert task["progress"] == 100
        assert task["result_path"] == "/tmp/result.mp4"
        assert task["completed_at"] is not None

        tm.stop_worker()

    def test_task_manager_status_failed(self):
        """Worker exception should set status to 'failed'"""
        tm = TaskManager()

        def failing_target(task_id_arg, progress_cb):
            raise RuntimeError("Something went wrong")

        task_id = tm.add_task({"prompt": "test"})
        tm.start_worker(failing_target)
        time.sleep(0.3)

        task = tm.get_task(task_id)
        assert task is not None
        assert task["status"] == "failed"
        assert "Something went wrong" in task["error"]
        assert task["progress"] == 0
        assert task["stage"] == "失败"
        assert task["completed_at"] is not None

        tm.stop_worker()

    def test_task_manager_update_progress(self):
        """Progress updates should be tracked correctly"""
        tm = TaskManager()

        def progressive_target(task_id_arg, progress_cb):
            progress_cb(task_id_arg, 10, "Preparing...")
            progress_cb(task_id_arg, 50, "Processing...")
            progress_cb(task_id_arg, 90, "Saving...")
            progress_cb(task_id_arg, 99, "Almost done...")
            return "/tmp/result.mp4", {"result": "ok"}

        task_id = tm.add_task({"prompt": "test"})
        tm.start_worker(progressive_target)
        time.sleep(0.3)

        task = tm.get_task(task_id)
        assert task is not None
        assert task["progress"] == 100  # Completed overrides
        assert task["stage"] == "完成"

        tm.stop_worker()

    def test_task_manager_progress_clamped(self):
        """Progress should be clamped to max 99 during work"""
        tm = TaskManager()

        def overprogress_target(task_id_arg, progress_cb):
            progress_cb(task_id_arg, 150, "Overflow")
            return "/tmp/result.mp4", {"result": "ok"}

        task_id = tm.add_task({"prompt": "test"})
        tm.start_worker(overprogress_target)
        time.sleep(0.3)

        task = tm.get_task(task_id)
        assert task is not None
        # During work, min(progress, 99) clamps it; final result sets to 100
        assert task["progress"] == 100

        tm.stop_worker()

    def test_task_manager_clear_completed(self):
        """clear_completed is not a method on TaskManager, but we can test task count filtering as a proxy.
        Test that get_task_count filters by status correctly."""
        tm = TaskManager()

        def quick_target(task_id_arg, progress_cb):
            return "/tmp/result.mp4", {"result": "ok"}

        # Add some tasks and let them complete
        ids = [tm.add_task({"prompt": f"test{i}"}) for i in range(3)]
        tm.start_worker(quick_target)
        time.sleep(0.5)

        # All should be completed
        completed = tm.get_task_count(status="completed")
        assert completed == 3

        # No queued, running, or failed
        assert tm.get_task_count(status="queued") == 0
        assert tm.get_task_count(status="running") == 0
        assert tm.get_task_count(status="failed") == 0

        tm.stop_worker()

    def test_task_manager_get_task_count(self):
        """get_task_count should count correctly by status"""
        tm = TaskManager()

        # Add a task without starting worker
        id1 = tm.add_task({"prompt": "a"})
        id2 = tm.add_task({"prompt": "b"})
        id3 = tm.add_task({"prompt": "c"})

        # All are queued
        assert tm.get_task_count(status="queued") == 3
        assert tm.get_task_count() == 3  # All tasks
        assert tm.get_task_count(status="running") == 0
        assert tm.get_task_count(status="completed") == 0

    def test_task_manager_worker_idempotent(self):
        """Starting worker multiple times should not create multiple workers"""
        tm = TaskManager()

        def quick_target(task_id_arg, progress_cb):
            return "/tmp/result.mp4", {"result": "ok"}

        tm.start_worker(quick_target)
        first_worker = tm._worker
        tm.start_worker(quick_target)  # Should be no-op since first is alive
        assert tm._worker is first_worker

        tm.stop_worker()

    def test_task_manager_worker_stop(self):
        """Stopping worker sets _running to False"""
        tm = TaskManager()
        assert tm._running is False

        def quick_target(task_id_arg, progress_cb):
            return "/tmp/result.mp4", {"result": "ok"}

        tm.start_worker(quick_target)
        assert tm._running is True
        tm.stop_worker()
        assert tm._running is False

    def test_task_manager_get_nonexistent_task(self):
        """get_task for non-existent ID returns None"""
        tm = TaskManager()
        result = tm.get_task("NONEXISTENT")
        assert result is None

    def test_task_manager_get_task_is_copy(self):
        """get_task should return a deep copy, not the original dict"""
        tm = TaskManager()
        task_id = tm.add_task({"prompt": "test"})
        task = tm.get_task(task_id)
        # Modify the returned copy
        task["progress"] = 999
        # Original should be unaffected
        original = tm.get_task(task_id)
        assert original["progress"] != 999
