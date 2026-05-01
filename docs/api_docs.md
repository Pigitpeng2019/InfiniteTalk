# InfiniteTalk REST API 文档

## 概述

InfiniteTalk REST API 提供基于 HTTP 的视频生成服务，支持音频驱动的视频生成和视频配音功能。

**基础 URL**: `http://<host>:8419`

**端口说明**: API 服务器默认运行在 8419 端口，可与 Gradio UI（8418 端口）同时运行。

---

## 接口列表

### 1. 健康检查

```
GET /api/health
```

检查服务器和 GPU 状态。

**响应示例**:
```json
{
  "status": "ok",
  "timestamp": "2025-05-01T10:30:00.123456",
  "gpu_busy": false,
  "active_tasks": 0
}
```

---

### 2. 提交视频生成任务

```
POST /api/generate
```

**Content-Type**: `multipart/form-data`

**参数说明**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| task_mode | string | 否 | `VideoDubbing` | 任务模式：`SingleImageDriven` 或 `VideoDubbing` |
| mode_selector | string | 否 | `Single Person(Local File)` | 音频模式选择 |
| prompt | string | 否 | `""` | 视频描述提示词 |
| image | file | 否 | — | 输入图片文件（SingleImageDriven 模式） |
| video | file | 否 | — | 输入视频文件（VideoDubbing 模式） |
| audio_1 | file | 否 | — | 说话人 1 的音频文件 |
| audio_2 | file | 否 | — | 说话人 2 的音频文件 |
| image_path | string | 否 | — | 输入图片的本地路径（替代上传） |
| video_path | string | 否 | — | 输入视频的本地路径（替代上传） |
| audio_1_path | string | 否 | — | 说话人 1 音频的本地路径 |
| audio_2_path | string | 否 | — | 说话人 2 音频的本地路径 |
| tts_text | string | 否 | — | TTS 模式下的文本内容 |
| voice_1_path | string | 否 | — | 说话人 1 的语音嵌入路径 |
| voice_2_path | string | 否 | — | 说话人 2 的语音嵌入路径 |
| resolution | string | 否 | `infinitetalk-480` | 分辨率：`infinitetalk-480` 或 `infinitetalk-720` |
| sample_steps | int | 否 | `8` | 扩散采样步数 |
| seed | int | 否 | `42` | 随机种子（-1 表示随机） |
| text_guide_scale | float | 否 | `1.0` | 文本引导比例 |
| audio_guide_scale | float | 否 | `2.0` | 音频引导比例 |
| frame_num | int | 否 | `81` | 每段帧数（需为 4n+1） |
| motion_frame | int | 否 | `9` | 长视频驱动帧长度 |
| max_frame_num | int | 否 | `1000` | 最大视频帧数 |
| mode | string | 否 | `streaming` | 生成模式：`clip` 或 `streaming` |
| n_prompt | string | 否 | — | 负面提示词 |
| color_correction_strength | float | 否 | `1.0` | 色彩校正强度（0.0-1.0） |

**mode_selector 可选值**:
- `Single Person(Local File)` — 单人，上传本地音频
- `Single Person(TTS)` — 单人，TTS 生成语音
- `Multi Person(Local File, audio add)` — 双人，音频叠加
- `Multi Person(Local File, audio parallel)` — 双人，音频并行
- `Multi Person(TTS)` — 双人，TTS 生成语音

**响应**:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

---

### 3. 查询任务状态

```
GET /api/tasks/{task_id}
```

**响应示例**:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": 45,
  "log": [
    "[10:30:01] Task started.",
    "[10:30:05] Processing audio files...",
    "[10:30:08] Audio processing complete. Starting video generation...",
    "[10:31:15] Running InfiniteTalk pipeline..."
  ],
  "error": null,
  "result_url": null,
  "created_at": "2025-05-01T10:30:00.123456",
  "updated_at": "2025-05-01T10:31:15.654321"
}
```

**status 可能的取值**:
| 值 | 说明 |
|------|------|
| `queued` | 任务已创建，等待执行 |
| `running` | 任务正在执行中 |
| `done` | 任务已完成 |
| `failed` | 任务执行失败 |
| `cancelled` | 任务已被取消 |

---

### 4. 取消任务

```
POST /api/tasks/{task_id}/cancel
```

取消一个排队中或运行中的任务。由于 GPU 任务的特殊性，取消信号会在当前生成阶段结束后生效。

**响应**:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelling"
}
```

---

### 5. 下载生成的视频

```
GET /api/download/{task_id}
```

下载已完成的视频文件。仅在任务状态为 `done` 时可下载。

**响应**: `video/mp4` 文件流

**错误**: 任务未完成时返回 400，任务不存在时返回 404。

---

### 6. 列出所有任务

```
GET /api/tasks?limit=50&offset=0&status_filter=running
```

**查询参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| limit | int | `50` | 返回的任务数量上限 |
| offset | int | `0` | 跳过的任务数（分页） |
| status_filter | string | — | 按状态筛选（可选） |

**响应**:
```json
{
  "total": 3,
  "limit": 50,
  "offset": 0,
  "tasks": [
    {
      "task_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "running",
      "progress": 45,
      "log": ["[10:30:01] Task started."],
      "error": null,
      "result_url": null,
      "created_at": "2025-05-01T10:30:00.123456",
      "updated_at": "2025-05-01T10:31:15.654321"
    }
  ]
}
```

---

## 注意事项

### GPU 资源管理
- GPU 是共享资源，同一时间只能运行一个生成任务
- 其他任务会在队列中等待 GPU 锁释放
- 可通过 `/api/health` 的 `gpu_busy` 字段查看当前 GPU 占用状态

### 文件上传 vs 本地路径
- 可以通过文件上传（`multipart/form-data`）提交输入文件
- 也可以通过提供本地路径（`*_path` 参数）引用服务器上的现有文件
- 若同时提供了上传文件和路径，上传文件优先

### 与 Gradio 共存
- API 服务器默认端口：**8419**
- Gradio UI 默认端口：**8418**
- 两者共享 GPU 资源，请勿同时提交任务
- 如果 Gradio 界面正在运行生成任务，API 请求会排队等待

### 任务清理
- 生成的视频文件保存在 `api_results/{task_id}/` 目录下
- 服务器重启时相同 `task_id` 的任务文件会被清除
- 建议生产环境配置定期清理策略

---

## curl 使用示例

```bash
# 1. 健康检查
curl http://localhost:8419/api/health

# 2. 提交视频配音任务（上传文件）
curl -X POST http://localhost:8419/api/generate \
  -F "task_mode=VideoDubbing" \
  -F "mode_selector=Single Person(Local File)" \
  -F "prompt=A man is talking" \
  -F "video=@/path/to/input_video.mp4" \
  -F "audio_1=@/path/to/audio.wav" \
  -F "resolution=infinitetalk-480" \
  -F "sample_steps=8"

# 3. 提交图片驱动任务（使用本地路径）
curl -X POST http://localhost:8419/api/generate \
  -F "task_mode=SingleImageDriven" \
  -F "mode_selector=Single Person(TTS)" \
  -F "prompt=A woman singing in studio" \
  -F "image_path=/path/to/image.png" \
  -F "tts_text=Hello, this is a test" \
  -F "voice_1_path=weights/Kokoro-82M/voices/am_adam.pt"

# 4. 查询任务状态
curl http://localhost:8419/api/tasks/<TASK_ID>

# 5. 取消任务
curl -X POST http://localhost:8419/api/tasks/<TASK_ID>/cancel

# 6. 下载结果
curl -o output.mp4 http://localhost:8419/api/download/<TASK_ID>

# 7. 列出所有 running 状态的任务
curl "http://localhost:8419/api/tasks?status_filter=running"
```
