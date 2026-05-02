<div align="center">

<p align="center">
  <img src="assets/logo2.jpg" alt="InfinteTalk" width="440"/>
</p>

<h1>InfiniteTalk: Audio-driven Video Generation for Sparse-Frame Video Dubbing</h1>


[Shaoshu Yang*](https://scholar.google.com/citations?user=JrdZbTsAAAAJ&hl=en) · [Zhe Kong*](https://scholar.google.com/citations?user=4X3yLwsAAAAJ&hl=zh-CN) · [Feng Gao*](https://scholar.google.com/citations?user=lFkCeoYAAAAJ) · [Meng Cheng*]() · [Xiangyu Liu*]() · [Yong Zhang](https://yzhang2016.github.io/)<sup>&#9993;</sup> · [Zhuoliang Kang](https://scholar.google.com/citations?user=W1ZXjMkAAAAJ&hl=en)

[Wenhan Luo](https://whluo.github.io/) · [Xunliang Cai](https://openreview.net/profile?id=~Xunliang_Cai1) · [Ran He](https://scholar.google.com/citations?user=ayrg9AUAAAAJ&hl=en)· [Xiaoming Wei](https://scholar.google.com/citations?user=JXV5yrZxj5MC&hl=zh-CN) 

<sup>*</sup>Equal Contribution
<sup>&#9993;</sup>Corresponding Authors

<a href='https://meigen-ai.github.io/InfiniteTalk/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2508.14033'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
<a href='https://huggingface.co/MeiGen-AI/InfiniteTalk'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
</div>

> **TL; DR:**  InfiniteTalk is an unlimited-length talking video generation​​ model that supports both audio-driven video-to-video and image-to-video generation

<p align="center">
  <img src="assets/pipeline.png">
</p>







## 🔥 Latest News
* Dec 16, 2025: 🚀 We are excited to announce the release of **[LongCat-Video-Avatar](https://github.com/MeiGen-AI/LongCat-Video-Avatar)**, a unified model that delivers expressive and highly dynamic audio-driven character animation, supporting native tasks including Audio-Text-to-Video, Audio-Text-Image-to-Video, and Video Continuation with seamless compatibility for both single-stream and multi-stream audio inputs. The release includes our Technical Report, [code](https://github.com/meituan-longcat/LongCat-Video), [model weights](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar), and [project page](https://meigen-ai.github.io/LongCat-Video-Avatar/).
* August 19, 2025: We release the [Technique-Report](https://arxiv.org/abs/2508.14033) , weights, and code of **InfiniteTalk**. The Gradio and the [ComfyUI](https://github.com/MeiGen-AI/InfiniteTalk/tree/comfyui) branch have been released. 
* August 19, 2025: We release the [project page](https://meigen-ai.github.io/InfiniteTalk/) of **InfiniteTalk** 


## ✨ Key Features
We propose **InfiniteTalk**​​, a novel sparse-frame video dubbing framework. Given an input video and audio track, InfiniteTalk synthesizes a new video with ​​accurate lip synchronization​​ while ​​simultaneously aligning head movements, body posture, and facial expressions​​ with the audio. Unlike traditional dubbing methods that focus solely on lips, InfiniteTalk enables ​​infinite-length video generation​​ with accurate lip synchronization and consistent identity preservation. Beside, InfiniteTalk can also be used as an image-audio-to-video model with an image and an audio as input. 
- 💬 ​​Sparse-frame Video Dubbing​​ – Synchronizes not only lips, but aslo head, body, and expressions
- ⏱️ ​​Infinite-Length Generation​​ – Supports unlimited video duration
- ✨ ​​Stability​​ – Reduces hand/body distortions compared to MultiTalk
- 🚀 ​​Lip Accuracy​​ – Achieves superior lip synchronization to MultiTalk



## 🌐 Community  Works
- [Wan2GP](https://github.com/deepbeepmeep/Wan2GP/): Thanks [deepbeepmeep](https://github.com/deepbeepmeep) for integrating InfiniteTalk in Wan2GP that is optimized for low VRAM and offers many video edtiting option and other models (MMaudio support, Qwen Image Edit, ...). 
- [ComfyUI](https://github.com/kijai/ComfyUI-WanVideoWrapper): Thanks for the comfyui support of [kijai](https://github.com/kijai). 



## 📑 Todo List

- [x] Release the technical report
- [x] Inference
- [x] Checkpoints
- [x] Multi-GPU Inference
- [ ] Inference acceleration
  - [x] TeaCache
  - [x] int8 quantization
  - [ ] LCM distillation
  - [x] Sparse Attention
- [x] Run with very low VRAM
- [x] Gradio demo
- [x] ComfyUI

## 🚀 新增功能

### Mac (Apple Silicon / MPS) 支持
- **设备无关架构** — 自动适配 CUDA / MPS / CPU，无需手动切换
- **原生 PyTorch SDPA** — 替代 xformers/flash-attention，在 MPS 上使用 `scaled_dot_product_attention`
- **PyAV 替代 Decord** — 视频读写使用 `av` 库，兼容 Mac 环境
- **自动降级** — xfuser、optimum-quanto、FSDP 等在非 CUDA 设备上自动跳过
- **注意事项** — 14B 模型需要 ~28GB+ 统一内存，首次加载较慢（T5 模型 11GB）

### Gradio UI 增强
- 🌐 **中英文界面** — 支持一键切换中英文显示
- 📋 **任务队列** — 批量任务排队执行，实时进度追踪
- 📜 **历史记录** — 自动保存生成记录，支持查看/删除
- ⚙️ **参数预设** — 保存/加载常用配置
- ⚠️ **Gradio 版本** — 仅支持 Gradio 5.x（已锁定 `<6`，Gradio 6 前端存在兼容性问题）

### REST API 服务
- 提供完整的 REST API，支持异步任务提交、查询、取消、下载
- 端口 8419，与 Gradio 界面（8418）分离
- 详情见 [docs/api_docs.md](docs/api_docs.md)

### Sparse Attention 加速
- 推理时稀疏化 attention 计算，降低延迟
- 支持 `--use_sparse_attention` 和 `--sparse_attention_ratio` 参数
- 与 TeaCache 可叠加使用
- 详情见 [docs/sparse_attention.md](docs/sparse_attention.md)

### Docker 一键部署
- 提供 Dockerfile 和 docker-compose.yml，支持 GPU 容器化部署
- 包含模型自动下载脚本
- 详情见 [docs/deployment.md](docs/deployment.md)

### TeaCache 推理加速
- 推理时跳过冗余计算步骤，降低延迟
- 支持 `--use_teacache` 和 `--teacache_thresh` 参数
- 详情见 [docs/teacache.md](docs/teacache.md)

## Video Demos


### Video-to-video (HQ videos can be found on [Google Drive](https://drive.google.com/drive/folders/1BNrH6GJZ2Wt5gBuNLmfXZ6kpqb9xFPjU?usp=sharing) )


<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/04f15986-8de7-4bb4-8cde-7f7f38244f9f" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/1500f72e-a096-42e5-8b44-f887fa8ae7cb" width="320" controls loop></video>
     </td>
     <td>
          <video src="https://github.com/user-attachments/assets/28f484c2-87dc-4828-a9e7-cb963da92d14" width="320" controls loop></video>
     </td>
     <td>
          <video src="https://github.com/user-attachments/assets/665fabe4-3e24-4008-a0a2-a66e2e57c38b" width="320" controls loop></video>
     </td>
  </tr>
</table>

### Image-to-video

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/7e4a4dad-9666-4896-8684-2acb36aead59" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/bd6da665-f34d-4634-ae94-b4978f92ad3a" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/510e2648-82db-4648-aaf3-6542303dbe22" width="320" controls loop></video>
     </td>
     <td>
          <video src="https://github.com/user-attachments/assets/27bb087b-866a-4300-8a03-3bbb4ce3ddf9" width="320" controls loop></video>
     </td>
     
  </tr>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/3263c5e1-9f98-4b9b-8688-b3e497460a76" width="320" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/5ff3607f-90ec-4eee-b964-9d5ee3028005" width="320" controls loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/e504417b-c8c7-4cf0-9afa-da0f3cbf3726" width="320" controls loop></video>
     </td>
     <td>
          <video src="https://github.com/user-attachments/assets/56aac91e-c51f-4d44-b80d-7d115e94ead7" width="320" controls loop></video>
     </td>
     
  </tr>
</table>

## Quick Start

### 🛠️Installation

Choose your platform:

<details>
<summary><b>Mac (Apple Silicon / MPS)</b></summary>

```bash
# 1. 创建虚拟环境
python3 -m venv venv_mac
source venv_mac/bin/activate

# 2. Install PyTorch (Mac — MPS compatible)
pip install torch torchvision torchaudio

# 3. Install Mac-specific dependencies
pip install -r requirements-mac.txt

# 4. Install FFmpeg
brew install ffmpeg
```

> **Mac 注意事项：**
> - Mac MPS 不支持 xformers、flash-attn、xfuser、FSDP、量化等 CUDA 专用功能，这些会自动跳过
> - 模型使用原生 PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`)
> - 视频解码使用 PyAV 替代 Decord（`pip install av`）
> - **14B 模型需要 ~28GB+ 统一内存**，24GB Mac 可能需要使用 `--num_persistent_param_in_dit 0` 分片加载
> - 首次启动加载 T5 模型（约 11GB）可能需要 5-10 分钟，请耐心等待
> - 如果遇到 `killed` 或内存错误，请减少 `--frame_num` 值或使用更小的分辨率
> - 安装依赖请使用 `requirements-mac.txt`（不含 CUDA 专用包）
> - Gradio 仅支持 5.x 版本（`<6`），Gradio 6 前端存在按钮点击兼容性问题
</details>

<details>
<summary><b>Linux (CUDA GPU)</b></summary>

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with CUDA
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. Install dependencies
pip install -r requirements.txt
# Mac 用户请使用: pip install -r requirements-mac.txt

# 4. [可选] CUDA 优化组件（提升性能）
pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu124
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install xfuser>=0.4.1
pip install optimum-quanto==0.2.6

# 4. FFmpeg
sudo apt install ffmpeg   # Ubuntu/Debian
# conda install -c conda-forge ffmpeg
```

> **Linux 注意事项：**
> - CUDA 优化组件为可选项，`pip install -r requirements.txt` 已包含所有必需依赖
> - flash-attn 编译需要较长时间，推荐使用 `MAX_JOBS=4` 限制并行度
> - 对于低显存 GPU，使用 `--num_persistent_param_in_dit 0` 启用显存管理
</details>

### 🧱Model Preparation

#### 1. Model Download

| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-I2V-14B-480P  |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)       | Base model
| chinese-wav2vec2-base |      🤗 [Huggingface](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base)          | Audio encoder
| MeiGen-InfiniteTalk      |      🤗 [Huggingface](https://huggingface.co/MeiGen-AI/InfiniteTalk)              | Our audio condition weights

Download models using huggingface-cli:
``` sh
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk

```

### 🔑 Quick Inference

Our model is compatible with both 480P and 720P resolutions. 
> Some tips
> - Lip synchronization accuracy:​​ Audio CFG works optimally between 3–5. Increase the audio CFG value for better synchronization.
> - FusionX： While it enables faster inference and higher quality, FusionX LoRA exacerbates color shift over 1 minute and reduces ID preservation in videos.
> - V2V generation: Enables unlimited length generation. The model mimics the original video's camera movement, though not identically. Using SDEdit improves camera movement accuracy significantly but introduces color shift and is best suited for short clips. Improvements for long video camera control are planned.
> - I2V generation: Generates good results from a single image for up to 1 minute. Beyond 1 minute, color shifts become more pronounced. One trick for the high-quailty generation beyond 1 min is to copy the image to a video by translating or zooming in the image.  Here is a script to [convert image to video](https://github.com/MeiGen-AI/InfiniteTalk/blob/main/tools/convert_img_to_video.py).  
> - Quantization model: If your inference process is killed due to insufficient memory, we suggest using the quantization model, which can help **reduce memory usage**.

#### Usage of InfiniteTalk
```
--mode streaming: long video generation.
--mode clip: generate short video with one chunk. 
--use_teacache: run with TeaCache.
--size infinitetalk-480: generate 480P video.
--size infinitetalk-720: generate 720P video.
--use_apg: run with APG.
--teacache_thresh: A coefficient used for TeaCache acceleration
—-sample_text_guide_scale： When not using LoRA, the optimal value is 5. After applying LoRA, the recommended value is 1.
—-sample_audio_guide_scale： When not using LoRA, the optimal value is 4. After applying LoRA, the recommended value is 2.
—-sample_audio_guide_scale： When not using LoRA, the optimal value is 4. After applying LoRA, the recommended value is 2.
--max_frame_num: The max frame length of the generated video, the default is 40 seconds(1000 frames).
```

#### 1. Inference

##### 1) Run with single GPU


```
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res

```

##### 2) Run with 720P

If you want run with 720P, set `--size infinitetalk-720`:

```
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-720 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res_720p

```

##### 3) Run with very low VRAM

If you want run with very low VRAM, set `--num_persistent_param_in_dit 0`:


```
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --num_persistent_param_in_dit 0 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res_lowvram
```

##### 4) Multi-GPU inference

```
GPU_NUM=8
torchrun --nproc_per_node=$GPU_NUM --standalone generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --dit_fsdp --t5_fsdp \
    --ulysses_size=$GPU_NUM \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res_multigpu
```

##### 5) Multi-Person animation

```
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/multi/infinitetalk.safetensors \
    --input_json examples/multi_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --num_persistent_param_in_dit 0 \
    --mode streaming \
    --motion_frame 9 \
    --save_file infinitetalk_res_multiperson
```


#### 2. Run with FusioniX or Lightx2v(Require only 4~8 steps)

[FusioniX](https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/blob/main/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors) require 8 steps and [lightx2v](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors) requires only 4 steps.

```
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --lora_dir weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
    --input_json examples/single_example_image.json \
    --lora_scale 1.0 \
    --size infinitetalk-480 \
    --sample_text_guide_scale 1.0 \
    --sample_audio_guide_scale 2.0 \
    --sample_steps 8 \
    --mode streaming \
    --motion_frame 9 \
    --sample_shift 2 \
    --num_persistent_param_in_dit 0 \
    --save_file infinitetalk_res_lora
```



#### 3. Run with the quantization model (Only support run with single gpu)

```
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/single_example_image.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --quant fp8 \
    --quant_dir weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors \
    --motion_frame 9 \
    --num_persistent_param_in_dit 0 \
    --save_file infinitetalk_res_quant
```


#### 4. Run with Gradio

Gradio Web 界面提供以下功能：
- **任务模式**：单图驱动（I2V）或视频配音（V2V）
- **音频模式**：单人/多人，本地音频文件或 TTS 文字转语音
- **任务队列**：批量添加生成任务，后台依次执行
- **历史记录**：自动保存生成结果，支持查看和删除
- **参数预设**：保存常用参数配置，快速切换

```bash
# 单人模式
python app.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --num_persistent_param_in_dit 0 \
    --motion_frame 9

# 多人模式
python app.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/multi/infinitetalk.safetensors \
    --num_persistent_param_in_dit 0 \
    --motion_frame 9
```

> 启动后访问 `http://localhost:8418` 打开 Web 界面。

#### 5. Run with REST API

REST API 服务提供 HTTP 接口，支持编程式调用：

```bash
python api_server.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --num_persistent_param_in_dit 0 \
    --motion_frame 9 \
    --port 8419
```

> 启动后访问 `http://localhost:8419/docs` 查看 API 文档。
> 详情见 [docs/api_docs.md](docs/api_docs.md)


## 📚 Citation

If you find our work useful in your research, please consider citing:

```
@misc{yang2025infinitetalkaudiodrivenvideogeneration,
      title={InfiniteTalk: Audio-driven Video Generation for Sparse-Frame Video Dubbing}, 
      author={Shaoshu Yang and Zhe Kong and Feng Gao and Meng Cheng and Xiangyu Liu and Yong Zhang and Zhuoliang Kang and Wenhan Luo and Xunliang Cai and Ran He and Xiaoming Wei},
      year={2025},
      eprint={2508.14033},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.14033}, 
}
```

## 📜 License
The models in this repository are licensed under the Apache 2.0 License. We claim no rights over the your generated contents, 
granting you the freedom to use them while ensuring that your usage complies with the provisions of this license. 
You are fully accountable for your use of the models, which must not involve sharing any content that violates applicable laws, 
causes harm to individuals or groups, disseminates personal information intended for harm, spreads misinformation, or targets vulnerable populations. 

