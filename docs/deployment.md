# InfiniteTalk Docker 部署指南

## 📋 目录

1. [环境要求](#1-环境要求)
2. [安装 nvidia-container-toolkit](#2-安装-nvidia-container-toolkit)
3. [快速开始](#3-快速开始)
4. [配置说明](#4-配置说明)
5. [常见操作](#5-常见操作)
6. [故障排除](#6-故障排除)
7. [生产部署建议](#7-生产部署建议)

---

## 1. 环境要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU 显存 | 16GB（480P） | 24GB+（720P） |
| 内存 | 32GB | 64GB |
| 磁盘空间 | 100GB（含模型权重 ~70GB） | 200GB+ |
| CPU | 8 cores | 16 cores |

### 软件要求

| 组件 | 版本 |
|------|------|
| Docker | 24.0+ |
| Docker Compose | 2.20+ |
| NVIDIA 驱动 | 535+（支持 CUDA 12.1） |
| nvidia-container-toolkit | 最新版 |
| 操作系统 | Linux（推荐 Ubuntu 22.04） |

> ⚠️ **重要**: 当前 Docker 镜像基于 x86_64 架构构建，不兼容 Apple Silicon (ARM64) Mac。
> 要在 Mac 上开发测试，建议使用远程 Linux 服务器或云 GPU 实例。

---

## 2. 安装 nvidia-container-toolkit

### Ubuntu/Debian

```bash
# 添加 NVIDIA 容器工具包仓库
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 安装
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 配置 Docker 运行时
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### CentOS/RHEL

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
    sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

sudo yum install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 3. 快速开始

### 第一步：克隆项目

```bash
git clone <your-repo-url> InfiniteTalk
cd InfiniteTalk
```

### 第二步：构建镜像

```bash
# 构建 Docker 镜像（首次约 30-60 分钟，因为 flash-attn 需要编译）
docker compose build

# 或直接使用 Docker
docker build -t infinitetalk:latest .
```

### 第三步：配置 HuggingFace Token（可选）

创建 `.env` 文件：

```bash
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

> 某些模型可能需要登录才能下载。如果模型是公开的，可以不设置。

### 第四步：下载模型权重

```bash
# 方式一：使用 Docker 下载（推荐）
docker compose run --rm download-models

# 方式二：使用宿主机下载（需要有 Python 环境）
chmod +x scripts/download_models.sh
./scripts/download_models.sh
```

下载内容（约 50-70GB）：

| 模型 | 大小（约） | 说明 |
|------|-----------|------|
| Wan2.1-I2V-14B-480P | ~30GB | 基础视频生成模型 |
| chinese-wav2vec2-base | ~1GB | 音频编码器 |
| InfiniteTalk | ~20GB | 音频条件模型权重 |

### 第五步：启动服务

```bash
# 启动 Gradio UI（端口 8418）
docker compose up -d gradio

# 查看日志
docker compose logs -f gradio
```

### 第六步：访问

打开浏览器访问：`http://<服务器IP>:8418`

---

## 4. 配置说明

### 目录结构

```
InfiniteTalk/
├── weights/               # 模型权重（volumes 挂载）
│   ├── Wan2.1-I2V-14B-480P/
│   ├── chinese-wav2vec2-base/
│   └── InfiniteTalk/
├── outputs/               # 生成视频输出（volumes 挂载）
├── .env                   # 环境变量配置
├── docker-compose.yml     # 服务编排
├── Dockerfile             # 镜像构建
└── scripts/
    └── download_models.sh # 模型下载脚本
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HF_TOKEN` | `""` | HuggingFace 访问令牌 |
| `GRADIO_SERVER_PORT` | `8418` | Gradio 服务端口 |
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Gradio 监听地址 |

### 使用 HuggingFace 镜像站（中国大陆）

如果无法直接访问 huggingface.co，设置镜像站：

```bash
# 方法一：修改 .env 文件
echo "HF_ENDPOINT=https://hf-mirror.com" >> .env

# 方法二：直接传递给下载容器
docker compose run --rm -e HF_ENDPOINT=https://hf-mirror.com download-models
```

---

## 5. 常见操作

### 停止服务

```bash
docker compose down
```

### 重启服务

```bash
docker compose restart gradio
```

### 查看 GPU 使用情况

```bash
docker compose exec gradio nvidia-smi
```

### 查看日志

```bash
docker compose logs -f --tail=100 gradio
```

### 进入容器内部

```bash
docker compose exec -it gradio bash
```

### 升级镜像

```bash
# 拉取最新代码
git pull

# 重新构建
docker compose build --no-cache

# 重启服务
docker compose up -d gradio
```

### 使用 API（如果后续实现）

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"audio": "...", "image": "..."}'
```

---

## 6. 故障排除

### ❌ GPU 不可用

```bash
# 检查 nvidia-container-toolkit 是否正确安装
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 检查 Docker 运行时配置
docker info | grep -i runtime

# 重新配置
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### ❌ 显存不足

```bash
# 查看 GPU 显存使用
nvidia-smi

# 修改 docker-compose.yml，使用低显存模式
# 在 command 中添加：--num_persistent_param_in_dit 0

# 或者使用量化模型
# 需要额外下载量化权重
```

### ❌ 模型下载失败

```bash
# 1. 检查网络连接
ping huggingface.co

# 2. 使用镜像站
docker compose run --rm -e HF_ENDPOINT=https://hf-mirror.com download-models

# 3. 手动下载后拷贝到 weights/ 目录
```

### ❌ 生成视频质量差

确保以下参数设置合理：

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `sample_audio_guide_scale` | 3-5 | 音频引导强度 |
| `sample_text_guide_scale` | 5 (无LoRA) / 1 (有LoRA) | 文本引导强度 |
| `sample_steps` | 40 (默认) / 8 (FusionX) | 采样步数 |

### ❌ 容器启动后立即退出

```bash
# 查看退出原因
docker compose logs gradio

# 常见原因：
# 1. 模型权重未下载 -> 运行 download-models
# 2. GPU 驱动版本不匹配 -> 检查 CUDA 版本
# 3. 显存不足 -> 使用 --num_persistent_param_in_dit 0
```

---

## 7. 生产部署建议

### 安全配置

```bash
# 限制 Gradio 访问来源（修改 app.py 或通过环境变量）
# 建议使用 Nginx 反向代理 + HTTPS

# Nginx 配置示例
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8418;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

### 资源限制

```yaml
# docker-compose.yml 中可添加资源限制
deploy:
  resources:
    limits:
      cpus: '8'
      memory: 64G
    reservations:
      cpus: '4'
      memory: 32G
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### 监控

```bash
# 使用 nvidia-smi 监控 GPU
watch -n 1 nvidia-smi

# 查看容器资源使用
docker stats infinitetalk-gradio
```

### 多 GPU 支持

如果需要多 GPU 推理，修改 `docker-compose.yml`：

```yaml
services:
  gradio:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all  # 使用所有 GPU
              capabilities: [gpu]
```

然后需要在容器内使用 `torchrun`：

```bash
docker compose exec gradio bash -c "
torchrun --nproc_per_node=\$(nvidia-smi -L | wc -l) --standalone \
  generate_infinitetalk.py \
  --ckpt_dir /app/weights/Wan2.1-I2V-14B-480P \
  --wav2vec_dir /app/weights/chinese-wav2vec2-base \
  --infinitetalk_dir /app/weights/InfiniteTalk/single/infinitetalk.safetensors \
  --dit_fsdp --t5_fsdp \
  --ulysses_size=\$(nvidia-smi -L | wc -l) \
  --input_json /app/examples/single_example_image.json \
  --size infinitetalk-480 \
  --sample_steps 40 \
  --mode streaming \
  --motion_frame 9 \
  --save_file /app/outputs/multigpu_result
"
```

---

## 附录

### Docker 命令备忘

```bash
# 构建（不使用缓存）
docker build --no-cache -t infinitetalk:latest .

# 查看镜像
docker images infinitetalk

# 查看运行中的容器
docker ps

# 清理
docker system prune -a --volumes
```

### 推荐的云 GPU 实例

| 云平台 | 实例类型 | GPU | 显存 | 适用分辨率 |
|--------|----------|-----|------|-----------|
| AWS | g5.xlarge | A10G | 24GB | 480P |
| AWS | g5.2xlarge | A10G | 24GB | 480P/720P |
| Azure | NCas_T4_v3 | T4 | 16GB | 480P |
| Azure | NC6s_v3 | V100 | 16GB | 480P |
| 阿里云 | ecs.gn7i-c16g1.4xlarge | A10 | 24GB | 480P/720P |
| 腾讯云 | GN10Xp.2XLARGE | T4 | 16GB | 480P |
