#!/usr/bin/env bash
# =============================================================================
# Model Download Script for InfiniteTalk
# =============================================================================
# Downloads all required model weights to /app/weights/
#
# Usage:
#   chmod +x scripts/download_models.sh
#   ./scripts/download_models.sh
#
# Environment:
#   HF_TOKEN       — HuggingFace token (optional, for gated models)
#   HF_ENDPOINT    — HuggingFace mirror endpoint (optional)
#   WEIGHTS_DIR    — Target directory (default: /app/weights)
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
WEIGHTS_DIR="${WEIGHTS_DIR:-/app/weights}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_TOKEN_ARG=""
NUM_WORKERS="${NUM_WORKERS:-4}"

if [ -n "${HF_TOKEN:-}" ]; then
    HF_TOKEN_ARG="--token $HF_TOKEN"
fi

echo "=============================================="
echo "  InfiniteTalk Model Downloader"
echo "=============================================="
echo "Weights directory: $WEIGHTS_DIR"
echo "HF Endpoint:       $HF_ENDPOINT"
echo "=============================================="

# Ensure huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-hub..."
    pip install -U "huggingface-hub[cli]" hf_transfer
fi

mkdir -p "$WEIGHTS_DIR"

# ── Helper ────────────────────────────────────────────────────────────────────
download_model() {
    local repo="$1"
    local target="$2"
    local extra_args="${3:-}"

    echo ""
    echo "📥 Downloading $repo → $target"
    echo "----------------------------------------"

    huggingface-cli download \
        $HF_TOKEN_ARG \
        --resume-download \
        --local-dir-use-symlinks False \
        --local-dir "$WEIGHTS_DIR/$target" \
        $extra_args \
        "$repo"

    echo "✅ Done: $repo"
}

# ── Model 1: Wan2.1-I2V-14B-480P ─────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║  1/3: Wan2.1-I2V-14B-480P (Base model)   ║"
echo "╚═══════════════════════════════════════════╝"
download_model "Wan-AI/Wan2.1-I2V-14B-480P" "Wan2.1-I2V-14B-480P"

# ── Model 2: chinese-wav2vec2-base ────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║  2/3: chinese-wav2vec2-base (Audio enc)   ║"
echo "╚═══════════════════════════════════════════╝"
download_model "TencentGameMate/chinese-wav2vec2-base" "chinese-wav2vec2-base"

# Also download the safetensors revision
echo "  → Downloading safetensors revision..."
huggingface-cli download \
    $HF_TOKEN_ARG \
    TencentGameMate/chinese-wav2vec2-base \
    model.safetensors \
    --revision refs/pr/1 \
    --local-dir "$WEIGHTS_DIR/chinese-wav2vec2-base" \
    --resume-download

# ── Model 3: InfiniteTalk ─────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║  3/3: MeiGen-AI/InfiniteTalk               ║"
echo "╚═══════════════════════════════════════════╝"
download_model "MeiGen-AI/InfiniteTalk" "InfiniteTalk"

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Verification"
echo "=============================================="

verify_dir() {
    local dir="$1"
    local label="$2"
    if [ -d "$WEIGHTS_DIR/$dir" ] && [ "$(ls -A "$WEIGHTS_DIR/$dir" 2>/dev/null)" ]; then
        echo "✅ $label: found ($(du -sh "$WEIGHTS_DIR/$dir" | cut -f1))"
    else
        echo "❌ $label: MISSING or empty!"
    fi
}

verify_dir "Wan2.1-I2V-14B-480P"         "Wan2.1-I2V-14B-480P"
verify_dir "chinese-wav2vec2-base"        "chinese-wav2vec2-base"
verify_dir "InfiniteTalk"                 "InfiniteTalk"

echo ""
echo "=============================================="
echo "  All downloads complete!"
echo "  Total size:"
du -sh "$WEIGHTS_DIR"
echo "=============================================="
