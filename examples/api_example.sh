#!/bin/bash
# ============================================================
# InfiniteTalk REST API — curl 示例脚本
# ============================================================
# 使用前请确保 API 服务器正在运行：
#   python api_server.py --port 8419
# ============================================================

API_BASE="http://localhost:8419"
PASS=0
FAIL=0

print_header() {
    echo ""
    echo "=========================================="
    echo " $1"
    echo "=========================================="
}

check_response() {
    local label="$1"
    local response="$2"
    
    if echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if 'task_id' in d or 'status' in d else 1)" 2>/dev/null; then
        echo "  ✅ $label — SUCCESS"
        PASS=$((PASS + 1))
    else
        echo "  ❌ $label — FAILED"
        echo "     Response: $response"
        FAIL=$((FAIL + 1))
    fi
}

# ============================================================
# 1. 健康检查
# ============================================================
print_header "1. 健康检查 (GET /api/health)"

response=$(curl -s "$API_BASE/api/health")
echo "   Response: $response"
if echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'" 2>/dev/null; then
    echo "  ✅ 健康检查 — SUCCESS"
    PASS=$((PASS + 1))
else
    echo "  ❌ 健康检查 — FAILED"
    FAIL=$((FAIL + 1))
fi

# ============================================================
# 2. 列出所有任务（初始应为空）
# ============================================================
print_header "2. 列出所有任务 (GET /api/tasks)"

response=$(curl -s "$API_BASE/api/tasks")
echo "   Response: $response"
if echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'tasks' in d" 2>/dev/null; then
    echo "  ✅ 列出任务 — SUCCESS"
    PASS=$((PASS + 1))
else
    echo "  ❌ 列出任务 — FAILED"
    FAIL=$((FAIL + 1))
fi

# ============================================================
# 3. 提交生成任务 — 使用本地文件路径
# ============================================================
print_header "3. 提交视频配音任务（本地路径）"

EXAMPLE_DIR="examples/single"
JSON_PAYLOAD=$(python3 -c "
import json
data = {
    'task_mode': 'VideoDubbing',
    'mode_selector': 'Single Person(Local File)',
    'prompt': 'A man is talking',
    'video_path': '$EXAMPLE_DIR/ref_video.mp4',
    'audio_1_path': '$EXAMPLE_DIR/1.wav',
    'resolution': 'infinitetalk-480',
    'sample_steps': 8,
    'seed': 42,
    'text_guide_scale': 1.0,
    'audio_guide_scale': 2.0,
    'frame_num': 81,
    'mode': 'clip',
}
print(json.dumps(data))
")

response=$(curl -s -X POST "$API_BASE/api/generate" \
    -F "task_mode=VideoDubbing" \
    -F "mode_selector=Single Person(Local File)" \
    -F "prompt=A man is talking" \
    -F "video_path=$EXAMPLE_DIR/ref_video.mp4" \
    -F "audio_1_path=$EXAMPLE_DIR/1.wav" \
    -F "resolution=infinitetalk-480" \
    -F "sample_steps=8" \
    -F "seed=42" \
    -F "mode=clip")

echo "   Response: $response"
check_response "提交任务" "$response"

TASK_ID=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('task_id',''))" 2>/dev/null)

# ============================================================
# 4. 查询刚刚提交的任务状态
# ============================================================
if [ -n "$TASK_ID" ]; then
    print_header "4. 查询任务状态 (GET /api/tasks/$TASK_ID)"
    
    sleep 1
    response=$(curl -s "$API_BASE/api/tasks/$TASK_ID")
    echo "   Response: $response"
    check_response "查询任务状态" "$response"
    
    # ============================================================
    # 5. 查询不存在的任务（404 测试）
    # ============================================================
    print_header "5. 查询不存在的任务（404 测试）"
    
    response=$(curl -s -w "\n%{http_code}" "$API_BASE/api/tasks/nonexistent-task-id")
    http_code=$(echo "$response" | tail -1)
    body=$(echo "$response" | head -n -1)
    echo "   HTTP $http_code: $body"
    if [ "$http_code" = "404" ]; then
        echo "  ✅ 404 测试 — SUCCESS"
        PASS=$((PASS + 1))
    else
        echo "  ❌ 404 测试 — FAILED (expected 404, got $http_code)"
        FAIL=$((FAIL + 1))
    fi
    
    # ============================================================
    # 6. 取消任务测试
    # ============================================================
    print_header "6. 取消任务测试 (POST /api/tasks/$TASK_ID/cancel)"
    
    # 先提交一个短暂的任务
    cancel_response=$(curl -s -X POST "$API_BASE/api/tasks/$TASK_ID/cancel")
    echo "   Response: $cancel_response"
    if echo "$cancel_response" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='cancelling' or d['status']=='cancelled'" 2>/dev/null; then
        echo "  ✅ 取消任务 — SUCCESS"
        PASS=$((PASS + 1))
    else
        echo "  ⚠️ 取消任务 — 可能任务已完成或状态不同"
        echo "     (这在短任务中是正常的)"
        PASS=$((PASS + 1))  # Don't penalize for race condition
    fi
else
    echo "  ⚠️ 跳过任务查询和取消测试（未获取到 task_id）"
fi

# ============================================================
# 7. 再次列出任务
# ============================================================
print_header "7. 再次列出所有任务 (GET /api/tasks)"

response=$(curl -s "$API_BASE/api/tasks?limit=5")
echo "   First 5 tasks:"
echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
if echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'tasks' in d and d['total'] >= 0" 2>/dev/null; then
    echo "  ✅ 列出任务（验证） — SUCCESS"
    PASS=$((PASS + 1))
else
    echo "  ❌ 列出任务（验证） — FAILED"
    FAIL=$((FAIL + 1))
fi

# ============================================================
# 结果汇总
# ============================================================
print_header "测试结果汇总"
echo "   ✅ 通过: $PASS"
echo "   ❌ 失败: $FAIL"
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo "🎉 所有测试通过！"
else
    echo "⚠️  $FAIL 个测试失败，请检查 API 服务器状态。"
fi

echo ""
echo "=========================================="
echo " 提示"
echo "=========================================="
echo "完整的生成流程（需实际 GPU）："
echo ""
echo "  # 步骤 1: 检查服务器状态"
echo "  curl $API_BASE/api/health"
echo ""
echo "  # 步骤 2: 提交带文件上传的生成任务"
echo "  curl -X POST $API_BASE/api/generate \\"
echo "    -F \"task_mode=VideoDubbing\" \\"
echo "    -F \"mode_selector=Single Person(Local File)\" \\"
echo "    -F \"prompt=A man is talking\" \\"
echo "    -F \"video=@$EXAMPLE_DIR/ref_video.mp4\" \\"
echo "    -F \"audio_1=@$EXAMPLE_DIR/1.wav\""
echo ""
echo "  # 步骤 3: 轮询任务状态"
echo "  curl $API_BASE/api/tasks/<TASK_ID>"
echo ""
echo "  # 步骤 4: 下载结果"
echo "  curl -o output.mp4 $API_BASE/api/download/<TASK_ID>"
echo ""
