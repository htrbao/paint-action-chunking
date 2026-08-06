#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Host a GR00T inference server running repaint-euler on TensorRT.
#
# Prerequisites (run once):
#   python deployment_scripts/export_onnx.py
#   bash   deployment_scripts/build_engine.sh
#
# Usage:
#   bash deployment_scripts/run_repaint_trt_server.sh
#
# Override anything via environment variables:
#   MODEL_PATH=/path/to/ckpt DATA_CONFIG=so100 PORT=5556 \
#       bash deployment_scripts/run_repaint_trt_server.sh
#
#   HTTP_SERVER=1 PORT=8000 bash deployment_scripts/run_repaint_trt_server.sh

set -euo pipefail

MODEL_PATH=${MODEL_PATH:-nvidia/GR00T-N1.5-3B}
DATA_CONFIG=${DATA_CONFIG:-fourier_gr1_arms_waist}
EMBODIMENT_TAG=${EMBODIMENT_TAG:-gr1}
DENOISING_STEPS=${DENOISING_STEPS:-4}

TRT_ENGINE_PATH=${TRT_ENGINE_PATH:-gr00t_engine}
VIT_DTYPE=${VIT_DTYPE:-fp8}     # Options: fp16, fp8
LLM_DTYPE=${LLM_DTYPE:-nvfp4}   # Options: fp16, nvfp4, fp8
DIT_DTYPE=${DIT_DTYPE:-fp8}     # Options: fp16, fp8

HOST=${HOST:-localhost}
PORT=${PORT:-5555}
HTTP_SERVER=${HTTP_SERVER:-0}   # 1 -> REST server instead of ZMQ
API_TOKEN=${API_TOKEN:-}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ── preflight: the engines repaint-euler will sweep three times per chunk ─────────
REQUIRED_ENGINES=(
    "vlln_vl_self_attention.engine"
    "state_encoder.engine"
    "action_encoder.engine"
    "action_decoder.engine"
    "DiT_${DIT_DTYPE}.engine"
    "vit_${VIT_DTYPE}.engine"
    "llm_${LLM_DTYPE}.engine"
)

if [ ! -d "$TRT_ENGINE_PATH" ]; then
    echo "Error: engine directory '${TRT_ENGINE_PATH}' not found."
    echo "Build the engines first:"
    echo "  python deployment_scripts/export_onnx.py"
    echo "  bash   deployment_scripts/build_engine.sh"
    exit 1
fi

MISSING=0
for engine in "${REQUIRED_ENGINES[@]}"; do
    if [ ! -e "${TRT_ENGINE_PATH}/${engine}" ]; then
        echo "Error: missing engine ${TRT_ENGINE_PATH}/${engine}"
        MISSING=1
    fi
done
if [ "$MISSING" = "1" ]; then
    echo ""
    echo "Rebuild with matching precisions, e.g.:"
    echo "  VIT_DTYPE=${VIT_DTYPE} LLM_DTYPE=${LLM_DTYPE} DIT_DTYPE=${DIT_DTYPE} bash deployment_scripts/build_engine.sh"
    exit 1
fi

echo "============================================================"
echo "GR00T inference server - repaint-euler on TensorRT"
echo "============================================================"
echo "  Model:            ${MODEL_PATH}"
echo "  Data config:      ${DATA_CONFIG}"
echo "  Embodiment:       ${EMBODIMENT_TAG}"
echo "  Denoising steps:  ${DENOISING_STEPS}  (repaint costs 3x = $((DENOISING_STEPS * 3)) DiT evals/chunk)"
echo "  Engines:          ${TRT_ENGINE_PATH}  (ViT ${VIT_DTYPE} / LLM ${LLM_DTYPE} / DiT ${DIT_DTYPE})"
if [ "$HTTP_SERVER" = "1" ]; then
    echo "  Transport:        HTTP on ${HOST}:${PORT}"
else
    echo "  Transport:        ZMQ on ${HOST}:${PORT}"
fi
echo "============================================================"
echo ""
echo "Clients must send the chunking context with each observation:"
echo '  {"observations": obs, "inference_delay": 2, "execute_horizon": 3,'
echo '   "prefix_attention_horizon": 13, "actual_action_dim": 14}'
echo ""

ARGS=(
    --server
    --model-path "$MODEL_PATH"
    --data-config "$DATA_CONFIG"
    --embodiment-tag "$EMBODIMENT_TAG"
    --denoising-steps "$DENOISING_STEPS"
    --smooth-option repaint-euler
    --use-tensorrt
    --trt-engine-path "$TRT_ENGINE_PATH"
    --vit-dtype "$VIT_DTYPE"
    --llm-dtype "$LLM_DTYPE"
    --dit-dtype "$DIT_DTYPE"
    --host "$HOST"
    --port "$PORT"
)

if [ "$HTTP_SERVER" = "1" ]; then
    ARGS+=(--http-server)
fi

if [ -n "$API_TOKEN" ]; then
    ARGS+=(--api-token "$API_TOKEN")
fi

exec python scripts/inference_service.py "${ARGS[@]}"
