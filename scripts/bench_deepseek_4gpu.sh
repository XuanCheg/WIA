#!/usr/bin/env bash
# Run the 60-task debug bench. Usage: bash scripts/bench_deepseek_4gpu.sh [--dry-run] [--gpus 0,1,2,3]
set -euo pipefail
set +x
WIA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WIA_ROOT"
WIA_PYTHON="${WIA_PYTHON:-/local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python}"
WIA_DRY_RUN=0
for arg in "$@"; do [[ "$arg" != --dry-run ]] || WIA_DRY_RUN=1; done
if [[ "$WIA_DRY_RUN" == 0 && "${WIA_BENCH_FOREGROUND:-0}" != 1 ]]; then
    tmux has-session -t codex-gpu 2>/dev/null || tmux new-session -d -s codex-gpu
    printf -v WIA_COMMAND '%q ' env WIA_BENCH_FOREGROUND=1 bash "$WIA_ROOT/scripts/bench_deepseek_4gpu.sh" "$@"
    WIA_WINDOW="bench-deepseek-$(date +%H%M%S)"
    tmux new-window -t codex-gpu -n "$WIA_WINDOW"
    tmux send-keys -t "codex-gpu:$WIA_WINDOW" -l "$WIA_COMMAND"
    tmux send-keys -t "codex-gpu:$WIA_WINDOW" Enter
    printf 'Started persistent terminal codex-gpu:%s\nAttach: tmux attach -t codex-gpu\n' "$WIA_WINDOW"
    exit 0
fi
if [[ "$WIA_DRY_RUN" == 0 && -z "${AI_GATEWAY_API_KEY:-}" ]]; then
    IFS= read -r AI_GATEWAY_API_KEY < "${WIA_API_KEY_FILE:-$HOME/apikey_vercel.key}" || [[ -n "$AI_GATEWAY_API_KEY" ]]
    export AI_GATEWAY_API_KEY
fi
export LD_LIBRARY_PATH="${WIA_RUNTIME_LIBS:-/local_home2/zhouxinyang/runtime_libs}:${LD_LIBRARY_PATH:-}"
export OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-/local_home2/zhouxinyang/data/behavior}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$WIA_ROOT/configs/esi_vulkan_egl.json}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
"$WIA_PYTHON" -m wia bench --config configs/bench_deepseek_60.yaml --workers 4 \
    --gpus "${WIA_GPUS:-auto}" --output "outputs/deepseek_bench_$(date +%Y%m%d_%H%M%S)" "$@"
