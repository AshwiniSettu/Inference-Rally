#!/usr/bin/env bash
# Start the local llama.cpp model server for PLAY mode on Apple Silicon (M1/M2/M3).
#
#   Llama-3.2-1B-Instruct (Q4_0, ~0.8 GB) on the GPU via Metal -> http://127.0.0.1:8102/v1
#
# The model downloads from Hugging Face on the first run and is cached afterwards.
# Logs go to logs/llama-<port>.log. Ctrl+C stops the server.
#
# Optional: CPU_TOO=1 ./scripts/start_models_mac.sh also starts the same model on the
# CPU only (port 8103), for a GPU-vs-CPU rehearsal.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

MODEL_REPO="bartowski/Llama-3.2-1B-Instruct-GGUF"
MODEL_FILE="Llama-3.2-1B-Instruct-Q4_0.gguf"
COMMON=(--hf-repo "$MODEL_REPO" --hf-file "$MODEL_FILE" --host 127.0.0.1 --ctx-size 2048 --parallel 1)

if ! command -v llama-server >/dev/null 2>&1; then
  echo "llama-server not found. Install llama.cpp first:  brew install llama.cpp" >&2
  exit 1
fi

wait_ready() {
  local port=$1
  printf "Waiting for model on :%s " "$port"
  for _ in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo " ready."
      return 0
    fi
    printf "."
    sleep 1
  done
  echo " timed out — see $LOG_DIR/llama-$port.log" >&2
  return 1
}

PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

# GPU racer: offload every layer to Metal.
llama-server "${COMMON[@]}" --port 8102 --n-gpu-layers 99 >"$LOG_DIR/llama-8102.log" 2>&1 &
PIDS+=($!)
# Wait before starting anything else: the first run downloads the model, and two
# processes downloading the same file at once would clash.
wait_ready 8102

if [[ "${CPU_TOO:-0}" == "1" ]]; then
  # CPU racer: same model, no GPU layers, 4 performance cores.
  llama-server "${COMMON[@]}" --port 8103 --n-gpu-layers 0 --threads 4 >"$LOG_DIR/llama-8103.log" 2>&1 &
  PIDS+=($!)
  wait_ready 8103
fi

echo "Model server(s) running. Switch the game to PLAY mode (key G). Ctrl+C to stop."
wait
