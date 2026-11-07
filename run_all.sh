#!/usr/bin/env bash
# ==============================================================================
# ArcMDTI One-Click Background Training, Resume & Evaluation Script (run_all.sh)
# ==============================================================================
# Usage:
#   1. Default: Run / Resume full pipeline in background (safe to close SSH terminal!)
#      bash run_all.sh
#
#   2. Specify GPU and run / resume in background:
#      bash run_all.sh --gpu 0
#
#   3. Run only the new Q1 SCI upgrade experiments in background:
#      bash run_all.sh --gpu 0 --mode paper_upgrade
#
#   4. Force a complete fresh re-run from Epoch 1 (clears old checkpoints):
#      bash run_all.sh --gpu 0 --reset-all
#
#   5. Watch real-time training logs (Ctrl+C ONLY exits viewing, does NOT stop training):
#      tail -f arcmdti_nohup.log
#      # or:
#      bash run_all.sh --logs
#
#   6. Check live training status & latest epoch metrics:
#      bash run_all.sh --status
#
#   7. Run in foreground (for interactive debugging):
#      bash run_all.sh --fg --gpu 0
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/arcmdti_nohup.log"
PID_FILE="$SCRIPT_DIR/run_all.pid"

# ------------------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------------------
IS_WORKER=0
FOREGROUND=0
SHOW_STATUS=0
SHOW_LOGS=0
STOP_JOB=0
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
MODE="all"
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --_worker)
            IS_WORKER=1
            shift
            ;;
        --fg|--foreground)
            FOREGROUND=1
            shift
            ;;
        --status)
            SHOW_STATUS=1
            shift
            ;;
        --logs|--tail)
            SHOW_LOGS=1
            shift
            ;;
        --stop|--kill)
            STOP_JOB=1
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            FORWARD_ARGS+=("--gpu" "$2")
            shift 2
            ;;
        --mode)
            MODE="$2"
            FORWARD_ARGS+=("--mode" "$2")
            shift 2
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

# Ensure --mode is passed to worker/pipeline if not already in FORWARD_ARGS
HAS_MODE=0
for arg in "${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}"; do
    if [[ "$arg" == "--mode" ]]; then
        HAS_MODE=1
        break
    fi
done
if [[ "$HAS_MODE" -eq 0 ]]; then
    FORWARD_ARGS=("--mode" "$MODE" "${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}")
fi

HAS_GPU=0
for arg in "${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}"; do
    if [[ "$arg" == "--gpu" ]]; then
        HAS_GPU=1
        break
    fi
done
if [[ "$HAS_GPU" -eq 0 ]]; then
    FORWARD_ARGS=("--gpu" "$GPU_ID" "${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}")
fi

# ------------------------------------------------------------------------------
# Helper: --stop (Stop running background job)
# ------------------------------------------------------------------------------
if [[ "$STOP_JOB" -eq 1 ]]; then
    if [[ -f "$PID_FILE" ]]; then
        OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
            echo "[STOP] Stopping background ArcMDTI pipeline (PID: $OLD_PID)..."
            kill "$OLD_PID" 2>/dev/null || true
            echo "[STOP] Signal sent. Latest epoch checkpoints are preserved for seamless resume."
            exit 0
        fi
    fi
    echo "[INFO] No running background ArcMDTI job found."
    exit 0
fi

# ------------------------------------------------------------------------------
# Helper: --status (Inspect live process and per-epoch metrics)
# ------------------------------------------------------------------------------
if [[ "$SHOW_STATUS" -eq 1 ]]; then
    echo "======================================================================"
    echo " ArcMDTI Training Status & Per-Epoch Progress"
    echo "======================================================================"
    if [[ -f "$PID_FILE" ]]; then
        OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
            echo "  Process Status : RUNNING (PID: $OLD_PID)"
        else
            echo "  Process Status : NOT RUNNING (stale PID file: $OLD_PID)"
        fi
    else
        echo "  Process Status : NOT RUNNING (or completed)"
    fi
    echo "  Main Log File  : $LOG_FILE"
    echo "----------------------------------------------------------------------"
    if compgen -G "$SCRIPT_DIR/result/*/run_status.json" > /dev/null; then
        echo "  Latest Experiment Status (run_status.json):"
        ls -t "$SCRIPT_DIR"/result/*/run_status.json 2>/dev/null | head -n 3 | while read -r sj; do
            echo "  --> $sj"
            cat "$sj"
            echo ""
        done
    fi
    if compgen -G "$SCRIPT_DIR/log/*_train_log.txt" > /dev/null; then
        LATEST_TRAIN_LOG="$(ls -t "$SCRIPT_DIR"/log/*_train_log.txt 2>/dev/null | head -n 1)"
        echo "----------------------------------------------------------------------"
        echo "  Latest Per-Epoch Metric Log ($LATEST_TRAIN_LOG):"
        tail -n 12 "$LATEST_TRAIN_LOG"
    fi
    echo "======================================================================"
    exit 0
fi

# ------------------------------------------------------------------------------
# Helper: --logs (Tail real-time training log; Ctrl+C only exits viewing)
# ------------------------------------------------------------------------------
if [[ "$SHOW_LOGS" -eq 1 ]]; then
    echo "======================================================================"
    echo " Watching live training log: $LOG_FILE"
    echo " (Press Ctrl+C at any time to exit viewing — training continues in background)"
    echo "======================================================================"
    touch "$LOG_FILE"
    exec tail -n 50 -f "$LOG_FILE"
fi

# ------------------------------------------------------------------------------
# Auto-Nohup Launcher (Default when running `bash run_all.sh`)
# ------------------------------------------------------------------------------
if [[ "$IS_WORKER" -eq 0 && "$FOREGROUND" -eq 0 ]]; then
    if [[ -f "$PID_FILE" ]]; then
        OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
            echo "======================================================================"
            echo "[INFO] ArcMDTI pipeline is ALREADY running in the background!"
            echo "  PID      : $OLD_PID"
            echo "  Log file : $LOG_FILE"
            echo ""
            echo "  To watch live progress (Ctrl+C only exits viewing, does NOT stop training):"
            echo "    tail -f $LOG_FILE"
            echo "  To inspect per-epoch evaluation metrics:"
            echo "    bash run_all.sh --status"
            echo "  To stop the current run:"
            echo "    kill $OLD_PID"
            echo "======================================================================"
            exit 0
        fi
    fi

    echo "" >> "$LOG_FILE"
    echo "======================================================================" >> "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching run_all.sh in background (nohup)" >> "$LOG_FILE"
    echo "Args: ${FORWARD_ARGS[*]}" >> "$LOG_FILE"
    echo "======================================================================" >> "$LOG_FILE"

    nohup bash "$SCRIPT_DIR/run_all.sh" --_worker "${FORWARD_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
    BG_PID=$!
    echo "$BG_PID" > "$PID_FILE"

    echo "======================================================================"
    echo "[LAUNCHED] ArcMDTI Pipeline is now running in the background!"
    echo "  PID             : $BG_PID (saved to $PID_FILE)"
    echo "  GPU ID          : $GPU_ID"
    echo "  Pipeline Mode   : $MODE"
    echo "  Main Log File   : $LOG_FILE"
    echo "  Per-Epoch Logs  : $SCRIPT_DIR/log/*_train_log.txt"
    echo "                    $SCRIPT_DIR/log/*_epoch_metrics.csv"
    echo "                    $SCRIPT_DIR/result/*/epoch_metrics_all_folds.csv"
    echo ""
    echo "  Useful commands:"
    echo "    • Watch live logs (Ctrl+C only exits viewing, will NOT stop training):"
    echo "      tail -f $LOG_FILE"
    echo "    • Watch formatted per-epoch evaluation table:"
    echo "      tail -f $SCRIPT_DIR/log/*_train_log.txt"
    echo "    • Check live fold/epoch progress:"
    echo "      bash run_all.sh --status"
    echo "    • Check if process is alive:"
    echo "      ps -p $BG_PID"
    echo "    • Stop background run (can resume anytime by re-running bash run_all.sh):"
    echo "      kill $BG_PID"
    echo "======================================================================"
    echo "You can safely close your SSH terminal now."
    exit 0
fi

# ------------------------------------------------------------------------------
# Worker Execution (Runs inside nohup background process or --fg mode)
# ------------------------------------------------------------------------------
cleanup_pid() {
    if [[ "$IS_WORKER" -eq 1 && -f "$PID_FILE" ]]; then
        rm -f "$PID_FILE"
    fi
}
trap cleanup_pid EXIT

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

echo "======================================================================"
echo " ArcMDTI Automated Training, Resume & Evaluation Worker"
echo " Working Dir : $SCRIPT_DIR"
echo " Python Bin  : $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
echo " GPU Device  : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo " Mode        : $MODE"
echo " Started At  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

echo "[Step 1/2] Executing Master Server Pipeline (Auto-Resume Enabled)..."
"$PYTHON_BIN" -u run_server_pipeline.py "${FORWARD_ARGS[@]}"

echo "[Step 2/2] Verifying Per-Epoch Logs and Final Download Archive..."
if [[ -f "$SCRIPT_DIR/ArcMDTI_Server_Results.tar.gz" ]]; then
    ls -lh "$SCRIPT_DIR/ArcMDTI_Server_Results.tar.gz"
fi

echo "======================================================================"
echo " All ArcMDTI experiments & figures finished successfully!"
echo " Finished At : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Results     : $SCRIPT_DIR/result/"
echo " Epoch Logs  : $SCRIPT_DIR/log/ & $SCRIPT_DIR/result/all_epochs_training_history.csv"
echo " Archive     : $SCRIPT_DIR/ArcMDTI_Server_Results.tar.gz"
echo "======================================================================"

