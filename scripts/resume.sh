#!/usr/bin/env bash
#
# Run any resumable script to completion, surviving the intermittent
# segfaults documented in docs/defect_log.md D24.
#
#   bash scripts/resume.sh scripts/generate_dataset.py --instrument hobbyist \
#        --n-samples 16000 --chunk-size 500 --output data/dataset_hobbyist.npz
#
#   bash scripts/resume.sh scripts/run_baseline.py --chunk-size 20
#
# Adds --max-chunks-per-run 1 automatically, so each chunk gets a fresh
# interpreter and a crash costs one chunk rather than the whole run.
#
# Retries ONLY on "chunks remain" (exit 2) or a signal death (>=128).  Any
# other status is a real error -- missing dependency, bad argument -- and stops
# immediately, so a typo cannot spin the terminal forever.
#
# Tuning:
#   SLEEP_SECONDS=3    pause between attempts; raise if the machine struggles
#   MAX_ATTEMPTS=400   give up eventually rather than looping all night
#
# Ctrl-C is always safe: completed chunks are on disk and are not redone.

set -uo pipefail

PYTHON="${PYTHON:-python}"
SLEEP_SECONDS="${SLEEP_SECONDS:-3}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-400}"

if [ "$#" -eq 0 ]; then
    echo "usage: bash scripts/resume.sh <script.py> [args...]"
    exit 64
fi

if ! "$PYTHON" -c "import numpy" >/dev/null 2>&1; then
    echo "error: numpy is not importable by '$PYTHON'."
    echo "The virtualenv is probably not active:"
    echo "  source .venv/Scripts/activate     # Git Bash on Windows"
    exit 1
fi

echo "resuming: $*"
echo "one chunk per process, ${SLEEP_SECONDS}s between attempts."
echo "Ctrl-C is safe - finished chunks are not redone."
echo

attempt=0
crashes=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    "$PYTHON" "$@" --max-chunks-per-run 1
    status=$?

    case "$status" in
        0)
            [ "$crashes" -gt 0 ] && echo && \
                echo "complete after $crashes crash(es) - recovered by resuming."
            exit 0
            ;;
        2)
            ;;                                   # more chunks; continue quietly
        *)
            if [ "$status" -lt 128 ]; then
                echo
                echo "exited $status - that is an error, not a crash. Stopping."
                exit "$status"
            fi
            crashes=$((crashes + 1))
            echo "  (crashed with signal status $status - resuming)"
            ;;
    esac

    attempt=$((attempt + 1))
    [ "$SLEEP_SECONDS" != "0" ] && sleep "$SLEEP_SECONDS"
done

echo "gave up after $MAX_ATTEMPTS attempts."
exit 1
