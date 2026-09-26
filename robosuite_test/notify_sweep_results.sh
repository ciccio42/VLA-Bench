#!/bin/bash
# Watches the 10-job interleave-vla eval sweep chained via SLURM dependencies and posts a
# Telegram message per job when it finishes (parsed final Reached/Picked/Success rates from
# its slurm-<id>.out), plus a final summary when the whole chain completes. Reuses the same
# bot credentials as telegram_bot.sh rather than duplicating the secret.
set -uo pipefail
cd "$(dirname "$0")"

CREDENTIALS_FILE="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Interleave-VLA/open-pi-zero/slurm/.telegram_credentials"
source "$CREDENTIALS_FILE"

CHAIN_LOG="chain_watcher.log"
STATE_FILE="notify_sweep_state.txt"  # "jobid<TAB>description" lines, known jobs
REPORTED_FILE="notify_sweep_reported.txt"
touch "$STATE_FILE" "$REPORTED_FILE"

send_telegram() {
    local msg="$1"
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        --data-urlencode chat_id="${TELEGRAM_CHAT_ID}" \
        --data-urlencode text="${msg}" >/dev/null
}

# Seed the 7 already-submitted jobs.
cat > "$STATE_FILE" << 'EOF'
567948	delta_all
567949	NORMTEST (normal norm)
567950	removed_spawn_regions (csr=false)
567951	removed_spawn_regions (csr=true)
567952	rm_0_5_10_15 (training, otd=false)
567953	rm_0_5_10_15 (test, otd=true)
567954	rm_12_13_14_15 (training, otd=false)
EOF

TOTAL_EXPECTED=10

get_final_rates() {
    local jobid="$1"
    local log="slurm-${jobid}.out"
    if [ ! -f "$log" ]; then
        echo "no log found"
        return
    fi
    local reached picked success
    reached=$(grep "^Reached rate:" "$log" | tail -1)
    picked=$(grep "^Picked rate:" "$log" | tail -1)
    success=$(grep "^Success rate:" "$log" | tail -1)
    if [ -z "$reached$picked$success" ]; then
        echo "no rate lines found in log yet"
    else
        echo "${reached} | ${picked} | ${success}"
    fi
}

send_telegram "Eval sweep watcher started, tracking $(wc -l < "$STATE_FILE") job(s) so far (expecting $TOTAL_EXPECTED total)."

while true; do
    # Discover newly-submitted chain-tail jobs from chain_watcher.log.
    if [ -f "$CHAIN_LOG" ]; then
        while IFS= read -r line; do
            desc=$(echo "$line" | sed -n 's/.*SUBMITTED \[\(.*\)\] -> job \([0-9]*\).*/\1/p')
            jid=$(echo "$line" | sed -n 's/.*SUBMITTED \[\(.*\)\] -> job \([0-9]*\).*/\2/p')
            if [ -n "$jid" ] && ! grep -q "^${jid}	" "$STATE_FILE"; then
                echo -e "${jid}\t${desc}" >> "$STATE_FILE"
                echo "[$(date -Iseconds)] discovered new chain job ${jid} (${desc})"
            fi
        done < "$CHAIN_LOG"
    fi

    while IFS=$'\t' read -r jobid desc; do
        [ -z "$jobid" ] && continue
        grep -q "^${jobid}$" "$REPORTED_FILE" && continue

        state=$(sacct -j "$jobid" --format=State -n -p 2>/dev/null | head -1 | tr -d '|')
        case "$state" in
            COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
                rates=$(get_final_rates "$jobid")
                send_telegram "[${jobid}] ${desc}
State: ${state}
${rates:-no rate lines found in log}"
                echo "${jobid}" >> "$REPORTED_FILE"
                ;;
        esac
    done < "$STATE_FILE"

    n_reported=$(wc -l < "$REPORTED_FILE")
    n_known=$(wc -l < "$STATE_FILE")
    if [ "$n_reported" -ge "$TOTAL_EXPECTED" ] && [ "$n_known" -ge "$TOTAL_EXPECTED" ]; then
        send_telegram "Eval sweep complete: all ${TOTAL_EXPECTED} jobs finished. See individual messages above for per-run results."
        break
    fi

    sleep 60
done
