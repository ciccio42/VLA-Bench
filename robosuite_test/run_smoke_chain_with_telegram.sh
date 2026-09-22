#!/bin/bash
# Orchestrates the sequential smoke-test chain for the 4 remaining Interleave-VLA
# variants (rm_0_5_10_15, rm_12_13_14_15, rm_central_spawn, removed_spawn_regions),
# submitting the last 2 once slots free up (account is capped at 10 concurrent
# submitted jobs), then waits for each job in order and reports the CORRECT
# success rate (computed from raw info_*.json, not the harness's own buggy
# cumulative-average log lines) to Telegram after each one finishes.
set -uo pipefail
cd /mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/VLA-Benchmark/robosuite_test

source /mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Interleave-VLA/open-pi-zero/slurm/.telegram_credentials

send_telegram() {
    local msg="$1"
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d chat_id="${TELEGRAM_CHAT_ID}" \
        -d text="${msg}" >/dev/null
}

INTERLEAVE_DIR="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Interleave-VLA/open-pi-zero"
TFDS_DIR="/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Interleave-VLA/tensorflow_datasets/sim"

wait_for_slot_and_submit() {
    # args: dependency_jobid(or "") job_name checkpoint train_cfg stats port task_suite
    local dep="$1" name="$2" ckpt="$3" traincfg="$4" stats="$5" port="$6" suite="$7"
    while true; do
        n=$(squeue -A did_robot_learning_359 -h | wc -l)
        if [ "$n" -lt 10 ]; then
            local dep_args=()
            if [ -n "$dep" ]; then
                dep_args=(--dependency=afterany:"$dep")
            fi
            local jobid
            jobid=$(sbatch --parsable "${dep_args[@]}" --job-name="$name" --time=00:40:00 \
                run_interleave_vla_eval.sh models/interleave_vla_eval_config.yml \
                "$ckpt" "$traincfg" "$stats" "$port" 0 1 "$suite" false -1)
            echo "[SUBMITTED] $name -> job $jobid (depends on: ${dep:-none})"
            echo "$jobid"
            return 0
        fi
        sleep 20
    done
}

# --- submit remaining 2 jobs, chained ---
JOBID3=$(wait_for_slot_and_submit "567200" "smoke-rm_central_spawn" \
    "$INTERLEAVE_DIR/logs/train/ur5e_sim_interleave_grounding_bin_rm_central_spawn_train_tp4_beta/2026-09-21_13-20_42/checkpoint/step14500.pt" \
    "$INTERLEAVE_DIR/config/train/interleaved_ur5e_sim_grounding_bin_rm_central_spawn.yaml" \
    "$TFDS_DIR/rm_central_spawn/ur5e_sim_interleave_grounding_bin/0.1.0/dataset_statistics_15ad8c14daf169dea6efb0817ad03765dfe6fa414ce569cb4b75c5afdf0d6d4e.json" \
    8783 "ur5e_pick_place_rm_central_spawn" | tail -1)

JOBID4=$(wait_for_slot_and_submit "$JOBID3" "smoke-removed_spawn_regions" \
    "$INTERLEAVE_DIR/logs/train/ur5e_sim_interleave_grounding_bin_removed_spawn_regions_train_tp4_beta/2026-09-21_13-20_42/checkpoint/step12000.pt" \
    "$INTERLEAVE_DIR/config/train/interleaved_ur5e_sim_grounding_bin_removed_spawn_regions.yaml" \
    "$TFDS_DIR/removed_spawn_regions/ur5e_sim_interleave_grounding_bin/0.1.0/dataset_statistics_28dc174b22172f49f11885d4ce4287f3f1ed4dd54d892cd440291ffa332c5f7b.json" \
    8784 "ur5e_pick_place_removed_spawn_regions" | tail -1)

echo "[CHAIN] 567199 -> 567200 -> ${JOBID3} -> ${JOBID4}"

# --- wait for each remaining job in order, compute correct success rate, report to Telegram ---
# (567199/rm_0_5_10_15 already finished and was reported manually before this job was
# submitted -- see chat history 2026-09-21: Success rate 0.8333, 10/12 trials -- so it is
# deliberately NOT in this loop.)
declare -A VARIANT_NAME=(
    [567200]="rm_12_13_14_15 (step16500)"
    ["$JOBID3"]="rm_central_spawn (step14500)"
    ["$JOBID4"]="removed_spawn_regions (step12000)"
)
declare -A VARIANT_CKPT=(
    [567200]="$INTERLEAVE_DIR/logs/train/ur5e_sim_interleave_grounding_bin_rm_12_13_14_15_train_tp4_beta/2026-09-21_13-20_42/checkpoint/step16500.pt"
    ["$JOBID3"]="$INTERLEAVE_DIR/logs/train/ur5e_sim_interleave_grounding_bin_rm_central_spawn_train_tp4_beta/2026-09-21_13-20_42/checkpoint/step14500.pt"
    ["$JOBID4"]="$INTERLEAVE_DIR/logs/train/ur5e_sim_interleave_grounding_bin_removed_spawn_regions_train_tp4_beta/2026-09-21_13-20_42/checkpoint/step12000.pt"
)

for jobid in 567200 "$JOBID3" "$JOBID4"; do
    variant="${VARIANT_NAME[$jobid]}"
    echo "[WAITING] job $jobid ($variant)"
    while true; do
        st=$(sacct -j "$jobid" --format=State --noheader | head -1 | xargs)
        if [[ "$st" == "COMPLETED" || "$st" == "FAILED" || "$st" == "TIMEOUT" || "$st" == "CANCELLED"* ]]; then
            echo "[JOB STATE] $jobid ($variant): $st"
            break
        fi
        sleep 20
    done

    # EVAL_SAVE_DIR derivation must match run_interleave_vla_eval.sh's own logic exactly:
    # dirname(checkpoint)/basename(checkpoint,.pt)_eval
    ckpt="${VARIANT_CKPT[$jobid]}"
    ckpt_basename=$(basename "$ckpt" .pt)
    model_path="$(dirname "$ckpt")/${ckpt_basename}_eval"
    rollout_dir=$(find "$model_path" -maxdepth 1 -type d -iname "rollout_*" 2>/dev/null | head -1)

    if [ -z "$rollout_dir" ] || [ ! -d "$rollout_dir" ]; then
        msg="SMOKE TEST FAILED: ${variant}
Job ${jobid} did not produce rollout results (job state: $(sacct -j "$jobid" --format=State --noheader | head -1 | xargs)). Check slurm-${jobid}.out."
        echo "[NO RESULTS] $variant"
        send_telegram "$msg"
        continue
    fi

    result=$(python3 -c "
import json, glob, os
files = glob.glob('$rollout_dir/info_*.json')
if not files:
    print('NO_DATA')
else:
    reached, picked, success = [], [], []
    for f in files:
        d = json.load(open(f))
        reached.append(d.get('reached', 0))
        picked.append(d.get('picked', 0))
        success.append(d.get('success', 0))
    n = len(files)
    print(f'{n}|{sum(reached)/n:.4f}|{sum(picked)/n:.4f}|{sum(success)/n:.4f}')
" 2>/dev/null)

    if [ "$result" == "NO_DATA" ] || [ -z "$result" ]; then
        msg="SMOKE TEST: ${variant}
Job ${jobid} finished but no info_*.json files found in ${rollout_dir}."
    else
        IFS='|' read -r n reached picked success <<< "$result"
        msg="SMOKE TEST: ${variant}
Trials: ${n}
Reached rate: ${reached}
Picked rate: ${picked}
Success rate: ${success}
(computed from raw per-trial results, not the harness's own buggy cumulative-average log)"
    fi
    echo "[RESULT] $variant: $msg"
    send_telegram "$msg"
done

send_telegram "SMOKE TEST CHAIN COMPLETE: all 4 remaining variants (rm_0_5_10_15, rm_12_13_14_15, rm_central_spawn, removed_spawn_regions) have been evaluated."
echo "[DONE] all 4 smoke tests reported to Telegram"
