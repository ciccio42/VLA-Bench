#!/bin/bash
set -uo pipefail
cd /mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/VLA-Benchmark/robosuite_test

INTERLEAVE_DIR=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Interleave-VLA/open-pi-zero
TFDS_DIR=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Interleave-VLA/tensorflow_datasets/sim
CFG=models/interleave_vla_eval_config.yml
PORT=8790

wait_for_slot() {
  while true; do
    n=$(squeue -u "$USER" -h | wc -l)
    if [ "$n" -lt 10 ]; then
      break
    fi
    sleep 30
  done
}

submit_when_ready() {
  local desc="$1" dep="$2"; shift 2
  local jid=""
  while [ -z "$jid" ]; do
    wait_for_slot
    jid=$(sbatch --parsable --dependency="afterany:$dep" run_interleave_vla_eval.sh "$@" 2>/tmp_submit_err.log)
    if [ -z "$jid" ]; then
      echo "[$(date)] submit failed for [$desc], retrying in 30s: $(cat /tmp_submit_err.log)" >&2
      sleep 30
    fi
  done
  echo "[$(date)] SUBMITTED [$desc] -> job $jid (dep: afterany:$dep)" >&2
  echo "$jid"
}

j1=$(NORMALIZATION_TYPE=bounds submit_when_ready "rm_12_13_14_15 (test, otd=true)" 567954 \
  "$CFG" \
  "$INTERLEAVE_DIR/logs/train/ur5e_sim_interleave_grounding_bin_rm_12_13_14_15_train_tp4_beta/2026-09-22_04-16_42/checkpoint/step26000.pt" \
  "$INTERLEAVE_DIR/config/train/interleaved_ur5e_sim_grounding_bin_rm_12_13_14_15.yaml" \
  "$TFDS_DIR/rm_12_13_14_15/ur5e_sim_interleave_grounding_bin/0.1.0/dataset_statistics_5486b13b0cbc0ed9b526f1edff01b007af73a64ccdde2d00f064bb0411026b2a.json" \
  "$PORT" 0 30 ur5e_pick_place_rm_12_13_14_15 true -1)

j2=$(CSR=false NORMALIZATION_TYPE=bounds submit_when_ready "rm_central_spawn (csr=false)" "$j1" \
  "$CFG" \
  "$INTERLEAVE_DIR/logs/train/ur5e_sim_interleave_grounding_bin_rm_central_spawn_train_tp4_beta/2026-09-22_04-25_42/checkpoint/step19500.pt" \
  "$INTERLEAVE_DIR/config/train/interleaved_ur5e_sim_grounding_bin_rm_central_spawn.yaml" \
  "$TFDS_DIR/rm_central_spawn/ur5e_sim_interleave_grounding_bin/0.1.0/dataset_statistics_15ad8c14daf169dea6efb0817ad03765dfe6fa414ce569cb4b75c5afdf0d6d4e.json" \
  "$PORT" 0 10 ur5e_pick_place_rm_central_spawn false -1)

j3=$(CSR=true NORMALIZATION_TYPE=bounds submit_when_ready "rm_central_spawn (csr=true)" "$j2" \
  "$CFG" \
  "$INTERLEAVE_DIR/logs/train/ur5e_sim_interleave_grounding_bin_rm_central_spawn_train_tp4_beta/2026-09-22_04-25_42/checkpoint/step19500.pt" \
  "$INTERLEAVE_DIR/config/train/interleaved_ur5e_sim_grounding_bin_rm_central_spawn.yaml" \
  "$TFDS_DIR/rm_central_spawn/ur5e_sim_interleave_grounding_bin/0.1.0/dataset_statistics_15ad8c14daf169dea6efb0817ad03765dfe6fa414ce569cb4b75c5afdf0d6d4e.json" \
  "$PORT" 0 10 ur5e_pick_place_rm_central_spawn false -1)

echo "ALL REMAINING CHAIN JOBS SUBMITTED: $j1 $j2 $j3"
