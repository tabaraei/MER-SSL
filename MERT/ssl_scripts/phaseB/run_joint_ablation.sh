#!/bin/bash
# run_joint_ablation.sh — Simonetta-style k,p ratio sweep
# Replicates Simonetta et al. (2024) Fig. 4 to locate the optimal
# IADS-E:PMEmo mixing ratio. Run from phaseB/ with the venv active.
#
#   k = IADS-E ratio kept   |   p = PMEmo-train ratio kept
#
# Each run: 5-fold CV on PMEmo, train augmented with sampled IADS-E,
# evaluated on PMEmo test folds only.
set -e
mkdir -p logs

for kp in "1.0 0.0" "1.0 0.25" "1.0 0.5" "1.0 0.75" "1.0 1.0" \
          "0.75 1.0" "0.5 1.0" "0.25 1.0" "0.0 1.0"; do
    k=$(echo $kp | awk '{print $1}')
    p=$(echo $kp | awk '{print $2}')
    echo "=== Running k=$k p=$p ==="
    python mainB.py \
        --encoder dual_joint --mode kfold --epochs 100 \
        --feat_path pmemo_mert_all_layers.pt \
        --w2v_path  pmemo_wav2vec_all_layers.pt \
        --iadse_mert iadse_mert_all_layers.pt \
        --iadse_w2v  iadse_wav2vec_all_layers.pt \
        --iadse_csv  iadse_labels.csv \
        --csv_path /datasets/emotions/PMEmo2019/annotations/static_annotations.csv \
        --mix_k $k --mix_p $p --eval_on pmemo \
        2>&1 | tee logs/ablation_k${k}_p${p}.log
done

echo "=== Sweep complete. Summarize with: python summarize_ablation.py ==="
