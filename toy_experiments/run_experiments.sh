#!/bin/bash
SCRIPT="mhc_universal.py"

# --- HYPOTHESIS 1: LENGTH BREAKING POINT ---
# GPU 0: Short/Medium
# Increased steps to 8000 to ensure convergence if early stopping doesn't trigger
CUDA_VISIBLE_DEVICES=0 python $SCRIPT --exp_id "H1_SEQ128" --mode std_vs_mhc --seq 128 --task copy --steps 8000 &
CUDA_VISIBLE_DEVICES=1 python $SCRIPT --exp_id "H1_SEQ256" --mode std_vs_mhc --seq 256 --task copy --steps 8000 &
# GPU 0: Long Sequence (Hard mode, more steps)
CUDA_VISIBLE_DEVICES=2 python $SCRIPT --exp_id "H1_SEQ512" --mode std_vs_mhc --seq 512 --task copy --steps 12000 &

# # --- HYPOTHESIS 2: VOCAB CAPACITY ---
# # GPU 1
# CUDA_VISIBLE_DEVICES=3 python $SCRIPT --exp_id "H2_VOC1024" --mode std_vs_mhc --vocab 1024 --seq 256 --task copy --steps 10000 &
# CUDA_VISIBLE_DEVICES=4 python $SCRIPT --exp_id "H2_VOC4096" --mode std_vs_mhc --vocab 4096 --seq 256 --task copy --steps 12000 &

# # --- HYPOTHESIS 3 & 4: DEPTH & ABLATION ---
# # GPU 2
# CUDA_VISIBLE_DEVICES=5 python $SCRIPT --exp_id "H3_LAYERS16" --mode std_vs_mhc --n_layers 16 --d_model 128 --seq 256 --task copy --steps 10000 &
# CUDA_VISIBLE_DEVICES=6 python $SCRIPT --exp_id "H4_NOMIXER" --mode mhc_vs_nomixer --seq 512 --task copy --steps 12000 &

# # --- HYPOTHESIS 5: RECALL TASKS ---
# # GPU 3
# CUDA_VISIBLE_DEVICES=7 python $SCRIPT --exp_id "H5_MQAR" --mode std_vs_mhc --task mqar --seq 256 --d_model 128 --steps 10000 &
# CUDA_VISIBLE_DEVICES=7 python $SCRIPT --exp_id "H5_FUZZY" --mode std_vs_mhc --task fuzzy_recall --seq 256 --d_model 128 --steps 10000 &

wait
echo "ALL EXPERIMENTS COMPLETED."