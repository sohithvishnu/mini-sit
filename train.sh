#!/bin/bash
# =============================================
# 🚀 Mini-SiT Training Script for Procgen (CoinRun)
# ✅ Now with timing and log tracking
# =============================================

# ---------- CONFIG ----------
ENV_NAME="coinrun"
DEVICE_ID=0
TOTAL_STEPS=50000000          # 50M frames total (adjust as needed)
SAVE_DIR="./checkpoints"
LOG_DIR="./logs"
RUN_NAME="mini-sit-progressive"
SEED=42

# Create log directory
mkdir -p ${LOG_DIR}
TIME_LOG="${LOG_DIR}/training_time_$(date '+%Y%m%d_%H%M%S').log"

# ---------- START TIMING ----------
START_TIME=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
echo "🕒 Training started at: ${START_HUMAN}" | tee -a ${TIME_LOG}

# ---------- TRAINING ----------
echo "🧠 Starting Mini-SiT training on ${ENV_NAME}..." | tee -a ${TIME_LOG}

python train.py \
  --env_name ${ENV_NAME} \
  --num_levels 200 \
  --start_level 0 \
  --num_processes 8 \
  --num_steps 256 \
  --num_mini_batch 4 \
  --num_env_steps ${TOTAL_STEPS} \
  --gamma 0.999 \
  --lr 5e-4 \
  --clip_param 0.2 \
  --entropy_coef 0.02 \
  --value_loss_coef 0.5 \
  --save_dir ${SAVE_DIR} \
  --log_dir ${LOG_DIR} \
  --log_interval 10 \
  --save_interval 100 \
  --use_mini_sit \
  --distribution_mode easy \
  --device_id ${DEVICE_ID} \
  --seed ${SEED} 2>&1 | tee -a ${TIME_LOG}

# ---------- END TIMING ----------
END_TIME=$(date +%s)
END_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED=$((END_TIME - START_TIME))

# Convert seconds → hours/mins/secs
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "" | tee -a ${TIME_LOG}
echo "✅ Training finished at: ${END_HUMAN}" | tee -a ${TIME_LOG}
echo "⏱️ Total training time: ${HOURS}h ${MINUTES}m ${SECONDS}s" | tee -a ${TIME_LOG}

# ---------- EVALUATION ----------
echo "🎯 Starting evaluation..." | tee -a ${TIME_LOG}

LATEST_CKPT=$(ls -t ${SAVE_DIR}/agent-${ENV_NAME}-step*.pt | head -n 1)
python evaluate_mini_sit.py \
  --env_name ${ENV_NAME} \
  --model_path ${LATEST_CKPT} \
  --num_episodes 10 \
  --device cuda:${DEVICE_ID} \
  --record_video 2>&1 | tee -a ${TIME_LOG}

echo "🏁 All done! Logs saved in ${TIME_LOG}"