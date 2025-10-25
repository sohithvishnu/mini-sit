#!/bin/bash
# =============================================
# 🍎 Mini-SiT Training Script for Procgen (FruitBot)
# ✅ Includes timing, logs, and evaluation video
# =============================================

# ---------- CONFIG ----------
ENV_NAME="fruitbot"
DEVICE_ID=0
TOTAL_STEPS=50000000           # 50M frames total
SAVE_DIR="./checkpoints"
LOG_DIR="./logs"
RUN_NAME="mini-sit-fruitbot"
SEED=42

# ---------- PREPARE ----------
mkdir -p ${LOG_DIR}
mkdir -p ${SAVE_DIR}

TIME_LOG="${LOG_DIR}/training_time_${ENV_NAME}_$(date '+%Y%m%d_%H%M%S').log"

# ---------- START TIMING ----------
START_TIME=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')
echo "🕒 Training started at: ${START_HUMAN}" | tee -a ${TIME_LOG}
echo "🚀 Environment: ${ENV_NAME}" | tee -a ${TIME_LOG}
echo "💻 Device: cuda:${DEVICE_ID}" | tee -a ${TIME_LOG}
echo "🎯 Total Steps: ${TOTAL_STEPS}" | tee -a ${TIME_LOG}
echo "" | tee -a ${TIME_LOG}

# ---------- TRAINING ----------
echo "🧠 Starting Mini-SiT training on ${ENV_NAME}..." | tee -a ${TIME_LOG}
echo "----------------------------------------------" | tee -a ${TIME_LOG}

# Redirect stdout + stderr into both console & log file
python train_fruitbot.py 2>&1 | tee -a ${TIME_LOG}

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
echo "" | tee -a ${TIME_LOG}
echo "🎯 Starting evaluation & video recording..." | tee -a ${TIME_LOG}

LATEST_CKPT=$(ls -t ${SAVE_DIR}/agent-${ENV_NAME}-step*.pt 2>/dev/null | head -n 1)

if [ -z "$LATEST_CKPT" ]; then
  echo "❌ No checkpoint found in ${SAVE_DIR}!" | tee -a ${TIME_LOG}
else
  python evaluate_mini_sit.py \
    --env_name ${ENV_NAME} \
    --model_path ${LATEST_CKPT} \
    --num_episodes 10 \
    --device cuda:${DEVICE_ID} \
    --record_video 2>&1 | tee -a ${TIME_LOG}
  echo "🎥 Evaluation complete, videos saved in ./videos/" | tee -a ${TIME_LOG}
fi

echo "" | tee -a ${TIME_LOG}
echo "🏁 All done! Logs saved in ${TIME_LOG}" | tee -a ${TIME_LOG}
