# ==========================================================
# 🍎 Mini-SiT v4.3 Training Script for Procgen FruitBot
# 🧠 FP32 Training + Transfer Learning + Robust Eval Triggers
# ==========================================================

import os, time, torch, numpy as np
from collections import deque
from procgen import ProcgenEnv
from baselines import logger
from baselines.common.vec_env.vec_monitor import VecMonitor
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.vec_remove_dict_obs import VecExtractDictObs

from ucb_rl2_meta.algo.PPO import PPO
from ucb_rl2_meta.envs import VecPyTorchProcgen
from ucb_rl2_meta.storage import RolloutStorage
from ucb_rl2_meta.model_mini_sit import Policy_MiniSiT
import data_augs
from evaluate_mini_sit import evaluate

# ---------------- CONFIG ----------------
ENV_NAME       = "fruitbot"
TOTAL_STEPS    = 25_000_000
EVAL_INTERVAL  = 5_000_000        # target ~5M-env-step evaluations
SAVE_UPDATES   = 100              # save checkpoint every 100 PPO updates
NUM_PROCESSES  = 4
NUM_STEPS      = 512
SEED           = 42
LR             = 2e-4
GAMMA          = 0.999
GAE_LAMBDA     = 0.95
ENT_INIT       = 0.1
ENT_FINAL      = 0.02
DEVICE         = "cuda:0" if torch.cuda.is_available() else "cpu"

def train():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    dev = torch.device(DEVICE)

    log_dir, ckpt_dir = "./logs", "./checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.configure(dir=log_dir, format_strs=["csv", "stdout"], log_suffix=ENV_NAME)

    # ---------------- Environment ----------------
    venv = ProcgenEnv(num_envs=NUM_PROCESSES,
                      env_name=ENV_NAME,
                      num_levels=500,
                      start_level=0,
                      distribution_mode="easy")
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=True, ret=True)
    envs = VecPyTorchProcgen(venv, device=dev)

    obs_shape = envs.observation_space.shape
    num_actions = envs.action_space.n


    # ✅ Reset reward normalization (important when transferring checkpoints)
    envs.venv.ret_rms.reset()
    print("♻️ Reset reward normalization stats.")

    # ---------------- Model + PPO ----------------
    print(f"🧠 Using Mini-SiT v4.3 on {ENV_NAME} ({DEVICE})")
    actor_critic = Policy_MiniSiT(obs_shape, num_actions, dev).to(dev)

    rollouts = RolloutStorage(NUM_STEPS, NUM_PROCESSES,
                              obs_shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              aug_type="crop", split_ratio=0.5)
    rollouts.to(dev)

    batch_size = max(1, NUM_PROCESSES * NUM_STEPS // 4)
    aug_list = [data_augs.Crop(batch_size=batch_size)]

    agent = PPO(actor_critic,
                clip_param=0.1, ppo_epoch=4, num_mini_batch=4,
                value_loss_coef=0.5, entropy_coef=ENT_INIT,
                lr=LR, eps=1e-5, max_grad_norm=0.5,
                aug_id=data_augs.Identity,
                aug_func=aug_list, aug_coef=0.1,
                env_name=ENV_NAME)

    # ---------------- Transfer Learning / Resume ----------------
    resume_ckpt = "./checkpoints/agent-fruitbot-update11000.pt"
    print(f"♻️ Loading encoder only from {resume_ckpt}")
    state_dict = torch.load(resume_ckpt, map_location=dev)
    model_dict = actor_critic.state_dict()

    # keep encoder.* only
    pretrained = {k: v for k, v in state_dict.items() if k.startswith("encoder.")}
    model_dict.update(pretrained)
    actor_critic.load_state_dict(model_dict)
    print("🔄 Encoder restored, policy/value heads reinitialized.")


    # ---------------- Training Loop ----------------
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    episode_rewards = deque(maxlen=20)

    STEPS_PER_UPDATE = NUM_PROCESSES * NUM_STEPS
    UPDATES_TOTAL = TOTAL_STEPS // STEPS_PER_UPDATE
    EVAL_EVERY_UPDATES = max(1, int(np.ceil(EVAL_INTERVAL / STEPS_PER_UPDATE)))
    next_eval_update = EVAL_EVERY_UPDATES
    next_save_update = SAVE_UPDATES

    print(f"🚀 Starting training for {TOTAL_STEPS:,} env steps "
          f"(≈ {UPDATES_TOTAL} PPO updates, {STEPS_PER_UPDATE} steps/update)")
    print(f"🧪 Evaluation every ~{EVAL_INTERVAL:,} env steps "
          f"(≈ every {EVAL_EVERY_UPDATES} updates): will run at updates "
          f"{[EVAL_EVERY_UPDATES * k for k in range(1, (UPDATES_TOTAL // EVAL_EVERY_UPDATES) + 1)]}")

    start_time = time.time()

    for j in range(1, UPDATES_TOTAL + 1):
        # Collect rollout
        for step in range(NUM_STEPS):
            with torch.no_grad():
                value, action, logp, rnn_hxs = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if "episode" in info:
                    episode_rewards.append(info["episode"]["r"])

            done_t = torch.from_numpy(done).to(dev).float().unsqueeze(-1)
            masks = 1.0 - done_t
            bad_masks = torch.tensor([[0.0] if ("bad_transition" in i) else [1.0]
                                      for i in infos], dtype=torch.float32, device=dev)
            rollouts.insert(obs, rnn_hxs, action, logp, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, gamma=GAMMA, gae_lambda=GAE_LAMBDA)

        # PPO update
        value_loss, action_loss, entropy = agent.update(rollouts)
        rollouts.after_update()

        # Entropy anneal
        ent_coef = ENT_FINAL + (ENT_INIT - ENT_FINAL) * np.exp(-j / 1500)
        agent.entropy_coef = float(ent_coef)

        # Logging
        if j % 10 == 0 and episode_rewards:
            mean_r = float(np.mean(episode_rewards))
            logger.logkv("update", j)
            logger.logkv("mean_reward", mean_r)
            logger.dumpkvs()
            elapsed = (time.time() - start_time) / 60
            print(f"[Upd {j:05d}/{UPDATES_TOTAL}] R={mean_r:.2f} "
                  f"| VLoss={value_loss:.3f} | Ent={entropy:.3f} "
                  f"| EntCoef={agent.entropy_coef:.3f} | Time={elapsed:.1f}m")

        # -------- Frequent Checkpoints (by updates) --------
        if j >= next_save_update:
            path = f"{ckpt_dir}/agent-{ENV_NAME}-update{j:05d}.pt"
            torch.save(actor_critic.state_dict(), path)
            print(f"💾 Saved checkpoint → {path}")
            next_save_update += SAVE_UPDATES

        # -------- Evaluation with moving threshold --------
        if j >= next_eval_update:
            total_steps = j * STEPS_PER_UPDATE
            ckpt_path = f"{ckpt_dir}/agent-{ENV_NAME}-step{total_steps//1_000_000}M.pt"
            torch.save(actor_critic.state_dict(), ckpt_path)
            print(f"🎯 Evaluating after ~{total_steps//1_000_000}M steps (update {j})...")
            evaluate(env_name=ENV_NAME,
                     model_path=ckpt_path,
                     num_episodes=5,
                     device=DEVICE,
                     record_video=True)
            print("✅ Evaluation complete.\n")
            next_eval_update += EVAL_EVERY_UPDATES

    hours = (time.time() - start_time) / 3600.0
    print(f"🏁 Training finished in {hours:.2f} hours.")

if __name__ == "__main__":
    train()
