# train_fruitbot.py
import os, time, numpy as np, torch
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
from test import evaluate  # evaluation helper you already have

# -------------------------------------------------
# configuration
# -------------------------------------------------
ENV_NAME = "fruitbot"
TOTAL_STEPS = 50_000_000
EVAL_INTERVAL = 10_000_000      # evaluate + record every 10M env steps
SAVE_INTERVAL = 5_000_000       # save checkpoint every 5M env steps
NUM_PROCESSES = 8
NUM_STEPS = 256                 # PPO rollout length
SEED = 1
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def train():
    # setup
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    log_dir = "./logs"
    ckpt_dir = "./checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.configure(dir=log_dir, format_strs=["csv", "stdout"], log_suffix=ENV_NAME)

    # environments
    venv = ProcgenEnv(num_envs=NUM_PROCESSES,
                      env_name=ENV_NAME,
                      num_levels=500,
                      start_level=0,
                      distribution_mode="easy")
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    envs = VecPyTorchProcgen(venv, device=torch.device(DEVICE))

    obs_shape = envs.observation_space.shape
    num_actions = envs.action_space.n

    # model + PPO
    print(f"🧠 Using Mini-SiT++ model on {ENV_NAME} (device={DEVICE})")
    actor_critic = Policy_MiniSiT(obs_shape, num_actions, DEVICE).to(DEVICE)

    rollouts = RolloutStorage(NUM_STEPS, NUM_PROCESSES,
                              obs_shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              aug_type="crop", split_ratio=0.5)
    rollouts.to(DEVICE)

    # augs
    batch_size = max(1, NUM_PROCESSES * NUM_STEPS // 4)
    aug_list = [data_augs.Crop(batch_size=batch_size)]

    agent = PPO(actor_critic,
                clip_param=0.2, ppo_epoch=4, num_mini_batch=4,
                value_loss_coef=0.5, entropy_coef=0.03,
                lr=3e-4, eps=1e-5, max_grad_norm=0.5,
                aug_id=data_augs.Identity,
                aug_func=aug_list,
                aug_coef=0.1, env_name=ENV_NAME)

    # rollout buffers
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    episode_rewards = deque(maxlen=10)

    updates_per_run = TOTAL_STEPS // (NUM_PROCESSES * NUM_STEPS)
    print(f"🚀 Starting training for {TOTAL_STEPS:,} env steps "
          f"≈ {updates_per_run} PPO updates (n_envs={NUM_PROCESSES}, T={NUM_STEPS})")

    start_time = time.time()

    for j in range(1, updates_per_run + 1):
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

            # IMPORTANT: create masks on the same device as obs / rollouts
            dev = obs.device
            done_t = torch.from_numpy(done).to(dev).float().unsqueeze(-1)
            masks = 1.0 - done_t  # 1 = continue, 0 = episode done
            bad_masks = torch.tensor([[0.0] if ("bad_transition" in i) else [1.0] for i in infos],
                         dtype=torch.float32, device=dev)

            rollouts.insert(obs, rnn_hxs, action, logp, value, reward, masks, bad_masks)
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, gamma=0.999, gae_lambda=0.95)
        value_loss, action_loss, entropy = agent.update(rollouts)
        rollouts.after_update()

        # Logging
        if j % 10 == 0 and episode_rewards:
            mean_r = float(np.mean(episode_rewards))
            logger.logkv("update", j)
            logger.logkv("mean_reward", mean_r)
            logger.dumpkvs()
            elapsed_min = (time.time() - start_time) / 60.0
            print(f"[Update {j:05d}/{updates_per_run}] "
                  f"MeanR={mean_r:.2f} | v_loss={value_loss:.3f} | a_loss={action_loss:.3f} | H={entropy:.3f} | "
                  f"Time={elapsed_min:.1f} min")

        # Periodic checkpoints (by env steps)
        total_env_steps = j * NUM_PROCESSES * NUM_STEPS
        if total_env_steps % SAVE_INTERVAL == 0:
            path = f"{ckpt_dir}/agent-{ENV_NAME}-step{total_env_steps//1_000_000}M.pt"
            torch.save(actor_critic.state_dict(), path)
            print(f"💾 Saved checkpoint → {path}")

        # Evaluation + video every EVAL_INTERVAL env steps
        if total_env_steps % EVAL_INTERVAL == 0:
            ckpt_path = f"{ckpt_dir}/agent-{ENV_NAME}-step{total_env_steps//1_000_000}M.pt"
            torch.save(actor_critic.state_dict(), ckpt_path)
            print(f"🎯 Evaluating after {total_env_steps//1_000_000} M steps...")
            evaluate(env_name=ENV_NAME,
                     model_path=ckpt_path,
                     num_episodes=5,
                     device=DEVICE,
                     record_video=True)
            print("✅ Evaluation complete.\n")

    hours = (time.time() - start_time) / 3600.0
    print(f"🏁 Training finished in {hours:.2f} hours.")


if __name__ == "__main__":
    train()
