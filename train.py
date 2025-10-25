import os
from collections import deque
import numpy as np
import torch
from procgen import ProcgenEnv

import data_augs
from baselines import logger
from baselines.common.vec_env.vec_monitor import VecMonitor
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.vec_remove_dict_obs import VecExtractDictObs
from test import evaluate
from ucb_rl2_meta.algo.PPO import PPO
from ucb_rl2_meta.envs import VecPyTorchProcgen
from ucb_rl2_meta.storage import RolloutStorage
from ucb_rl2_meta import utils
from ucb_rl2_meta.model_mini_sit import Policy_MiniSiT  # <-- import your Mini-SiT policy
import argparse


# ----------------------------
# Argument parser
# ----------------------------
parser = argparse.ArgumentParser(description="Train Mini-SiT on ProcGen")
parser.add_argument("--env_name", type=str, default="coinrun", help="ProcGen environment name")
parser.add_argument("--num_levels", type=int, default=500)
parser.add_argument("--start_level", type=int, default=0)
parser.add_argument("--distribution_mode", type=str, default="easy")
parser.add_argument("--num_processes", type=int, default=8)
parser.add_argument("--num_steps", type=int, default=256)
parser.add_argument("--num_mini_batch", type=int, default=4)
parser.add_argument("--num_env_steps", type=int, default=1_000_000)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--clip_param", type=float, default=0.2)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--value_loss_coef", type=float, default=0.5)
parser.add_argument("--save_dir", type=str, default="./checkpoints")
parser.add_argument("--log_dir", type=str, default="./logs")
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--aug_coef", type=float, default=0.1)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--split_ratio", type=float, default=0.5)
parser.add_argument("--aug_type", type=str, default="crop")
parser.add_argument("--use_mini_sit", action="store_true", help="Use Mini-SiT model")


# ----------------------------
# Augmentation registry
# ----------------------------
aug_to_func = {
    "crop": data_augs.Crop,
    "random-conv": data_augs.RandomConv,
    "color-jitter": data_augs.ColorJitter,
    "flip": data_augs.Flip,
}


# ----------------------------
# Training loop
# ----------------------------
def train(args):
    # ---- Device setup ----
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --- Logging setup ---
    log_dir = os.path.expanduser(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{args.env_name}-miniSiT-s{args.seed}"
    logger.configure(dir=log_dir, format_strs=["csv", "stdout"], log_suffix=log_file)


    # ---- Environment setup ----
    venv = ProcgenEnv(num_envs=args.num_processes,
                      env_name=args.env_name,
                      num_levels=args.num_levels,
                      start_level=args.start_level,
                      distribution_mode=args.distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    envs = VecPyTorchProcgen(venv, device)

    obs_shape = envs.observation_space.shape
    num_actions = envs.action_space.n

    # ---- Model ----
    print("🧠 Using Mini-SiT model")
    actor_critic = Policy_MiniSiT(obs_shape, num_actions, device)
    actor_critic.to(device)

    # ---- PPO setup ----
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              aug_type=args.aug_type, split_ratio=args.split_ratio)

    batch_size = int(args.num_processes * args.num_steps / args.num_mini_batch)
    aug_id = data_augs.Identity
    aug_list = [aug_to_func[t](batch_size=batch_size) for t in aug_to_func.keys()]

    agent = PPO(actor_critic,
                args.clip_param, args.ppo_epoch if hasattr(args, "ppo_epoch") else 4,
                args.num_mini_batch, args.value_loss_coef, args.entropy_coef,
                lr=args.lr, eps=1e-5, max_grad_norm=0.5,
                aug_id=aug_id, aug_func=aug_list,
                aug_coef=args.aug_coef, env_name=args.env_name)

    # ---- Training loop ----
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    episode_rewards = deque(maxlen=10)
    num_updates = args.num_env_steps // args.num_steps // args.num_processes

    for j in range(num_updates):
        for step in range(args.num_steps):
            with torch.no_grad():
                obs_id = aug_id(rollouts.obs[step])
                value, action, action_log_prob, rnn_hxs = actor_critic.act(
                    obs_id, rollouts.recurrent_hidden_states[step], rollouts.masks[step])

            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if "episode" in info:
                    episode_rewards.append(info["episode"]["r"])

            masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info else [1.0] for info in infos])
            rollouts.insert(obs, rnn_hxs, action, action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            obs_id = aug_id(rollouts.obs[-1])
            next_value = actor_critic.get_value(obs_id,
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            mean_r = np.mean(episode_rewards)
            print(f"Update {j} | Mean reward: {mean_r:.2f}")
            logger.logkv("update", j)
            logger.logkv("mean_reward", mean_r)
            logger.dumpkvs()
            
        # ---- Save checkpoint ----
        if j % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"agent-{args.env_name}-step{j}.pt")
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(actor_critic.state_dict(), save_path)
            print(f"💾 Saved model at {save_path}")


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
