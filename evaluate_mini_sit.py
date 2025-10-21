import os
import torch
import numpy as np
import argparse
from procgen import ProcgenEnv

# -----------------------------------------
# ✅ Robust RecordVideo Import Logic
# -----------------------------------------
RecordVideo = None
try:
    # Gymnasium (modern)
    from gymnasium.wrappers import RecordVideo
    print("🎥 Using RecordVideo from gymnasium.wrappers")
except ImportError:
    try:
        # Older Gym (<=0.25)
        from gym.wrappers import RecordVideo
        print("🎥 Using RecordVideo from gym.wrappers")
    except ImportError:
        try:
            # Legacy deep path (gym <=0.23)
            from gym.wrappers.monitoring.video_recorder import VideoRecorder
            RecordVideo = "manual"
            print("⚙️ Using manual VideoRecorder (old Gym version)")
        except Exception:
            print("❌ No RecordVideo available; continuing without video support")
            RecordVideo = None

from ucb_rl2_meta.envs import VecPyTorchProcgen, VecExtractDictObs, VecMonitor, VecNormalize
from ucb_rl2_meta.model_mini_sit import Policy_MiniSiT


@torch.no_grad()
def evaluate(env_name="coinrun",
             model_path="./checkpoints/agent-coinrun-step15000.pt",
             num_episodes=10,
             device="cuda:0",
             render=False,
             record_video=False):
    """
    Evaluate a trained Mini-SiT agent on Procgen.
    Supports:
      --render       : live game window
      --record_video : saves .mp4 videos
    """
    print(f"🎯 Evaluating {model_path} on {env_name}")
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ---------------------------------
    #  Setup environment
    # ---------------------------------
    if record_video and RecordVideo not in [None, "manual"]:
        video_dir = "./videos"
        os.makedirs(video_dir, exist_ok=True)
        venv = ProcgenEnv(
            num_envs=1,
            env_name=env_name,
            num_levels=0,
            start_level=1000,
            distribution_mode="easy"
        )
        venv = RecordVideo(venv, video_dir=video_dir, episode_trigger=lambda x: True)
        print(f"📹 Recording videos to {video_dir}")

    elif record_video and RecordVideo == "manual":
        from gym.wrappers.monitoring import video_recorder
        video_dir = "./videos"
        os.makedirs(video_dir, exist_ok=True)
        venv = ProcgenEnv(
            num_envs=1,
            env_name=env_name,
            num_levels=0,
            start_level=1000,
            distribution_mode="easy"
        )
        video_path = os.path.join(video_dir, f"{env_name}_manual.mp4")
        recorder = video_recorder.VideoRecorder(venv, path=video_path)
        print(f"📹 Using manual video recorder: {video_path}")

    else:
        venv = ProcgenEnv(
            num_envs=1,
            env_name=env_name,
            num_levels=0,
            start_level=1000,
            distribution_mode="easy",
            render_mode="human" if render else None
        )

    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    env = VecPyTorchProcgen(venv, device=device)

    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    # ---------------------------------
    #  Load model
    # ---------------------------------
    model = Policy_MiniSiT(obs_shape, num_actions, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    total_rewards = []

    # ---------------------------------
    #  Evaluation loop
    # ---------------------------------
    for ep in range(num_episodes):
        obs = env.reset()
        done = torch.zeros(1, 1, device=device)
        total_reward = 0.0

        if record_video and RecordVideo == "manual":
            recorder.capture_frame()

        while not done.any():
            value, action, logp, _ = model.act(obs, None, None, deterministic=True)
            obs, reward, done, infos = env.step(action)
            total_reward += reward.item()

            if record_video and RecordVideo == "manual":
                recorder.capture_frame()

        if record_video and RecordVideo == "manual":
            recorder.close()

        total_rewards.append(total_reward)
        print(f"Episode {ep+1}: reward = {total_reward:.2f}")

    print(f"\n✅ Average Reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")

    if record_video:
        print(f"🎥 Videos saved under: {os.path.abspath('./videos')}")

    return total_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="coinrun")
    parser.add_argument("--model_path", type=str, default="./checkpoints/agent-coinrun-step15000.pt")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--render", action="store_true", help="Render live agent view")
    parser.add_argument("--record_video", action="store_true", help="Record gameplay videos")
    args = parser.parse_args()

    evaluate(env_name=args.env_name,
             model_path=args.model_path,
             num_episodes=args.num_episodes,
             device=args.device,
             render=args.render,
             record_video=args.record_video)
