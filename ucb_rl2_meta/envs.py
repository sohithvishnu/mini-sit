import gym
import torch
from gym.spaces.box import Box

from baselines.common.vec_env.vec_env import VecEnvWrapper


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImageProcgen(TransposeObs):
    def __init__(self, env=None, op=[0, 3, 2, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImageProcgen, self).__init__(env)
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[2], obs_shape[1], obs_shape[0]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        if ob.shape[0] == 1:
            ob = ob[0]
        return ob.transpose(self.op[0], self.op[1], self.op[2], self.op[3])


class VecPyTorchProcgen(VecEnvWrapper):
    def __init__(self, venv, device):
        """
        Environment wrapper that returns tensors (for obs and reward)
        """
        super(VecPyTorchProcgen, self).__init__(venv)
        self.device = device

        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [3, 64, 64],
            dtype=self.observation_space.dtype)

    def reset(self):
        obs = self.venv.reset()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor) or len(actions.shape) > 1:
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

import numpy as np

# -----------------------------------------------------------------------------
# Minimal VecExtractDictObs: extracts RGB frames from Procgen's dict obs
# -----------------------------------------------------------------------------
class VecExtractDictObs(VecEnvWrapper):
    def __init__(self, venv, key="rgb"):
        super(VecExtractDictObs, self).__init__(venv)
        self.key = key
        self.observation_space = venv.observation_space.spaces[key]

    def reset(self):
        obs = self.venv.reset()
        return obs[self.key]

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs[self.key], rewards, dones, infos


# -----------------------------------------------------------------------------
# Minimal VecMonitor: track rewards for logging
# -----------------------------------------------------------------------------
class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, filename=None, keep_buf=100):
        super(VecMonitor, self).__init__(venv)
        self.rewards = np.zeros(venv.num_envs, dtype=np.float32)
        self.episode_returns = []
        self.keep_buf = keep_buf

    def reset(self):
        obs = self.venv.reset()
        self.rewards.fill(0)
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.rewards += rewards
        for i, done in enumerate(dones):
            if done:
                ep_info = {"episode": {"r": self.rewards[i].item()}}
                infos[i].update(ep_info)
                self.episode_returns.append(self.rewards[i])
                self.rewards[i] = 0
                if len(self.episode_returns) > self.keep_buf:
                    self.episode_returns.pop(0)
        return obs, rewards, dones, infos


# -----------------------------------------------------------------------------
# Minimal VecNormalize: normalize rewards (no obs normalization)
# -----------------------------------------------------------------------------
class VecNormalize(VecEnvWrapper):
    def __init__(self, venv, ob=True, ret=True, cliprew=10.0):
        super(VecNormalize, self).__init__(venv)
        self.ret = ret
        self.cliprew = cliprew
        self.ret_rms = np.zeros(1)

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        if self.ret:
            rews = np.clip(rews, -self.cliprew, self.cliprew)
        return obs, rews, dones, infos
