import gym
from gym import spaces
import torch
import torch.nn.functional as F
import numpy as np


class KeypointEnv(gym.Env):
    def __init__(self, image, gt_coord, is_3d=False, max_steps=50, patch_size=33,
                 heatmap=None, min_steps=3,
                 max_displacement_3d=5.0, max_displacement_2d=8.0,
                 spacing_3d=None, rl_reward_scale=1.0):
        super().__init__()

        if image.dim() > 3:
            image = image.squeeze(0) if image.dim() == 5 else image
            image = image[0] if image.dim() == 4 else image
        if not is_3d and image.dim() == 3 and image.shape[0] <= 3:
            image = image.mean(dim=0)

        self.image = image.float()
        self.image = (self.image - self.image.mean()) / (self.image.std() + 1e-6)
        self.gt_coord = torch.tensor(gt_coord, dtype=torch.float32) if gt_coord is not None else None
        self.is_3d = is_3d
        self.max_steps = max_steps
        self.patch_size = patch_size
        self.min_steps = min_steps
        self.rl_reward_scale = rl_reward_scale

        if heatmap is not None:
            heatmap = heatmap.float()
            if heatmap.min() < 0 or heatmap.max() > 1.5:
                heatmap = torch.sigmoid(heatmap)
        self.heatmap = heatmap

        self.max_displacement = max_displacement_3d if is_3d else max_displacement_2d

        if spacing_3d is None:
            spacing_3d = [1.5, 0.8, 0.8]
        self.spacing = torch.tensor(spacing_3d, dtype=torch.float32) if is_3d else torch.tensor([0.48, 0.48], dtype=torch.float32)

        self.current_pos = None
        self.init_pos = None
        self.steps = 0
        self.prev_dist = None
        self.init_dist = None
        self.best_dist = None 
        self.action_space = spaces.Discrete(5)
        self.stop_action = 4
        self.action_deltas = [
            torch.tensor([-1.0, 0]), torch.tensor([1.0, 0]),
            torch.tensor([0, -1.0]), torch.tensor([0, 1.0]),
            torch.tensor([0, 0])
        ]

    def reset(self, init_pos=None, train_mode=False):
        self.steps = 0
        self.current_pos = torch.tensor(init_pos, dtype=torch.float32)

        if train_mode and self.gt_coord is not None:
            noise_scale = 3.0
            noise = (torch.rand_like(self.current_pos) - 0.5) * 2 * noise_scale
            self.current_pos = self.current_pos + noise
            num_dims = 2
            img_shape = torch.tensor(self.image.shape[-num_dims:], dtype=torch.float32)
            self.current_pos = torch.clamp(self.current_pos, torch.zeros(num_dims), img_shape - 1)

        self.init_pos = self.current_pos.clone()
        if self.gt_coord is not None:
            self.prev_dist = self._get_dist_mm()
            self.init_dist = self.prev_dist
            self.best_dist = self.prev_dist 
        else:
            self.prev_dist = 0
            self.init_dist = 0
            self.best_dist = 0

        return self._get_obs()

    def step(self, action):
        self.steps += 1
        done = False
        reward = 0.0

        if action == self.stop_action:
            done = True
            if self.gt_coord is not None:
                curr_dist = self._get_dist_mm()
                total_improvement = self.init_dist - curr_dist
                if total_improvement > 0.3:
                    reward = 2.0  
                elif total_improvement > 0:
                    reward = 0.5
                else:
                    reward = -1.0 
            return self._get_obs(), reward, done, {}


        if self.steps >= self.max_steps:
            done = True
            return self._get_obs(), -0.5, done, {}

   
        delta = self.action_deltas[action]
        new_pos = self.current_pos + delta

        num_dims = 2
        img_shape = torch.tensor(self.image.shape[-num_dims:], dtype=torch.float32)
        new_pos = torch.clamp(new_pos, torch.zeros(num_dims), img_shape - 1)

        displacement = torch.norm(new_pos - self.init_pos)
        if displacement > self.max_displacement:
            reward = -1.0
            done = True
            return self._get_obs(), reward, done, {}

        self.current_pos = new_pos
        if self.gt_coord is not None:
            curr_dist = self._get_dist_mm()
            dist_improvement = self.prev_dist - curr_dist


            if dist_improvement > 0:
                reward = min(dist_improvement * 2.0, 1.5)  
            else:
                reward = max(dist_improvement * 3.0, -2.0)  
            if curr_dist < 1.0:
                reward += 0.5
            elif curr_dist < 2.0:
                reward += 0.2

            if curr_dist < self.best_dist:
                reward += 0.3
                self.best_dist = curr_dist
            reward -= 0.05

            self.prev_dist = curr_dist

        return self._get_obs(), reward, done, {}

    def _get_dist_mm(self):
        if self.gt_coord is None:
            return 0.0
        diff = (self.current_pos - self.gt_coord) * self.spacing[:len(self.current_pos)]
        return float(torch.norm(diff))

    def _extract_patch(self, arr, *coords, patch_size=33):
        arr = arr.detach().cpu() if isinstance(arr, torch.Tensor) else arr
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        coords = [int(round(c.item() if isinstance(c, torch.Tensor) else float(c))) for c in coords]
        half = patch_size // 2
        x, y = coords
        y0, y1 = max(0, y - half), min(arr.shape[0], y + half + 1)
        x0, x1 = max(0, x - half), min(arr.shape[1], x + half + 1)
        patch = arr[y0:y1, x0:x1]
        pad = (max(0, half - x), max(0, x + half + 1 - arr.shape[1]),
                max(0, half - y), max(0, y + half + 1 - arr.shape[0]))
        patch = F.pad(patch, pad, value=0)
        return patch

    def _compute_gradient(self, patch):
        gy = torch.zeros_like(patch)
        gx = torch.zeros_like(patch)
        gy[1:-1, :] = (patch[2:, :] - patch[:-2, :]) / 2.0
        gx[:, 1:-1] = (patch[:, 2:] - patch[:, :-2]) / 2.0
        return gy, gx
     

    def _get_obs(self):
        x, y = self.current_pos
        img_patch = self._extract_patch(self.image, x, y, patch_size=self.patch_size)

        if self.heatmap is not None:
            h_patch = self._extract_patch(self.heatmap, x, y, patch_size=self.patch_size)
            gy, gx = self._compute_gradient(h_patch)
            obs = torch.stack([img_patch, h_patch, gy, gx], dim=0)
        else:
            zeros = torch.zeros_like(img_patch)
            obs = torch.stack([img_patch, zeros, zeros, zeros], dim=0)
        return obs.clone().detach()
    
    