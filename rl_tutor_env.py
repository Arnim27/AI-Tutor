import gymnasium as gym
import numpy as np

class TutorEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(6)
        self.state = np.random.rand(5)
        
    def step(self, action):
        reward = np.random.choice([1, -1])
        self.state = np.clip(self.state + np.random.randn(5)*0.05, 0, 1)
        terminated = False
        truncated = False
        info = {}
        return self.state, reward, terminated, truncated, info
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.rand(5)
        info = {}
        return self.state, info
