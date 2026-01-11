# RL Tutor

A Reinforcement Learning (RL) based tutoring environment that simulates a learning scenario and trains an agent to adaptively respond to different student states. Built using **Gymnasium** and **Stable-Baselines3 (DQN)**.

---

## Features

- Custom Gymnasium environment simulating a student learning process.
- DQN agent trained to maximize student engagement or performance (reward-based feedback).
- Fully compatible with **Gymnasium** and **Stable-Baselines3**.
- Easy to extend for new tutoring strategies, reward systems, or student models.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/rl-tutor.git
cd rl-tutor

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
