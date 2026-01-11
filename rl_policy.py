import os
import numpy as np
import json
from stable_baselines3 import DQN
from rl_tutor_env import TutorEnv

MODEL_PATH = "models/tutor_policy"
STATE_PATH = "data/student_state.json"

ACTIONS = {
    0: "Explain very simply with analogy",
    1: "Explain step-by-step with theory",
    2: "Explain using real-world examples",
    3: "Give a hint only",
    4: "Ask a simpler prerequisite question",
    5: "Ask a harder follow-up question"
}

# ---------------- Student State ----------------
def load_student_state():
    try:
        with open(STATE_PATH, "r") as f:
            return np.array(json.load(f))
    except:
        return np.array([0.4, 0.6, 0.5, 0.5, 0.3])

def save_student_state(state):
    os.makedirs("data", exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state.tolist(), f)

def update_state(state, reward, response_time):
    mastery, confusion, rt, success, fatigue = state
    mastery += 0.1 if reward > 0 else -0.05
    confusion -= 0.1 if reward > 0 else 0.1
    success += 0.1 if reward > 0 else -0.1
    fatigue += 0.05 if response_time > 20 else -0.05

    new_state = np.clip([mastery, confusion, rt, success, fatigue], 0, 1)
    return new_state

# ---------------- Load Existing Model ----------------
env = TutorEnv()

if os.path.exists(MODEL_PATH + ".zip"):
    model = DQN.load(
        MODEL_PATH,
        env=env,
        custom_objects={
            "action_space": env.action_space,
            "observation_space": env.observation_space
        }
    )
    print("[OK] Loaded existing model")
else:
    raise FileNotFoundError(f"No model found at {MODEL_PATH}.zip. Please train the model first.")

# ---------------- Get Teaching Style ----------------
def get_teaching_style():
    state = load_student_state()
    action, _ = model.predict(state, deterministic=True)
    return ACTIONS[int(action)], state

# ---------------- Example Interaction ----------------
if __name__ == "__main__":
    action, state = get_teaching_style()
    print(f"Selected teaching action: {action}")
    print(f"Current student state: {state}")
