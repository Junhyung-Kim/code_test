import pygame
import mujoco
import mujoco.viewer
import numpy as np
from collections import deque
import time
import torch
import torch.nn as nn

initial_obs = None

JOINT_NAME = ['Waist_yaw_joint', 
'left_hip_pitch_joint', 
'right_hip_pitch_joint', 
'left_hip_roll_joint', 
'right_hip_roll_joint', 
'left_hip_yaw_joint',
'right_hip_yaw_joint', 
'left_knee_joint', 
'right_knee_joint',
'left_ankle_pitch_joint', 
'right_ankle_pitch_joint', 
'left_ankle_roll_joint', 
'right_ankle_roll_joint'
]

ACTUATOR_JOINT_ORDER = [
    "Waist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint"
]


#kps = np.array([200, 150, 150, 150, 200, 100, 100, 150, 150,  150, 200, 100, 100], dtype=np.float32) 
#kds = np.array([5, 5, 5, 5, 8, 4, 4, 5, 5, 5, 8, 4, 4], dtype=np.float32)
kps = np.array([300, 300, 300, 300, 300, 300, 300, 300, 300,  300, 300, 300, 300], dtype=np.float32) 
kds = np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], dtype=np.float32)

joint_qpos_ids = []
joint_qvel_ids = []

model = mujoco.MjModel.from_xml_path("/home/kimm/kimmBot/kimmBot/source/kimmBot/kimmBot/assets/model/kimm_bot_scene.xml")
data  = mujoco.MjData(model)

for name in JOINT_NAME:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"Joint not found in MuJoCo model: {name}")

    joint_qpos_ids.append(model.jnt_qposadr[jid])
    joint_qvel_ids.append(model.jnt_dofadr[jid])

joint_qpos_ids = np.array(joint_qpos_ids, dtype=int)
joint_qvel_ids = np.array(joint_qvel_ids, dtype=int)

NUM_JOINTS = 13
HISTORY_LEN = 5

# nominal joint pose (Isaac Lab에서 쓰던 값과 동일해야 함)
qpos_default = np.zeros(NUM_JOINTS)

qpos_default[1] = 0.1
qpos_default[2] = -0.1
qpos_default[7] = 0.25
qpos_default[8] = 0.25
qpos_default[9] = -0.15
qpos_default[10] = -0.15

# history buffer
# obs_history = deque(maxlen=HISTORY_LEN)
obs_history = {
    "base_ang_vel":        deque(maxlen=HISTORY_LEN),
    "projected_gravity":   deque(maxlen=HISTORY_LEN),
    "velocity_commands":   deque(maxlen=HISTORY_LEN),
    "joint_pos_rel":       deque(maxlen=HISTORY_LEN),
    "joint_vel_rel":       deque(maxlen=HISTORY_LEN),
    "last_action":         deque(maxlen=HISTORY_LEN),
}

history_initialized = False

# last action (policy output)
last_action = np.zeros(NUM_JOINTS)

def reorder_action(policy_action, policy_joint_order, mujoco_joint_order):
    action_dict = {
        joint: policy_action[i]
        for i, joint in enumerate(policy_joint_order)
    }
    return np.array(
        [action_dict[j] for j in mujoco_joint_order],
        dtype=np.float32
    )

class JoystickCmd:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.js = pygame.joystick.Joystick(0)
        self.js.init()

    def get_cmd(self):
        pygame.event.pump()

        lin_x = 0.3 #-0.3  * self.js.get_axis(1)   # forward/back
        lin_y = 0.0 #0.1 * self.js.get_axis(0)   # left/right
        ang_z = 0.0 #0.1 * self.js.get_axis(3)   # yaw
        
        return lin_x, lin_y, ang_z

class RslRlActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)

class IsaacPolicy:
    def __init__(self, policy_path):#, obs_mean, obs_std):
        ckpt = torch.load(policy_path, map_location="cpu")

        # 1) Actor 네트워크 생성
        obs_dim = 240
        act_dim = 13
        self.policy = RslRlActor(obs_dim, act_dim)

        # 2) model_state_dict 꺼내기
        if "model_state_dict" not in ckpt:
            raise RuntimeError("No model_state_dict in checkpoint")

        full_state_dict = ckpt["model_state_dict"]

        # 3) actor weight만 필터링 + net. prefix 복구
        actor_state_dict = {}
        for k, v in full_state_dict.items():
            if k.startswith("actor."):
                new_key = k.replace("actor.", "")
                # 🔥 핵심 수정 포인트
                actor_state_dict[f"net.{new_key}"] = v

        # 🔥 sanity check
        print("Loaded actor keys:")
        for k in actor_state_dict.keys():
            print(" ", k)

        # 4) 로드
        self.policy.load_state_dict(actor_state_dict)
        self.policy.eval()

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = self.policy(obs)
        return action.numpy()
'''
def build_obs(data, cmd, last_action, obs_history):
    global initial_obs
    # --- base angular velocity (body frame) ---
    base_ang_vel = data.qvel[3:6]           # (3,)
    base_ang_vel = 0.2 * base_ang_vel

    # --- projected gravity ---
    base_body_id = 0  # 보통 floating base
    R = data.xmat[base_body_id].reshape(3, 3)
    g_world = np.array([0, 0, -1.0])
    projected_gravity = R.T @ g_world       # (3,)

    # --- joint states ---
    qpos = data.qpos[joint_qpos_ids]
    qvel = data.qvel[joint_qvel_ids]
    
    joint_pos_rel = qpos - qpos_default     # (13,)
    joint_vel_rel = 0.05 * qvel              # (13,)

    # --- one-step observation (48 dim) ---
    obs_t = np.concatenate([
        base_ang_vel,            # 3
        projected_gravity,       # 3
        np.array(cmd),           # 3
        joint_pos_rel,           # 13
        joint_vel_rel,           # 13
        last_action,             # 13
    ])

    if initial_obs is None:
        initial_obs = obs_t.copy()
        obs_history.append(obs_t)
        while len(obs_history) < HISTORY_LEN:
            obs_history.appendleft(obs_t)
    else:
        obs_history.append(obs_t)

    # (240,)
    obs = np.concatenate(list(obs_history))
    return obs
'''

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
def quat_rotate_inverse(q, v):
        w, x, y, z = q
        q_conj = np.array([w, -x, -y, -z])
        v_quat = np.array([0, v[0], v[1], v[2]])
        def quat_mul(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])

        result = quat_mul(quat_mul(q_conj, v_quat), q)
        return result[1:4]

def build_obs(data, cmd, last_action, obs_history):
    global history_initialized

    # ==============================
    # 1. Compute one-step terms
    # ==============================

    # --- base angular velocity (scaled) ---
    base_ang_vel = 0.2 * data.qvel[3:6]                 # (3,)

    # --- projected gravity ---
    base_body_id = 0
    #R = data.xmat[base_body_id].reshape(3, 3)
    #projected_gravity = R.T @ np.array([0, 0, -1.0])    # (3,)
    root_quat_w = data.qpos[3:7]
    gravity_world = np.array([0.0, 0.0, -1.0])
    projected_gravity = quat_rotate_inverse(root_quat_w, gravity_world)

    # --- velocity command ---
    velocity_commands = cmd                             # (3,)

    # --- joint states ---
    qpos = data.qpos[joint_qpos_ids]
    qvel = data.qvel[joint_qvel_ids]

    joint_pos_rel = qpos - qpos_default                 # (13,)
    joint_vel_rel = 0.05 * qvel                          # (13,)
    
    # --- last action ---
    #last_action = last_action                           # (13,)

    # ==============================
    # 2. Initialize history (once)
    # ==============================
    
    if not history_initialized:
        for _ in range(HISTORY_LEN):
            obs_history["base_ang_vel"].append(base_ang_vel)
            obs_history["projected_gravity"].append(projected_gravity)
            obs_history["velocity_commands"].append(velocity_commands)
            obs_history["joint_pos_rel"].append(joint_pos_rel)
            obs_history["joint_vel_rel"].append(joint_vel_rel)
            obs_history["last_action"].append(last_action.copy())
        history_initialized = True

    # ==============================
    # 3. Push current step
    # ==============================
    else:
        obs_history["base_ang_vel"].append(base_ang_vel)
        obs_history["projected_gravity"].append(projected_gravity)
        obs_history["velocity_commands"].append(velocity_commands)
        obs_history["joint_pos_rel"].append(joint_pos_rel)
        obs_history["joint_vel_rel"].append(joint_vel_rel)
        obs_history["last_action"].append(last_action.copy())

    # ==============================
    # 4. Concatenate (240 dim)
    # ==============================

    '''
    obs = np.concatenate([
        *obs_history["base_ang_vel"],
        *obs_history["projected_gravity"],
        *obs_history["velocity_commands"],
        *obs_history["joint_pos_rel"],
        *obs_history["joint_vel_rel"],
        *obs_history["last_action"],
    ])
    '''
    obs_seq = []
    for t in range(HISTORY_LEN):
        obs_seq.append(np.concatenate([
            obs_history["base_ang_vel"][t],
            obs_history["projected_gravity"][t],
            obs_history["velocity_commands"][t],
            obs_history["joint_pos_rel"][t],
            obs_history["joint_vel_rel"][t],
            obs_history["last_action"][t],
        ]))

    obs = np.concatenate(obs_seq)   # (240,)
    #print("obs")
    #print(obs)
    
    return obs


js = JoystickCmd()


policy = IsaacPolicy(
        policy_path="/home/kimm/kimmBot/kimmBot/logs/rsl_rl/kimmbot_direct/2026-01-05_10-53-24/model_1000.pt",          # 🔹 Isaac Lab export
        #obs_mean=np.load("obs_mean.npy"), # 🔹 학습 시 저장한 값
        #obs_std=np.load("obs_std.npy"),
    )

kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "front")
assert kf_id >= 0

mujoco.mj_resetDataKeyframe(model, data, kf_id)
mujoco.mj_forward(model, data)

a = 0
step_count = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        lin_x, lin_y, ang_z = js.get_cmd()
        cmd = [lin_x, lin_y, ang_z]
        cmd = [0.3, 0.0, 0]

        # 2) policy inference
        if step_count % 5 == 0:
            obs = build_obs(data, cmd, last_action, obs_history)
            action = policy.act(obs)   # shape: (13,)
    
            # 3) action scaling (⚠️ 중요)
            # Isaac Lab policy output ∈ [-1, 1]
            # → MuJoCo actuator에 맞게 scale
            action_scaled = 0.25 * action   # 예시 (로봇마다 다를 수 있음)
            last_action[:] = action
            qpos_ref = qpos_default + action_scaled

        mujoco_action = reorder_action(
            qpos_ref,
            JOINT_NAME,
            ACTUATOR_JOINT_ORDER
        )
        
        # 4) apply control
        qvel_ref = np.zeros(len(kps))
        tau = (kps * (mujoco_action - data.qpos[7:]) + kds * (qvel_ref - data.qvel[6:]))
        data.ctrl[:] = 1 * tau
        #print(data.ctrl[:])

        if(step_count % 100 == 0):
            print("qpos")
            print(mujoco_action)
            print(data.qpos[7:])
            print(cmd)

        # 5) last_action 업데이트
        mujoco.mj_step(model, data)
        viewer.sync()
    
        step_count = step_count + 1

        #if step_count == 2:
        #    time.sleep(100000)
        
        #time.sleep(0.0005)
