from ur_env.envs.ur5_env import DefaultEnvConfig
import numpy as np


class UR5CameraConfig(DefaultEnvConfig):
    RESET_Q = np.array([[1.3502, -1.2897, 1.9304, -2.2098, -1.5661, 1.4027]])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.00,)
    RANDOM_ROT_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.array([0.2, -0.4, 0.22, 3.2, 0.18, 3.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.2, -0.7, - 0.006, 2.8, -0.18, -3.2])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

    ROBOT_IP: str = "192.168.0.9"
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)

    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.1, 1., 1., 1.])

    REALSENSE_CAMERAS = {
        "wrist": "109622073973",
    }


class UR5CameraConfigBox5(DefaultEnvConfig):
    RESET_Q = np.array([
        [1.3776, -1.0603, 1.6296, -2.1462, -1.5704, -0.2019],
        [0.9104, -0.9716, 1.3539, -1.9824, -1.545, -0.662],
        [0.4782, -1.4072, 2.1258, -2.3129, -1.5816, -1.1417],
        [1.2083, -1.656, 2.272, -2.202, -1.5828, -0.4231],
        [-0.0388, -1.754, 2.2969, -2.1271, -1.5423, -1.7011]
    ])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_ROT_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.array([0.05, 0.1, 0.22, 3.2, 0.18, 3.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.49, -0.75, -0.006, 2.8, -0.18, -3.2])
    ACTION_SCALE = np.array([0.01, 0.01, 0.01], dtype=np.float32)

    ROBOT_IP: str = "192.168.0.9"
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.1, 1., 1., 1.])

    # REALSENSE_CAMERAS = {
    #     "wrist": "218622277164",
    #     # "shoulder": "218622279756"
    # }
    REALSENSE_CAMERAS = {
        "wrist": "109622073973",
    }

class UR5CameraConfigFinal(DefaultEnvConfig):  # config for 10 boxes
    RESET_Q = np.array([
        [-0.10577947298158819, -1.647076269189352, 1.6802166144000452, -1.592215200463766, -1.6467116514789026, -0.22041160265077764],
    ])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_ROT_RANGE = (0.04,)
    ABS_POSE_LIMIT_HIGH = np.array([-0.22, 0.31, 0.20, 4, 4.0, 4.0])
    ABS_POSE_LIMIT_LOW = np.array([-0.735, -0.2354, -0.0217, -4.0, -4.0, -4.0])
    # ABS_POSE_LIMIT_HIGH = np.array([-0.325, 0.145, 0.025, 4, 0.1, 3.14])
    # ABS_POSE_LIMIT_HIGH = np.array([-0.34, -0.045, 0.025, 3.1415, 0.1, 0.7])
    # ABS_POSE_LIMIT_LOW = np.array([-0.495, -0.1572, -0.08, 2.86, -0.1, -0.5])
    # ABS_POSE_LIMIT_LOW = np.array([-0.495, -0.1572, -0.085, 2.86, -0.1, -3.14])
    ABS_POSE_RANGE_LIMITS = np.array([0.36, 0.83])
    
    INNER_ABS_POSE_LIMIT_HIGH = np.array([-0.325, 0.05, -0.0375])
    INNER_ABS_POSE_LIMIT_LOW = np.array([-0.505, -0.05, -0.1])
    
    INNER_ABS_POSE_HIGH = np.array([-0.315, 0.05, 0.175])
    INNER_ABS_POSE_LOW = np.array([-0.51, -0.06, -0.1])
    ACTION_SCALE = np.array([0.025, 0.075, 1.], dtype=np.float32)

    ROBOT_IP: str = "192.168.0.253"
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.05  # faster but more vulnerable to crash...
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])

    # REALSENSE_CAMERAS = {
    #     "wrist": "218622277164",
    #     "wrist_2": "218622279756"
    # }
    REALSENSE_CAMERAS = {
        "front": "109622073973",
        # "wrist": "218622272114",
    }

class UR5CameraConfigDemo(UR5CameraConfigFinal):
    RESET_Q = np.array([[0., -np.pi / 2., np.pi / 2., -np.pi / 2., -np.pi / 2., 0.]])
    ABS_POSE_LIMIT_HIGH = np.array([1., 1., 1., 0.1, 0.1, 0.3])
    ABS_POSE_LIMIT_LOW = np.array([-1., -1., -0.004, -0.1, -0.1, -0.3])

    ROBOT_IP: str = "192.168.1.66"


class UR5CameraConfigFinalTests(UR5CameraConfigFinal):
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_ROT_RANGE = (0.05,)

    RESET_Q = np.array([
        # [0.0421, -1.3161, 1.9649, -2.2358, -1.3221, -1.5237 + 0 * np.pi / 2.],  # schräge position
        # [0.1882, -1.2777, 1.9699, -2.2983, -1.5567, -1.384 + 2 * np.pi / 2],  # gerade pos
        # [1.4843, -1.1314, 1.6531, -2.0676, -1.6014, 1.6402]
        [0.4691, -1.3288, 1.9659, -2.2276, -1.5962, 0.3519]
    ])


class UR5CameraConfigFinalEvaluation(UR5CameraConfigFinal):
    # config for the evaluation on 5 boxes the policy has never seen
    RANDOM_RESET = True
    RANDOM_XY_RANGE = (0.01,)
    RANDOM_ROT_RANGE = (0.05,)
    ABS_POSE_LIMIT_HIGH = np.array([0.6, 0.1, 0.25, 0.1, 0.1, 0.3])
    ABS_POSE_LIMIT_LOW = np.array([-0.7, -0.85, -0.006, -0.1, -0.1, -0.3])

    RESET_Q = np.array([
        [0.4102, -1.304, 1.9315, -2.1707, -1.5583, 2.0127],
        [0.9212, - 0.8757, 1.3325, - 2.0209, - 1.5508, 2.5185],
        [1.2869, - 0.9778, 1.476, - 2.0783, - 1.5458, 2.9585],
        [1.717, -1.1379, 1.7179, -2.4872, -1.4362, 2.5804],
        [2.2614, - 1.4378, 2.145, - 2.5039, - 1.7649, 2.2541],
    ])


class UR5VoxelConfig(UR5CameraConfigFinal):
    RESET_Q = np.array([
        [1.4665, -0.8476, 1.2612, -1.9817, -1.5623, -0.0916],
        [1.1021, -0.7136, 1.0127, -1.8756, -1.5689, -0.4738],
        [0.966, -1.3074, 2.0641, -2.3285, -1.5542, -2.1304],
        [0.696, -0.9965, 1.5125, -2.0854, -1.5655, -0.8484],
        [0.2125, -1.2685, 1.8644, -2.168, -1.5703, -1.3682],
    ])

    ACTION_SCALE = np.array([0.01, 0.05, 1.], dtype=np.float32)
    RANDOM_RESET = True
    CONTROLLER_HZ = 50      # for the c++ wrapper controller
    RANDOM_XY_RANGE = (0.02,)
    RANDOM_Z_RANGE = (0.02)
    RANDOM_ROT_RANGE = (0.02,)

    ABS_POSE_LIMIT_HIGH = np.array([0.6, 0.1, 0.25, 0.05, 0.05, 0.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.7, -0.85, -0.006, -0.05, -0.05, -0.2])
    ABS_POSE_RANGE_LIMITS = np.array([0.4, 0.78])

    # REALSENSE_CAMERAS = {
    #     "wrist": "218622270808",
    # }
    REALSENSE_CAMERAS = {
        "wrist": "109622073973",
    }