import numpy as np
from scipy.spatial.transform import Rotation as R

"""
UR5 represents the orientation in axis angle representation
"""


def rotvec_2_quat(rotvec):
    return R.from_rotvec(rotvec).as_quat()

def rotvec_2_mrp(rotvec):
    return R.from_rotvec(rotvec).as_mrp()


def quat_2_rotvec(quat):
    return R.from_quat(quat).as_rotvec()


def quat_2_euler(quat):
    return R.from_quat(quat).as_euler('xyz')


def quat_2_mrp(quat):
    return R.from_quat(quat).as_mrp()


def euler_2_quat(euler):
    return R.from_euler(euler).as_quat()


def pose_2_quat(rotvec_pose) -> np.ndarray:
    return np.concatenate((rotvec_pose[:3], rotvec_2_quat(rotvec_pose[3:])))


def pose_2_rotvec(quat_pose) -> np.ndarray:
    return np.concatenate((quat_pose[:3], quat_2_rotvec(quat_pose[3:])))

def pose_2_euler(rotvec_pose) -> np.ndarray:
    # import pdb; pdb.set_trace()
    return np.concatenate((rotvec_pose[:3], R.from_rotvec(rotvec_pose[3:]).as_euler('xyz')))

def euler_pose_2_rv(euler_pose) -> np.ndarray:
    # return R.from_euler(euler_pose[:3], euler_pose[3:], False).as_rotvec()
    return np.concatenate((euler_pose[:3], R.from_euler('xyz', euler_pose[3:], False).as_rotvec()))

def rotate_rotvec(rotvec, rot_matrix):
    return (R.from_rotvec(rotvec) * R.from_matrix(rot_matrix)).as_rotvec()

def pose_to_tip_pose_rv(rotvec_pose, gripper_length=0.23):
    """
    Converts UR5e EEF pose (pos + rotvec) to tip pose (pos + rotvec)
    Args:
        eef_pose: [x, y, z, rx, ry, rz]
        gripper_length: scalar (in meters)
    Returns:
        tip_pose: [tip_x, tip_y, tip_z, tip_rx, tip_ry, tip_rz]
    """
    position = np.array(rotvec_pose[:3])
    rotvec = np.array(rotvec_pose[3:])
    rotation = R.from_rotvec(rotvec)

    # Gripper offset along local z-axis
    offset = np.array([0, 0, gripper_length])
    rotated_offset = rotation.apply(offset)

    tip_position = position + rotated_offset
    tip_rotvec = rotvec  # Orientation is same as EEF

    tip_pose = np.concatenate([tip_position, tip_rotvec])
    return tip_pose