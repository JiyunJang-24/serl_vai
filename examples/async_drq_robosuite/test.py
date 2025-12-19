import numpy as np
from scipy.spatial.transform import Rotation as R

# Normalize both vectors
v1 = np.array([0, 0, 1])
v2 = np.array([-0.4055798, -0.4055798, +0.8191520])
v2 = v2 / np.linalg.norm(v2)

# Calculate rotation
rot, _ = R.align_vectors([v2], [v1])
quat = rot.as_quat()  # [x, y, z, w]
print("Quaternion (x, y, z, w):", quat)

# 부모와 원하는 쿼터니언
quat_body = [0.924, 0, 0, -0.383]
quat_desired = quat

# 부모 쿼터니언 역변환
r_body_inv = R.from_quat(quat_body).inv()

# 원하는 orientation
r_desired = R.from_quat(quat_desired)

# 카메라에 넣을 쿼터니언
r_camera = r_body_inv * r_desired
quat_camera = r_camera.as_quat()  # xyzw
print("Camera Quaternion (x, y, z, w):", quat_camera)