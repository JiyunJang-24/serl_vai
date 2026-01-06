import numpy as np
import cv2
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path


def load_camera_from_xml(xml_path: Path):
    """
    1) OpenCV FileStorage(<opencv_storage> 루트)면 그걸로 읽고
    2) 아니면 ViSP XML(px,py,u0,v0,kud,kdu)로 읽는다
    """
    # --- 1) OpenCV FileStorage 시도 (실패하면 예외로 터지므로 try/except 필수) ---
    try:
        fs = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            K = fs.getNode("camera_matrix").mat()
            if K is None or np.array(K).size == 0:
                K = fs.getNode("K").mat()
            dist = fs.getNode("distortion_coefficients").mat()
            if dist is None or np.array(dist).size == 0:
                dist = fs.getNode("D").mat()

            fs.release()

            if K is not None and np.array(K).size != 0:
                K = np.array(K, dtype=float)
                if dist is None or np.array(dist).size == 0:
                    dist = np.zeros((5, 1), dtype=float)
                else:
                    dist = np.array(dist, dtype=float).reshape(-1, 1)
                return K, dist
    except (cv2.error, SystemError):
        # ViSP xml면 여기로 떨어짐
        pass

    # --- 2) ViSP XML 파싱 ---
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    def find_text(tag_name: str):
        for e in root.iter():
            tag = e.tag
            if "}" in tag:
                tag = tag.split("}", 1)[1]
            if tag == tag_name:
                return (e.text or "").strip()
        return None

    px = float(find_text("px"))
    py = float(find_text("py"))
    u0 = float(find_text("u0"))
    v0 = float(find_text("v0"))

    # ViSP distortion 파라미터 (참고용)
    kud = float(find_text("kud") or 0.0)
    kdu = float(find_text("kdu") or 0.0)

    K = np.array([[px, 0,  u0],
                  [0,  py, v0],
                  [0,  0,  1]], dtype=float)

    # ⚠️ ViSP의 kud/kdu는 OpenCV (k1,k2,p1,p2,k3)와 동일 모델이 아님.
    # 현재 너 파일은 kud=-0, kdu=0 이라 왜곡=0 취급 가능 → dist=0 사용 OK.
    dist = np.zeros((5, 1), dtype=float)

    print(f"[INFO] Loaded ViSP XML intrinsics. kud={kud}, kdu={kdu} (OpenCV dist set to zeros)")
    return K, dist

def load_yaml_matrix(yaml_path: Path) -> np.ndarray:
    """YAML에서 4x4 HomogeneousMatrix 유연 파싱"""
    with open(yaml_path, "r") as f:
        obj = yaml.safe_load(f)

    def parse(o):
        if isinstance(o, dict):
            if "data" in o and isinstance(o["data"], (list, tuple)):
                rows = int(o.get("rows", 4))
                cols = int(o.get("cols", 4))
                arr = np.array(o["data"], dtype=float).reshape(rows, cols)
                if arr.shape == (4, 4):
                    return arr
            if len(o) == 1:
                return parse(next(iter(o.values())))
            for v in o.values():
                m = parse(v)
                if m is not None:
                    return m

        if isinstance(o, list):
            if len(o) == 4 and all(isinstance(r, list) for r in o):
                arr = np.array(o, dtype=float)
                if arr.shape == (4, 4):
                    return arr
            if len(o) == 16:
                return np.array(o, dtype=float).reshape(4, 4)
        return None

    mat = parse(obj)
    if mat is None:
        raise ValueError(f"Cannot parse 4x4 matrix from YAML: {yaml_path}")
    return mat


def draw_axes(img, T_obj_to_cam, K, dist, axis_len=0.05, thickness=2, bgr=(0, 255, 0)):
    """축을 한 색상(bgr)으로 그리기"""
    R = T_obj_to_cam[:3, :3]
    t = T_obj_to_cam[:3, 3].reshape(3, 1)

    rvec, _ = cv2.Rodrigues(R)
    tvec = t.astype(float)

    pts_obj = np.array([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len],
    ], dtype=float)

    pts_img, _ = cv2.projectPoints(pts_obj, rvec, tvec, K, dist)
    pts_img = pts_img.reshape(-1, 2).astype(int)

    o = tuple(pts_img[0])
    x = tuple(pts_img[1])
    y = tuple(pts_img[2])
    z = tuple(pts_img[3])

    cv2.line(img, o, x, bgr, thickness, cv2.LINE_AA)
    cv2.line(img, o, y, bgr, thickness, cv2.LINE_AA)
    cv2.line(img, o, z, bgr, thickness, cv2.LINE_AA)
    cv2.circle(img, o, max(2, thickness), (255, 255, 255), -1, cv2.LINE_AA)


def main():
    data_dir = Path("/home/vai/Desktop/yujin/visp/build/apps/calibration/hand-eye/data-ur")
    image_idx = 1

    # intrinsics (네가 준 ViSP XML 구조로 로드)
    K, dist = load_camera_from_xml(data_dir / "ur_camera.xml")
    print("K=\n", K)
    print("kud, kdu =", kud, kdu, " (현재 값이면 왜곡 0 취급 가능)")

    # matrices
    rMc = load_yaml_matrix(data_dir / "ur_rPc.yaml")
    eMo = load_yaml_matrix(data_dir / "ur_ePo.yaml")
    rMe = load_yaml_matrix(data_dir / f"ur_pose_rPe_{image_idx}.yaml")
    cPo_direct = load_yaml_matrix(data_dir / f"ur_pose_cPo_{image_idx}.yaml")

    cPo_calib = np.linalg.inv(rMc) @ rMe @ eMo

    # image
    img_path = data_dir / f"ur_image-{image_idx}.png"
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    # GREEN: direct, RED: calib
    draw_axes(img, cPo_direct, K, dist, bgr=(0, 255, 0))
    draw_axes(img, cPo_calib,  K, dist, bgr=(0, 0, 255))

    cv2.imshow("Reprojection compare (Green=direct, Red=calib)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
