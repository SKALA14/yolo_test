from ultralytics import YOLO
from collections import deque
import cv2
import os
import math
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(BASE_DIR, "sample", "sample.mp4")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_model(filename: str) -> YOLO:
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"[다운로드] {filename} 을(를) 다운로드합니다...")
        orig_dir = os.getcwd()
        os.chdir(MODEL_DIR)
        try:
            YOLO(filename)
        finally:
            os.chdir(orig_dir)
    return YOLO(path)


detect_model = load_model("yolo26n.pt")
pose_model   = load_model("yolo26n-pose.pt")

# flickering 대응용 sliding window
anomaly_window = deque(maxlen=10)


def is_anomaly_people(det_result) -> bool:
    """
    detect 결과 기반 이상 판단.
    - 위험구역 침입
    - PPE 미착용 등
    """
    # TODO: 트리거 로직 작성
    return False


def is_anomaly_pose(pose_result) -> bool:
    """
    pose 결과 기반 낙상 감지 (휴리스틱 3종 점수 합산).

    점수 기준 (2점 이상 → 낙상):
      +2  토르소 각도: 어깨-엉덩이 벡터가 수직에서 55° 이상 기울어진 경우
      +2  코 위치: nose y좌표 > 엉덩이 중점 y좌표 (머리가 엉덩이보다 아래)
      +1  바운딩박스 비율: width / height > 1.3 (눕거나 쓰러진 실루엣)

    COCO keypoint 인덱스:
      0=nose, 5=left_shoulder, 6=right_shoulder,
      11=left_hip, 12=right_hip
    """
    if pose_result.keypoints is None:
        return False

    kpts  = pose_result.keypoints.xy    # (N, 17, 2)
    confs = pose_result.keypoints.conf  # (N, 17)
    boxes = pose_result.boxes

    NOSE = 0
    L_SH, R_SH   = 5, 6
    L_HIP, R_HIP = 11, 12
    CONF_THRESH   = 0.3

    for i in range(len(kpts)):
        pts = kpts[i]   # (17, 2)
        cf  = confs[i]  # (17,)

        if not (cf[L_SH] > CONF_THRESH and cf[R_SH] > CONF_THRESH
                and cf[L_HIP] > CONF_THRESH and cf[R_HIP] > CONF_THRESH):
            continue

        sh_mid  = (pts[L_SH]  + pts[R_SH])  / 2  # (x, y)
        hip_mid = (pts[L_HIP] + pts[R_HIP]) / 2

        dx = float(hip_mid[0] - sh_mid[0])
        dy = float(hip_mid[1] - sh_mid[1])
        torso_len = math.hypot(dx, dy)
        if torso_len < 1e-6:
            continue

        # 수직(dy 방향)에서의 기울기 각도: 0°=직립, 90°=수평
        angle = math.degrees(math.acos(min(abs(dy) / torso_len, 1.0)))

        score = 0

        if angle > 55:
            score += 2

        if cf[NOSE] > CONF_THRESH and float(pts[NOSE][1]) > float(hip_mid[1]):
            score += 2

        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = boxes.xyxy[i]
            w = float(x2 - x1)
            h = float(y2 - y1)
            if h > 0 and w / h > 1.3:
                score += 1

        if score >= 2:
            return True

    return False


def save_frame(frame, label: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(OUTPUT_DIR, f"{label}_{timestamp}.jpg")
    cv2.imwrite(path, frame)
    print(f"[저장] {path}")


if __name__ == "__main__":
    for det_result in detect_model.track(source=SOURCE, stream=True, conf=0.3):
        frame = det_result.orig_img
        boxes = det_result.boxes

        # 1. detect 기반 트리거
        detected = is_anomaly_people(det_result)
        anomaly_window.append(detected)

        # sliding window 다수결 (10프레임 중 7번 이상)
        if sum(anomaly_window) >= 7:
            save_frame(det_result.plot(), "people_anomaly")
            anomaly_window.clear()

        # 2. person 있을 때만 pose 호출
        if boxes is not None:
            classes = [det_result.names[int(c)] for c in boxes.cls]
            if "person" in classes:
                pose_result = pose_model(frame, verbose=False)[0]

                if is_anomaly_pose(pose_result):
                    save_frame(pose_result.plot(), "pose_anomaly")

        # 실시간 확인
        cv2.imshow("people", det_result.plot())
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()