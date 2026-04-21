from ultralytics import YOLO
from collections import deque
import cv2
import os
import shutil
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(BASE_DIR, "sample", "test.mp4")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_model(filename: str) -> YOLO:
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"[다운로드] {filename} 을(를) 다운로드합니다...")
        tmp = YOLO(filename)
        shutil.copy(tmp.ckpt_path, path)
        print(f"[저장] {path}")
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
    pose 결과 기반 이상 판단.
    - keypoint 기반 낙상 감지
    pose_result.keypoints.xy  → (N, 17, 2)
    COCO keypoint 순서:
      0=nose, 1=left_eye, 2=right_eye,
      5=left_shoulder, 6=right_shoulder,
      11=left_hip, 12=right_hip
    """
    # TODO: 트리거 로직 작성
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