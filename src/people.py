from ultralytics import YOLO
import cv2
import os
import json
from datetime import datetime

# ── 설정 ─────────────────────────────────────────────
SOURCE = "../sample/sample.mp4"  # 영상 경로 또는 URL
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 모델 로드 ─────────────────────────────────────────
detect_model = YOLO("yolo26n.pt")
pose_model   = YOLO("yolo26n-pose.pt")


# ── 트리거 정의 (직접 수정) ───────────────────────────

def is_anomaly_detect(result) -> bool:
    """
    detect 모델 결과 기반 이상 판단.
    예: 특정 클래스 등장, person 수 급증 등
    """
    # TODO: 트리거 로직 작성


    return False


def is_anomaly_pose(pose_result) -> bool:
    """
    pose 모델 결과 기반 이상 판단.
    예: keypoint 기반 낙상 감지
    pose_result.keypoints.xy  → (N, 17, 2) 좌표
    """
    # TODO: 트리거 로직 작성
    

    return False


# ── 이미지 저장 ───────────────────────────────────────

def save_frame(frame, label: str):
    """bbox 그려진 프레임을 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(OUTPUT_DIR, f"{label}_{timestamp}.jpg")
    cv2.imwrite(path, frame)
    print(f"[저장] {path}")


# ── 메인 루프 ─────────────────────────────────────────

def main():
    frame_idx = 0
    json_path = os.path.join(OUTPUT_DIR, "detections.jsonl")

    with open(json_path, "w") as f:
        for det_result in detect_model.track(source=SOURCE, stream=True, conf=0.3):
            frame = det_result.orig_img
            boxes = det_result.boxes
            frame_idx += 1

            if boxes is not None:
                for box in boxes:
                    record = {
                        "frame": frame_idx,
                        "class": det_result.names[int(box.cls)],
                        "confidence": round(float(box.conf), 4),
                        "bbox": [round(float(v), 2) for v in box.xyxy[0]],
                        "track_id": int(box.id) if box.id is not None else None,
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()

            # 1. detect 기반 트리거
            if is_anomaly_detect(det_result):
                annotated = det_result.plot()
                save_frame(annotated, "detect_anomaly")

            # 2. person 있을 때만 pose 모델 호출
            if boxes is not None:
                classes = [det_result.names[int(c)] for c in boxes.cls]
                if "person" in classes:
                    pose_result = pose_model(frame, verbose=False)[0]

                    if is_anomaly_pose(pose_result):
                        annotated = pose_result.plot()
                        save_frame(annotated, "pose_anomaly")

            # 실시간 확인용 (필요 없으면 제거)
            cv2.imshow("YOLO", det_result.plot())
            if cv2.waitKey(1) == ord("q"):
                break

    cv2.destroyAllWindows()
    print(f"[저장 완료] {json_path}")


if __name__ == "__main__":
    main()
