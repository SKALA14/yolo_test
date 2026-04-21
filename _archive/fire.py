from ultralytics import YOLO
import cv2
import os
import json
from datetime import datetime

# ── 설정 ─────────────────────────────────────────────
SOURCE = "https://www.youtube.com/watch?v=vL2Y6k8g1zY"  # 영상 경로 또는 URL
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 모델 로드 ─────────────────────────────────────────
fire_model = YOLO("best.pt")

# ── 트리거 정의 ───────────────────────────────────────

def is_fire_detected(result) -> bool:
    return result.boxes is not None and len(result.boxes) > 0


# ── 이미지 저장 ───────────────────────────────────────

def save_frame(frame, label: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(OUTPUT_DIR, f"{label}_{timestamp}.jpg")
    cv2.imwrite(path, frame)
    print(f"[저장] {path}")


# ── 메인 루프 ─────────────────────────────────────────

def main():
    frame_idx = 0
    json_path = os.path.join(OUTPUT_DIR, "fire_detections.jsonl")

    with open(json_path, "w") as f:
        for result in fire_model.track(source=SOURCE, stream=True, conf=0.3):
            boxes = result.boxes
            frame_idx += 1

            if boxes is not None:
                for box in boxes:
                    record = {
                        "frame": frame_idx,
                        "class": result.names[int(box.cls)],
                        "confidence": round(float(box.conf), 4),
                        "bbox": [round(float(v), 2) for v in box.xyxy[0]],
                        "track_id": int(box.id) if box.id is not None else None,
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()

            if is_fire_detected(result):
                save_frame(result.plot(), "fire")

            cv2.imshow("Fire Detection", result.plot())
            if cv2.waitKey(1) == ord("q"):
                break

    cv2.destroyAllWindows()
    print(f"[저장 완료] {json_path}")


if __name__ == "__main__":
    main()
