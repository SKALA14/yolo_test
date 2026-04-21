from ultralytics import YOLO
from PIL import Image
import numpy as np

detect_model = YOLO("yolo26n.pt")
pose_model   = YOLO("yolo26n-pose.pt")

source = "https://ultralytics.com/images/bus.jpg"  # 샘플 이미지

# detect
det_result = detect_model(source)[0]

print("=== BOXES ===")
print(det_result.boxes.xyxy)   # 좌표
print(det_result.boxes.cls)    # 클래스 ID
print(det_result.boxes.conf)   # confidence
print(det_result.names)        # ID → 클래스명

# pose
pose_result = pose_model(source)[0]

print("\n=== KEYPOINTS ===")
print(pose_result.keypoints.xy)    # (N, 17, 2)
print(pose_result.keypoints.conf)  # (N, 17)

# 시각화
det_result.show()   # bbox 그린 이미지 창으로 띄우기
pose_result.show()  # keypoint 그린 이미지 창으로 띄우기
det_result.save()   # bbox 그린 이미지 창으로 띄우기
pose_result.save()  # keypoint 그린 이미지 창으로 띄우기
