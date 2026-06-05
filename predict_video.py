import cv2
import json
import time
import sys
import os
from ultralytics import YOLO

# ROADWork classes
CLASS_NAMES = {
    0: 'Police Officer', 1: 'Police Vehicle', 2: 'Cone', 3: 'Fence',
    4: 'Drum', 5: 'Barricade', 6: 'Barrier', 7: 'Work Vehicle',
    8: 'Vertical Panel', 9: 'Tubular Marker', 10: 'Arrow Board',
    11: 'Bike Lane', 12: 'Work Equipment', 13: 'Worker',
    14: 'Other Roadwork Objects',
    15: 'TTC Message Board', 16: 'TTC Sign'
}

# Dynamic classes (for risk calculation)
DYNAMIC_CLASSES = {0, 1, 7, 13}  # Police Officer, Police Vehicle, Work Vehicle, Worker

# Paper parameters (Table II)
CAMERA_HEIGHT = 1.5
DEFAULT_FY = 1000.0
DEFAULT_CY = 360.0  # Assuming 720p video, cy = height/2
LAMBDA = 0.50
N_MAX = 12
D_NEAR = 5.0
D_FAR = 30.0
W_C = 0.50
W_D = 0.50
ALPHA_D = 0.15
ALPHA_C = 0.15
ALPHA_R = 0.20
CONF_THRESHOLD = 0.25

# EMA states
d_smoothed = None
c_smoothed = None
r_smoothed = None


def calculate_metrics(detections, img_h, img_w):
    global d_smoothed, c_smoothed, r_smoothed

    if len(detections) == 0:
        return d_smoothed, c_smoothed, r_smoothed

    N = len(detections)

    # ---- DISTANCE ----
    distances = []
    cy = img_h / 2.0
    fy = DEFAULT_FY

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        v_j = y2  # bottom edge
        if v_j > cy:
            d_j = (CAMERA_HEIGHT * fy) / (v_j - cy)
            distances.append(d_j)

    d_min = min(distances) if distances else None

    if d_min is not None:
        if d_smoothed is None:
            d_smoothed = d_min
        else:
            d_smoothed = ALPHA_D * d_min + (1 - ALPHA_D) * d_smoothed

    # ---- CONFIDENCE ----
    total_weighted = 0.0
    total_area = 0.0

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        A_j = (x2 - x1) * (y2 - y1)
        if A_j > 0:
            total_weighted += det['confidence'] * A_j
            total_area += A_j

    C_pres = total_weighted / total_area if total_area > 0 else 0
    C_count = min(1.0, N / N_MAX)
    C_frame = LAMBDA * C_pres + (1 - LAMBDA) * C_count

    if c_smoothed is None:
        c_smoothed = C_frame
    else:
        c_smoothed = ALPHA_C * C_frame + (1 - ALPHA_C) * c_smoothed

    # ---- RISK ----
    # Distance risk
    if d_smoothed is not None:
        if d_smoothed <= D_NEAR:
            R_d = 1.0
        elif d_smoothed >= D_FAR:
            R_d = 0.0
        else:
            R_d = (D_FAR - d_smoothed) / (D_FAR - D_NEAR)
    else:
        R_d = 0.0

    # Dynamic object ratio
    N_dyn = sum(1 for d in detections if d['class_id'] in DYNAMIC_CLASSES)
    R_dyn = N_dyn / N if N > 0 else 0.0

    # Contextual risk
    R_c = (W_C * c_smoothed + W_D * R_dyn) / (W_C + W_D)

    # Final risk
    R_frame = max(R_d, R_c)

    if r_smoothed is None:
        r_smoothed = R_frame
    else:
        r_smoothed = ALPHA_R * R_frame + (1 - ALPHA_R) * r_smoothed

    return d_smoothed, c_smoothed, r_smoothed


def draw_metrics(frame, d_val, c_val, r_val, num_det, proc_time):
    h, w = frame.shape[:2]

    # Dark overlay box for metrics
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (420, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "ODD Exit Metrics", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Distance
    d_text = f"Distance: {d_val:.1f}m" if d_val else "Distance: N/A"
    d_color = (0, 0, 255) if d_val and d_val < D_NEAR else (0, 255, 0) if d_val and d_val > D_FAR else (0, 165, 255)
    cv2.putText(frame, d_text, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, d_color, 2)

    # Confidence
    c_text = f"Confidence: {c_val*100:.1f}%" if c_val else "Confidence: N/A"
    c_color = (0, 0, 255) if c_val and c_val > 0.7 else (0, 255, 0) if c_val and c_val < 0.3 else (0, 165, 255)
    cv2.putText(frame, c_text, (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, c_color, 2)

    # Risk
    r_text = f"Risk: {r_val*100:.1f}%" if r_val else "Risk: N/A"
    r_color = (0, 0, 255) if r_val and r_val > 0.7 else (0, 255, 0) if r_val and r_val < 0.3 else (0, 165, 255)
    cv2.putText(frame, r_text, (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, r_color, 2)

    # Info line
    cv2.putText(frame, f"Detections: {num_det} | {proc_time:.0f}ms", (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return frame


def process_video(video_path, model_path, output_dir):
    global d_smoothed, c_smoothed, r_smoothed
    d_smoothed = None
    c_smoothed = None
    r_smoothed = None

    model = YOLO(model_path, task='detect')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_name}_detected.mp4")

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"Processing: {video_path}")
    print(f"Resolution: {w}x{h} | FPS: {fps} | Frames: {total}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        start = time.time()

        results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                'class_id': cls_id,
                'class': CLASS_NAMES.get(cls_id, f'class_{cls_id}'),
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })

            # Draw bounding box
            color = (0, 255, 0) if cls_id not in DYNAMIC_CLASSES else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{CLASS_NAMES.get(cls_id, cls_id)} {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        proc_time = (time.time() - start) * 1000

        d_val, c_val, r_val = calculate_metrics(detections, h, w)
        frame = draw_metrics(frame, d_val, c_val, r_val, len(detections), proc_time)

        writer.write(frame)

        if frame_count % 30 == 0:
            print(f"Frame {frame_count}/{total} | Det: {len(detections)} | "
                  f"D: {d_val:.1f}m | C: {c_val*100:.1f}% | R: {r_val*100:.1f}%" if d_val else
                  f"Frame {frame_count}/{total} | Det: {len(detections)} | No metrics")

    cap.release()
    writer.release()
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 predict_video.py <video_path> [model_path]")
        sys.exit(1)

    video = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else 'runs/detect/roadwork_yolov8x/weights/best.pt'
    output = os.path.expanduser('~/roadwork_project/video_results')

    process_video(video, model, output)
