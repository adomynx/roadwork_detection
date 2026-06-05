# object_detection — integration notes

This package adds a general-purpose YOLOv8x (COCO, 80 classes) detector that runs
alongside the fine-tuned roadwork detector. It is **fully integrated**:

- `object_detector_node` publishes `/objects/results`
- `fusion_node` gains a second detection stream and publishes `/objects/fused`
  (LiDAR-derived 3D distance per object) — **without ever touching `/metrics/distance`**
- `detector_node` shows a new HUD line with live object counts

```
rosbag /zed/.../compressed ──► object_detector_node
                                     │
                                     ├──► /objects/results ──► fusion_node  ──► /objects/fused
                                     │
                                     └──► /objects/results ──► detector_node (HUD counts)
```

---

## 1. Generate the COCO TensorRT engine (one-time)

```bash
source ~/roadwork_project/venv/bin/activate
mkdir -p ~/roadwork_project/workspace/src/object_detection/models
cd ~/roadwork_project/workspace/src/object_detection/models

# downloads stock COCO weights (~136 MB) then exports a TRT FP16 engine
yolo export model=yolov8x.pt format=engine half=True device=0 imgsz=640
mv yolov8x.engine yolov8x_coco.engine
```

> The `.engine` is hardware/TensorRT-version specific — regenerate it on the
> same machine that will run it, exactly like the roadwork engine. Add
> `*.engine` to `.gitignore` (you almost certainly already have this).

---

## 2. Build

```bash
cd ~/roadwork_project/workspace
colcon build --packages-select object_detection --symlink-install
source install/setup.bash
```

## 3. Run standalone (smoke test before wiring fusion/overlay)

```bash
ros2 launch object_detection object_detection_launch.py
# in another terminal:
ros2 topic hz /objects/results
ros2 run rqt_image_view rqt_image_view /objects/annotated_image
```

## 4. Add to run_all.sh

Add a 7th tmux window next to the existing launches, e.g.:

```bash
tmux new-window -t roadwork -n objects \
  "source ~/roadwork_project/workspace/install/setup.bash; \
   ros2 launch object_detection object_detection_launch.py; exec bash"
```

(`stop_all.sh` needs no change — it kills the whole `roadwork` tmux session.)

---

## 5. fusion_node.py — additions

You already have the box↔LiDAR projection/matching logic for `/detection/results`.
The integration reuses it for the objects stream and writes to a separate topic.

### 5a. Imports — already present (json), nothing new needed.

### 5b. In `__init__`, ADD storage, a subscription, and a publisher:

```python
# --- general object stream (added) ---
from std_msgs.msg import String          # if not already imported
self.latest_objects = []                 # list of detection dicts
self.objects_stamp = 0.0                 # ros time (sec) of last objects msg

self.objects_sub = self.create_subscription(
    String, '/objects/results', self.objects_results_callback, 10)

self.objects_fused_pub = self.create_publisher(
    String, '/objects/fused', 10)
```

### 5c. ADD the callback. It parses `/objects/results`, matches each box to the
most recent LiDAR obstacles using the SAME helper you already use for roadwork
detections, and republishes with distance attached.

Replace `self._match_box_to_lidar(...)` below with whatever your existing
method is actually called (the one that returns a distance for a given bbox by
projecting LiDAR centroids into the camera frame and checking the 50 px window).

```python
def objects_results_callback(self, msg):
    try:
        data = json.loads(msg.data)
    except json.JSONDecodeError:
        return

    self.objects_stamp = self.get_clock().now().nanoseconds * 1e-9
    self.latest_objects = data.get('detections', [])

    # Need a recent LiDAR cloud/obstacle set to fuse against.
    # Use the same `self.latest_lidar_*` buffer the roadwork path uses.
    fused = []
    for det in self.latest_objects:
        bbox = det['bbox']  # [x1, y1, x2, y2]

        # >>> reuse your existing matcher (returns distance in metres or None)
        distance = self._match_box_to_lidar(bbox)
        # <<<

        fused.append({
            'class_id':   det['class_id'],
            'class_name': det['class_name'],
            'confidence': det['confidence'],
            'bbox':       bbox,
            'distance_m': (round(distance, 2)
                           if distance is not None else None),
            'lidar_matched': distance is not None,
        })

    out = String()
    out.data = json.dumps({
        'stamp_sec':  data.get('stamp_sec', 0),
        'count':      len(fused),
        'objects':    fused,
    })
    self.objects_fused_pub.publish(out)
```

> Key point: this path publishes ONLY to `/objects/fused`. It never writes
> `/metrics/distance`, so the ODD-exit distance metric stays roadwork-only and
> a passing car can't move the risk value. If later you want the roadwork
> matcher refactored into a shared `_match_box_to_lidar(bbox) -> float|None`
> helper so both streams call identical code, that's a clean ~10-line extract.

---

## 6. detector_node.py — additions (HUD object-count line)

`detector_node` already owns the overlay. Add a subscription that keeps a rolling
class-count dict, then draw one extra line. Uses the same staleness timeout
pattern as your other metric subscriptions.

### 6a. In `__init__`, ADD:

```python
from std_msgs.msg import String          # if not already imported
self.object_counts = {}                  # {class_name: count} for last frame
self.objects_last_time = 0.0
self.objects_timeout = 2.0               # match your existing metric_timeout

self.objects_sub = self.create_subscription(
    String, '/objects/results', self.objects_results_callback, 10)
```

### 6b. ADD the callback:

```python
def objects_results_callback(self, msg):
    try:
        data = json.loads(msg.data)
    except json.JSONDecodeError:
        return
    counts = {}
    for det in data.get('detections', []):
        name = det['class_name']
        counts[name] = counts.get(name, 0) + 1
    self.object_counts = counts
    self.objects_last_time = time.time()
```

### 6c. In your HUD-drawing method, ADD a line. Insert this where you render the
other rows (after Weather, before/after Det — your call). `y_cursor` is whatever
running y-coordinate you use for stacking HUD rows:

```python
# --- objects line (added) ---
if time.time() - self.objects_last_time < self.objects_timeout \
        and self.object_counts:
    # top 3 classes by count, pluralised crudely
    top = sorted(self.object_counts.items(),
                 key=lambda kv: kv[1], reverse=True)[:3]
    parts = [f"{n} {name}{'s' if n > 1 else ''}" for name, n in top]
    obj_text = "Objects: " + "  ".join(parts)
    cv2.putText(overlay, obj_text, (x_text, y_cursor),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    y_cursor += line_height
```

### 6d. Grow the HUD background rectangle.

Your overlay panel is currently ~148 px tall. Adding one row needs ~20 px more —
bump the filled-rectangle height (e.g. `148 -> 168`) so the new line isn't clipped.
Find the `cv2.rectangle(...)` that draws the HUD background and increase its
bottom-y by ~20.

---

## 7. Verify end-to-end

```bash
# replay the bag (your usual command / run_all.sh), then:
ros2 topic echo /objects/results --once     # raw detections
ros2 topic echo /objects/fused   --once      # distances attached
# watch the HUD: the "Objects:" line should appear in rqt_image_view
```

---

## Notes / tunables

- `confidence_threshold` (default 0.35) and `iou_threshold` (0.45) live in
  `config/object_detection_params.yaml`.
- COCO has 80 classes; the node publishes all of them. If you only care about
  driving-relevant classes (person, bicycle, car, motorcycle, bus, truck,
  traffic light, stop sign…), filter in `objects_results_callback` rather than
  at the model, so `/objects/fused` keeps the full set for logging.
- Two YOLOv8x TRT engines on the A5000: ~400 MB VRAM combined, GPU time-slices
  between the two processes. Expect ~12–15 ms/frame each vs ~9 ms alone — still
  comfortably within 0.5× bag playback.
