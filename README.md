## Early Fire Detection Using Deep Learning

This project implements a real-time fire detection system using YOLOv3, a deep learning object detection model, trained on a diverse dataset of fire and non-fire images.

**Key Features:**

- **Image Fire Detection:** Analyzes a single image to identify potential fire presence.
- **Video Fire Detection:** Processes video streams to continuously detect fire in real-time.
- **Real-Time Camera Fire Detection:** Leverages OpenCV for fire detection directly from live camera feeds using video frames.

**Benefits:**

- **Fast and Accurate:**  YOLOv3's efficiency enables prompt fire detection.
- **Real-Time Monitoring:** Ideal for rapid response in forest fire scenarios.
- **Adaptability:** Can be potentially extended to other fire detection applications.

**Requirements:**

- Python 3.x
- TensorFlow 2.x
- OpenCV
- Darknet (for YOLOv3)

**Installation:**

1. Install Python and the required libraries using a package manager like `pip`:

   ```bash
   pip install tensorflow opencv-python darknet
   ```

2. Download the pre-trained YOLOv3 weights (e.g., `yolov3.weights`). Place it in the project directory.

**Code Structure:**

```
early_fire_detection/
├── videos/fire.mp4
├── yolo.py
├── yolov3.cfg
├── eval.py  
├── obj.names  
├── fire_alarm
└── README.md  # (This file)
```

**Usage:**

**Image Fire Detection:**

```python
python yolo.py --image True --image_path <path/to/image.jpg>
```

This script analyzes the provided image and displays results with bounding boxes around detected fire regions.

**Video Fire Detection:**

```python
python yolo.py --video True --video_path <path/to/video.mp4>
```

This script processes the video and displays fire detections in real-time.

**Real-Time Camera Fire Detection:**

```python
python yolo.py --webcam True
```

This script directly accesses your webcam and provides real-time fire detection on the camera feed.

**Further Enhancements:**
- Implementing much more advanced yolo models.
- Explore integrating with alert systems (e.g., email, SMS) for real-time notifications.
- Implement fire severity classification for prioritizing responses.
- Investigate transfer learning to adapt the model to specific fire scenarios (e.g., forest fires, indoor fires).

**References:**

- YOLOv3: [(https://github.com/pjreddie/darknet)](https://github.com/pjreddie/darknet)
- Darknet: [https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)
- OpenCV: [https://opencv.org/](https://opencv.org/)
