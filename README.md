# Luggage Detection with YOLO

This project implements luggage detection using the YOLO (You Only Look Once) object detection model.

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Download YOLO models:
   ```
   python download_yolo_model.py
   - It will download yolo 11 and yolo v10 all versions.
   ```

## Running the Detector

Run the luggage detector on a video file:
```
python luggage_detector.py --source file.mp4
```

Run with a webcam:
```
python luggage_detector.py --source 0
```
Run with a confidence:
```
python luggage_detector.py --source your_path  --confidence (e.g 0.7)
```

## Controls
- Press 'q' or 'Esc' to quit
- Press 'x' to close the window
- Click the close button (X) on the window to exit

## Getting the Latest YOLO Model

The script will automatically try to use YOLOv11 if available, or fall back to YOLOv10.

If you want to download the latest models:
1. Run the download script:
   ```
   python download_yolo_model.py
   ```
2. Select the model you want to download when prompted

## Customizing Detection

You can adjust the confidence threshold for detections:
```
python luggage_detector.py --source file.mp4 --confidence 0.7
```
