# 🧳 Luggage Detection with YOLO 🚀

This project implements **luggage detection** using the powerful **YOLO (You Only Look Once)** object detection model. Ideal for surveillance, security systems, and smart transport hubs.

---

## ⚙️ Setup

1. 📦 Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. ⬇️ Download YOLO models:

   ```bash
   python download_yolo_model.py
   ```

   * This will download **YOLOv11** and all **YOLOv10** versions.

---

## ▶️ Running the Detector

🎥 Run on a **video file**:

```bash
python luggage_detector.py --source file.mp4
```

📷 Run using your **webcam**:

```bash
python luggage_detector.py --source 0
```

🎯 Run with a **custom confidence threshold** (e.g., 0.7):

```bash
python luggage_detector.py --source your_path --confidence 0.7
```
## Screenshot
![image](https://github.com/user-attachments/assets/939958e7-9f2f-436e-991e-7491cd379da7)

---

## 🎮 Controls

* Press **`q`** or **`Esc`** ➜ Quit
* Press **`x`** ➜ Close the window
* Click the **❌ (close button)** on the window to exit

---

## 🔄 Getting the Latest YOLO Model

By default, the script will:

* Try to use **YOLOv11**
* Fallback to **YOLOv10** if v11 isn't available

📥 To manually download or update models:

```bash
python download_yolo_model.py
```

* Choose the model version when prompted.

---

## 🛠️ Customizing Detection

You can fine-tune the **confidence threshold** to reduce false detections or detect more objects:

```bash
python luggage_detector.py --source file.mp4 --confidence 0.7
```

---

## 📌 Notes

* Make sure your video source (file path or webcam) is accessible.
* YOLO models are downloaded only once and reused unless deleted.

---


