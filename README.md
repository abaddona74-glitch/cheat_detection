# 🚨 AI Cheat Detection System

An advanced automated proctoring system designed to detect suspicious behavior during exams using Computer Vision and AI. The system monitors students in real-time, tracking head pose, gaze direction, and detecting prohibited objects like mobile phones.

## ✨ Key Features

*   **👤 Advanced Face Registration:** 
    *   **3-Step Process:** Captures Front, Left, and Right angles for robust recognition.
    *   **Smart UI:** Provides real-time feedback (distance, angle) and auto-captures when the pose is perfect.
*   **📱 YOLOv8 Phone Detection:** 
    *   Uses the state-of-the-art **YOLOv8** model to detect mobile phones with high accuracy.
    *   Visualizes detected phones with a red mesh grid.
*   **👀 Gaze & Head Pose Tracking:** 
    *   **3D Axis Visualization:** Shows gaze direction with large colored arrows (Blue=Forward, Green=Down, Red=Right).
    *   **Degree Display:** Real-time display of Yaw and Pitch angles.
    *   **Iris Tracking:** Precise pupil tracking with visual indicators.
*   **🧠 Smart Violation Logic:**
    *   **State Machine:** Distinguishes between quick glances and prolonged looking away.
    *   **5-Second Rule:** Looking away for more than 5 seconds triggers a "Looking Backward" violation.
    *   **Glance Detection:** Repeated quick glances are also flagged.
*   **🖥️ Admin Dashboard:**
    *   User-friendly GUI to manage registrations and view databases.
    *   **Violations Table:** View detailed logs of phone detections and gaze violations.
    *   **Blocked Status:** Automatically blocks users after repeated violations.

## 🛠 Requirements

*   Python 3.8+
*   Webcam

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd cheat_detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This will install `ultralytics` (for YOLOv8), `mediapipe`, `face_recognition`, `opencv-python`, and `tkinter`.*

## 🖥 Usage

### 1. Start the System
Run the main script to open the Control Panel:
```bash
python main.py
```

### 2. Dashboard Controls
*   **Register New Face:** Opens the registration wizard. Follow the on-screen instructions to capture your face from 3 angles.
*   **View Faces:** Manage the list of registered users.
*   **View Violations:** See a table of all recorded cheating attempts.
*   **START DETECTION:** Launches the main monitoring window.

### 3. Monitoring Window
Once detection starts:
*   **Green Box:** Normal behavior.
*   **Orange Box:** Looking away (Warning/Violation).
*   **Red Box:** Phone detected.
*   **Purple Box:** User is BLOCKED.

### 4. Debug Visualizations
The system draws several overlays for debugging:
*   **Face Mesh:** Tesselation and contours of the face.
*   **Skeleton:** Body pose and hand tracking.
*   **Gaze Arrows:** 3D vectors showing head direction.
*   **Phone Grid:** Red mesh over detected phones.

## 📂 Project Structure

*   `main.py`: Core logic, detection loop, and visualization.
*   `dashboard.py`: GUI for registration and management.
*   `face_database.pkl`: Encoded face data.
*   `violations_database.pkl`: Database of recorded violations.
*   `requirements.txt`: Python dependencies.

## ⚠️ Troubleshooting
*   **YOLO Error:** If you see an error about `yolov8n.pt`, ensure you have an internet connection on the first run to download the model weights.
*   **Camera:** If the camera doesn't open, check the `camera_config.json` or ensure no other app is using the webcam.
