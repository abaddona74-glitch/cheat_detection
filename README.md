#  AI Cheat Detection System

An advanced automated proctoring system designed to detect suspicious behavior during exams using Computer Vision and AI. The system monitors students in real-time, tracking head pose, gaze direction, and detecting prohibited objects like mobile phones.

##  Key Features

###  Advanced Face Registration
*   **3-Step Process:** Captures **Front**, **Left**, and **Right** angles for robust recognition.
*   **Smart UI:** Provides real-time feedback (distance, angle) and auto-captures when the pose is perfect.
*   **Session Management:** Automatically resets violation counters when a new session starts.

###  High-Precision Object Detection
*   **YOLOv8 Integration:** Uses the state-of-the-art **YOLOv8** model for phone detection.
*   **Tuned Accuracy:** Optimized confidence thresholds (>60%) to eliminate false positives ("ghost" phones).
*   **Visual Alert:** Draws a red grid mesh over detected phones.

###  Gaze & Head Pose Tracking
*   **State Machine Logic:**
    *   **Normal:** Green face box.
    *   **Looking Away:** Orange box with a **5-second countdown** timer displayed on the face.
    *   **Violation:** Recorded if the user looks away for >5 seconds or repeatedly glances back.
*   **3D Axis Visualization:** Shows gaze direction with large colored arrows.
*   **Dense Face Mesh:** High-density 468-point face mesh in custom **Light Blue (#9aeffa)** and **Yellowish (#9cdeff)** colors.

###  Visualizations
*   **Unified Theme:** Face mesh, body pose (skeleton), and hand landmarks all share a consistent, clean color scheme.
*   **Clean UI:** Removed distracting eye contours and iris markers for a cleaner look.
*   **Status Overlay:** Real-time FPS (App & Camera) and system status monitoring.

###  Security & Blocking
*   **Auto-Block:** Users are automatically blocked (Purple Box) after:
    *   **1** Phone detection.
    *   **5** Backward looking violations.
*   **Evidence Logging:** Saves timestamped images of every violation in the cheat_images/ folder.

##  Requirements

*   Python 3.8+
*   Webcam (supports GPU acceleration via OpenCL)

##  Installation

1.  **Clone the repository:**
    `ash
    git clone <repository-url>
    cd cheat_detection
    `

2.  **Install dependencies:**
    `ash
    pip install -r requirements.txt
    `
    *Dependencies include: ultralytics, mediapipe, ace_recognition, opencv-python, 
umpy, Pillow.*

##  Usage

### 1. Start the System
Run the main script to open the Control Panel:
`ash
python main.py
`

### 2. Dashboard Controls
*   **Register New Face:** Opens the registration wizard. Follow the on-screen instructions.
*   **View Faces:** Manage the list of registered users.
*   **View Violations:** See a table of all recorded cheating attempts.
*   **START DETECTION:** Launches the main monitoring window.

### 3. Monitoring Window Indicators
*   ** Green Box:** Normal behavior.
*   ** Orange Box:** Warning (Looking away). Watch the countdown!
*   ** Red Box:** Phone detected (Immediate Violation).
*   ** Purple Box:** User is BLOCKED.

##  Project Structure

*   main.py: Core logic, detection loop, and visualization.
*   dashboard.py: GUI for registration and management.
*   ace_database.pkl: Encoded face data.
*   iolations_database.pkl: Database of recorded violations.
*   cheat_images/: Folder containing evidence photos of violations.
*   equirements.txt: Python dependencies.

##  Troubleshooting
*   **"Unknown" Face:** If the system doesn't recognize you, try re-registering with better lighting. The recognition tolerance has been tuned to **0.65** for better accuracy.
*   **False Phone Alerts:** The system requires 60% confidence to flag a phone. Ensure no rectangular, phone-like objects are in the frame.
