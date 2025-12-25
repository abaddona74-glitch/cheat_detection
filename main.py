import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition
import pickle
import os
from datetime import datetime
import mediapipe as mp
from time import time
import threading
import time as time_module

# Yuzlar ma'lumotlar bazasi fayli
FACE_DB_FILE = "face_database.pkl"
# Bloklanganlar va qoidabuzarlik bazasi
VIOLATIONS_DB_FILE = "violations_database.pkl"

# Yuzning turli nuqtalari uchun indekslar
FACE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291
}

# Yuz tanib olish tizimi sinfi
class FaceRecognitionSystem:
    def __init__(self):
        try:
            print("[INFO] System starting...", flush=True)
            self.known_face_encodings = []
            self.known_face_names = []
            
            print("[DEBUG] Loading face database...", flush=True)
            self.load_face_database()
            
            # Qoidabuzarliklar va bloklanganlar bazasini yaratish
            self.violations = {}
            self.blocked_users = []
            
            print("[DEBUG] Loading violations database...", flush=True)
            self.load_violations_database()
            
            # Yuz holati kuzatuvi (State Machine)
            # Format: {name: {'state': 'NORMAL', 'start_time': 0}}
            # States: NORMAL, WARNING, VIOLATED
            self.face_states = {}

            # FaceMesh modelini yaratish
            print("[DEBUG] Loading FaceMesh model...", flush=True)
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True, # Ko'z qorachig'ini aniqlash uchun
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            # Pose (Tana) modelini yaratish
            print("[DEBUG] Loading Pose model...", flush=True)
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            # Hands (Qo'l) modelini yaratish
            print("[DEBUG] Loading Hands model...", flush=True)
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            # 3D bosh pozitsiyasini aniqlash uchun model nuqtalari
            self.model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1)
            ], dtype=np.float32)

            # Telefonni aniqlash uchun YOLOv8 modeli
            print("[DEBUG] Loading YOLOv8 model...", flush=True)
            try:
                self.phone_model = YOLO("yolov8n.pt")
            except Exception as e:
                print(f"[ERROR] Could not load YOLO model: {e}")
                self.phone_model = None
            print("[DEBUG] System initialized successfully.", flush=True)
        except Exception as e:
            # Xatolikni faylga yozish
            try:
                with open("error_log.txt", "w", encoding="utf-8") as f:
                    import traceback
                    traceback.print_exc(file=f)
            except:
                pass # Faylga yozishda xatolik bo'lsa, indamaymiz

            # Konsolga chiqarishga urinish (agar iloji bo'lsa)
            try:
                print(f"[ERROR] Init failed: {e}", flush=True)
                import traceback
                traceback.print_exc()
            except:
                pass # Konsol ishlamasa, indamaymiz
            
            # Dasturni to'xtatish
            raise e

    # Yuzlar ma'lumotlar bazasini yuklash
    def load_face_database(self):
        if os.path.exists(FACE_DB_FILE):  # Fayl mavjud bo'lsa
            with open(FACE_DB_FILE, 'rb') as f:
                data = pickle.load(f)  # Ma'lumotlarni fayldan yuklash
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
                print(f"[INFO] {len(self.known_face_names)} ta yuz yuklandi")
                
    # Qoidabuzarliklar ma'lumotlar bazasini yuklash
    def load_violations_database(self):
        if os.path.exists(VIOLATIONS_DB_FILE):  # Fayl mavjud bo'lsa
            with open(VIOLATIONS_DB_FILE, 'rb') as f:
                data = pickle.load(f)  # Ma'lumotlarni fayldan yuklash
                self.violations = data['violations']
                self.blocked_users = data['blocked_users']
                print(f"[INFO] {len(self.blocked_users)} ta bloklangan foydalanuvchilar yuklandi")
        else:
            self.violations = {}
            self.blocked_users = []

    # Yuzlar ma'lumotlar bazasini saqlash
    def save_face_database(self):
        with open(FACE_DB_FILE, 'wb') as f:
            pickle.dump({
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }, f)
            
    # Qoidabuzarliklar ma'lumotlar bazasini saqlash
    def save_violations_database(self):
        with open(VIOLATIONS_DB_FILE, 'wb') as f:
            pickle.dump({
                'violations': self.violations,
                'blocked_users': self.blocked_users
            }, f)

    def process_gaze_state(self, name, is_looking_backward):
        if name == "Unknown": return False
        
        current_time = time()
        if name not in self.face_states:
            self.face_states[name] = {'state': 'NORMAL', 'start_time': 0}
            
        state_info = self.face_states[name]
        state = state_info['state']
        
        violation_confirmed = False
        
        if is_looking_backward:
            if state == 'NORMAL':
                # Endi qarashni boshladi
                self.face_states[name]['state'] = 'WARNING'
                self.face_states[name]['start_time'] = current_time
            elif state == 'WARNING':
                # Qarab turibdi
                elapsed = current_time - state_info['start_time']
                if elapsed >= 5.0: # 5 sekund qarab tursa
                    violation_confirmed = True
                    self.face_states[name]['state'] = 'VIOLATED' # Violation yozildi, endi kutamiz
        else:
            # Normal holatga qaytdi
            if state == 'WARNING':
                # Qarab turib qaytdi (Glance) - bu ham violation
                violation_confirmed = True
                self.face_states[name]['state'] = 'NORMAL'
            elif state == 'VIOLATED':
                # Uzoq qarab turib qaytdi
                self.face_states[name]['state'] = 'NORMAL'
            else:
                self.face_states[name]['state'] = 'NORMAL'
                
        return violation_confirmed

    # Qoidabuzarliklarni qayd qilish
    def record_violation(self, name, violation_type):
        if name == "Unknown":
            return
            
        if name not in self.violations:
            self.violations[name] = {"backward_look": 0, "phone": 0}
            
        # Qoidabuzarlikni qayd qilish
        if violation_type == "backward_look":
            self.violations[name]["backward_look"] += 1
        elif violation_type == "phone":
            self.violations[name]["phone"] += 1
            
        # Bloklash shartlarini tekshirish
        if violation_type == "phone" or self.violations[name]["backward_look"] >= 5:
            if name not in self.blocked_users:
                self.blocked_users.append(name)
                self.log_cheat(name, f"BLOCKED: due to {violation_type}")
                print(f"[BLOCKED] {name} - blocked due to {violation_type}")
                
        self.save_violations_database()
        
    # Foydalanuvchi bloklangan yoki yo'qligini tekshirish
    def is_blocked(self, name):
        return name in self.blocked_users

    # Yuzlarni noqonuniy (cheat) holatlarini loglash
    def log_cheat(self, name, reason):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Hozirgi vaqtni olish
        with open("log.txt", "a") as f:
            f.write(f"[{now}] Name: {name} | Reason: {reason}\n")

    # Noqonuniy holatda kadrni saqlash
    def save_cheat_frame(self, frame, name):
        # Bugungi sana uchun papka yaratish (DD.MM.YYYY formatida)
        today_date = datetime.now().strftime("%d.%m.%Y")
        date_folder = os.path.join("cheat_images", today_date)
        os.makedirs(date_folder, exist_ok=True)
        
        # Foydalanuvchi uchun papka yaratish
        user_folder = os.path.join(date_folder, name)
        os.makedirs(user_folder, exist_ok=True)
        
        # Rasm fayli nomini yaratish (vaqt bilan)
        now = datetime.now().strftime("%H-%M-%S")  # Hozirgi vaqt
        filename = f"{user_folder}/{now}.jpg"  # Fayl nomini yaratish

        # Rasmga vaqt qo'shish
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Rasmni saqlash
        cv2.imwrite(filename, frame)
        print(f"[INFO] Cheat image saved: {filename}")

    def detect_phone(self, frame):
        phone_detected = False
        max_conf = 0.0
        phone_rects = [] # Will contain phone bounding boxes

        if self.phone_model:
            try:
                # YOLOv8 inference on the full frame
                results = self.phone_model(frame, verbose=False)
                
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Class 67 is 'cell phone' in COCO dataset
                        if cls == 67 and conf > 0.4: 
                            phone_detected = True
                            max_conf = max(max_conf, conf * 100)
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            phone_rects.append((x1, y1, x2, y2))
                            
            except Exception as e:
                print(f"[ERROR] YOLO detection failed: {e}")
        
        return phone_detected, max_conf, phone_rects

    # Yuzni qo'shish
    def add_face(self, frame, name):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Rasmni RGB formatiga o'zgartirish
        face_locations = face_recognition.face_locations(rgb_frame)  # Yuzni topish

        if not face_locations:  # Agar yuz topilmasa
            return False, "Face not found"

        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]  # Yuzning kodini olish
        self.known_face_encodings.append(face_encoding)  # Yuzni ma'lumotlar bazasiga qo'shish
        self.known_face_names.append(name)  # Ismni ma'lumotlar bazasiga qo'shish

        self.save_face_database()  # Ma'lumotlar bazasini saqlash
        return True, "Face added successfully"  # Muvaffaqiyatli qo'shildi

    # Yuzni aniqlash
    def identify_face(self, frame, face_location):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Rasmni RGB formatiga o'zgartirish
        face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])[0]  # Yuz kodini olish

        name = "Unknown"  # Yuzni aniqlash uchun default nom
        if self.known_face_encodings:  # Agar ma'lumotlar bazasida yuzlar bo'lsa
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)  # Yuzlar orasidagi masofa
            if len(distances) > 0:
                best_match = np.argmin(distances)  # Eng yaqin yuzni topish
                if distances[best_match] < 0.6:  # Agar masofa kichik bo'lsa (0.6 ga oshirildi)
                    name = self.known_face_names[best_match]  # Nomni olish
        return name  # Yuzning nomini qaytarish

    # Orqaga qarayotgan yuzni aniqlash
    def detect_backward_looking(self, frame, face_location):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Rasmni RGB formatiga o'zgartirish
            height, width = frame.shape[:2]  # Rasm o'lchamlari
            results = self.face_mesh.process(rgb_frame)  # FaceMesh modelini ishlatish
            if not results.multi_face_landmarks:  # Agar yuzning nuqtalari mavjud bo'lmasa
                return False, None

            landmarks = results.multi_face_landmarks[0].landmark  # Yuzning nuqtalarini olish

            image_points = np.array([  # Yuzning asosiy nuqtalarini olish
                (landmarks[FACE_LANDMARKS["nose_tip"]].x * width,
                 landmarks[FACE_LANDMARKS["nose_tip"]].y * height),
                (landmarks[FACE_LANDMARKS["chin"]].x * width,
                 landmarks[FACE_LANDMARKS["chin"]].y * height),
                (landmarks[FACE_LANDMARKS["left_eye"]].x * width,
                 landmarks[FACE_LANDMARKS["left_eye"]].y * height),
                (landmarks[FACE_LANDMARKS["right_eye"]].x * width,
                 landmarks[FACE_LANDMARKS["right_eye"]].y * height),
                (landmarks[FACE_LANDMARKS["left_mouth"]].x * width,
                 landmarks[FACE_LANDMARKS["left_mouth"]].y * height),
                (landmarks[FACE_LANDMARKS["right_mouth"]].x * width,
                 landmarks[FACE_LANDMARKS["right_mouth"]].y * height)
            ], dtype=np.float32)

            focal_length = width  # Fokal uzunlik
            center = (width / 2, height / 2)  # Markaz
            camera_matrix = np.array([  # Kameraning matritsasi
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)

            dist_coeffs = np.zeros((4, 1))  # Distorsiya koeffitsiyentlari

            success, rotation_vec, translation_vec = cv2.solvePnP(  # PnP yechimi
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # Boshni aylantirish
                yaw = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0]) * 180 / np.pi  # Yaw burchagini hisoblash
                
                # Ko'z ma'lumotlarini olish
                left_iris = landmarks[468]
                right_iris = landmarks[473]
                
                # Ko'z atrofidagi nuqtalar (chegaralar uchun)
                # Mediapipe FaceMesh indekslari
                left_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
                right_eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 382, 381, 380, 374, 373, 390, 249]
                
                def get_pt(idx): return (int(landmarks[idx].x * width), int(landmarks[idx].y * height))
                
                eye_data = {
                    "left_iris": (int(left_iris.x * width), int(left_iris.y * height)),
                    "right_iris": (int(right_iris.x * width), int(right_iris.y * height)),
                    "left_eye_rect": cv2.boundingRect(np.array([get_pt(i) for i in left_eye_indices])),
                    "right_eye_rect": cv2.boundingRect(np.array([get_pt(i) for i in right_eye_indices])),
                    "left_eye_points": [get_pt(i) for i in left_eye_indices],
                    "right_eye_points": [get_pt(i) for i in right_eye_indices]
                }
                
                pose_data = (rotation_vec, translation_vec)
                return abs(yaw) > 50, image_points, eye_data, results.multi_face_landmarks[0], pose_data
        except Exception as e:
            print(f"[ERROR] Bosh pozitsiyasini aniqlashda xatolik: {str(e)}")  # Xatolikni chiqarish
            return False, None, None, None, None
        return False, None, None, None, None

import threading
import time as time_module # time modulini qayta nomlaymiz, chunki pastda time() funksiyasi bor

# ... (boshqa importlar)

# Asosiy dastur
def main():
    print("[INFO] Face recognition system starting...")
    
    # Kamera tanlash oynasini chaqirish
    from camera_selector import select_camera
    settings = select_camera()
    
    source = settings["source"]
    resolution_str = settings["resolution"]
    fps_limit_str = settings["fps_limit"]
    
    if source is None:
        print("[INFO] Program stopped (no camera selected).")
        return

    # Resolutionni parse qilish
    target_width, target_height = 1024, 768 # Default
    if "144p" in resolution_str: target_width, target_height = 256, 144
    elif "240p" in resolution_str: target_width, target_height = 426, 240
    elif "360p" in resolution_str: target_width, target_height = 640, 360
    elif "480p 4:3" in resolution_str: target_width, target_height = 640, 480
    elif "480p" in resolution_str: target_width, target_height = 854, 480
    elif "720p" in resolution_str: target_width, target_height = 1280, 720
    elif "1080p" in resolution_str: target_width, target_height = 1920, 1080
    elif "640x480" in resolution_str: target_width, target_height = 640, 480 # Eski configlar uchun

    # FPS limitni parse qilish
    target_fps = 0 # Cheklovsiz
    if "30 FPS" in fps_limit_str: target_fps = 30
    elif "15 FPS" in fps_limit_str: target_fps = 15
    elif "5 FPS" in fps_limit_str: target_fps = 5

    face_system = FaceRecognitionSystem()

    print(f"[INFO] Camera starting... Source: {source}")
    print(f"[INFO] Settings: {target_width}x{target_height}, FPS Limit: {fps_limit_str}")

    # AMD RX580 yoki boshqa GPU lardan foydalanish uchun OpenCL ni yoqish
    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            print(f"[INFO] GPU Optimization (OpenCL): ENABLED ({cv2.ocl.useOpenCL()})")
            print(f"[INFO] Device: {cv2.ocl.Device.getDefault().name()}")
        else:
            print("[INFO] OpenCL not available, using CPU.")
    except Exception as e:
        print(f"[WARNING] OpenCL setup error: {e}")

    # --- THREADED VIDEO CAPTURE ---
    # Kameradan o'qishni alohida oqimga olamiz.
    # Bu tarmoq sekinlashganda dastur qotib qolishining oldini oladi.
    class VideoStream:
        def __init__(self, src, width, height):
            # FFMPEG uchun maxsus headerlar (Skyline va boshqa himoyalangan kameralar uchun)
            if isinstance(src, str):
                # Bu yerda biz FFMPEG ga Referer va User-Agent ni uzatamiz
                # Bu Skyline kabi saytlar bizni bloklamasligi uchun kerak
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "headers;Referer: https://www.skylinewebcams.com/\r\nUser-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

            self.stream = cv2.VideoCapture(src, cv2.CAP_FFMPEG if isinstance(src, str) else cv2.CAP_ANY)
            
            # Hardware Acceleration
            try:
                self.stream.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                print(f"[INFO] Hardware Acceleration (Decode): ENABLED (Mode: {self.stream.get(cv2.CAP_PROP_HW_ACCELERATION)})")
            except: pass

            # Resolution
            if not isinstance(src, str):
                self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            (self.grabbed, self.frame) = self.stream.read()
            self.stopped = False
            self.lock = threading.Lock()
            self.width = width
            self.height = height
            self.frame_id = 0 # Har bir yangi kadr uchun ID

        def start(self):
            threading.Thread(target=self.update, args=(), daemon=True).start()
            return self

        def update(self):
            while not self.stopped:
                grabbed, frame = self.stream.read()
                if not grabbed:
                    # Agar oqim uzilsa, qayta ulanishga urinib ko'rish mumkin
                    # Hozircha shunchaki to'xtatamiz
                    self.stopped = True
                    break
                
                # Resize ni shu yerda qilamiz (agar kerak bo'lsa)
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                # Mirror effect (Ko'zgu effekti)
                frame = cv2.flip(frame, 1)

                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame
                    self.frame_id += 1 # Yangi kadr keldi
                
                # CPU ni biroz bo'shatish (juda muhim)
                time_module.sleep(0.001)

        def read(self):
            with self.lock:
                # Kadr va uning ID sini qaytaramiz
                return (self.frame.copy() if self.frame is not None else None), self.frame_id

        def stop(self):
            self.stopped = True
            self.stream.release()

    # Video oqimni ishga tushirish
    print("[INFO] Video stream starting in separate thread...")
    video_stream = VideoStream(source, target_width, target_height).start()
    
    # Biroz kutamiz, kamera o'nglanib olsin
    time_module.sleep(1.0)

    if video_stream.stopped:
        print(f"[ERROR] Camera/Stream failed to open: {source}")
        return
    else:
        print("[INFO] Camera started successfully.")

    # --- DASHBOARD (Yangi UI) ---
    from dashboard import show_dashboard
    print("[INFO] Showing Dashboard...")
    
    # Dashboardni ko'rsatish (Start bosilmaguncha shu yerda turadi)
    should_start = show_dashboard(face_system, video_stream)
    
    if not should_start:
        print("[INFO] User cancelled/closed dashboard.")
        video_stream.stop()
        return

    training_mode = False  # O'qitish rejimi
    current_name = ""  # Joriy ism
    last_save_time = 0  # Oxirgi saqlangan vaqti (sekundda)
    
    # FPS nazorati uchun
    prev_frame_time = 0

    # --- MULTITHREADING SETUP ---
    # Asosiy oqim (UI) va Hisoblash oqimi (Processing) ni ajratamiz.
    # Bu FPS ni oshiradi, chunki UI hisob-kitob tugashini kutib o'tirmaydi.
    
    # Mediapipe Drawing Utils
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    shared_data = {
        "frame": None,
        "results": {
            "face_locations": [],
            "face_names": [],
            "phone_detected": False,
            "phone_confidence": 0.0,
            "backward_looking_status": [],
            "landmarks": [],
            "eye_data": [],
            "face_mesh_landmarks": [], # To'liq mesh uchun
            "pose_landmarks": None, # Tana uchun
            "hand_landmarks": [], # Qo'llar uchun
            "hand_rects": [] # Telefon aniqlangan qo'l sohalari
        },
        "running": True,
        "lock": threading.Lock(),
        "new_frame_available": False,
        "processing_time": 0.0 # Qayta ishlash vaqti (ms)
    }

    def processing_loop(face_system, shared_data):
        """Og'ir hisob-kitoblarni bajaruvchi alohida oqim"""
        while shared_data["running"]:
            frame_to_process = None
            
            # Yangi kadr bormi tekshiramiz
            with shared_data["lock"]:
                if shared_data["new_frame_available"]:
                    frame_to_process = shared_data["frame"].copy()
                    shared_data["new_frame_available"] = False
            
            if frame_to_process is not None:
                start_proc = time()
                try:
                    # --- OPTIMIZATSIYA: Rasmni kichraytirish ---
                    # Katta rasmda yuz qidirish juda sekin. Biz uni vaqtincha kichraytiramiz.
                    # Bu aniqlikka deyarli ta'sir qilmaydi, lekin tezlikni 3-4 barobar oshiradi.
                    process_width = 320
                    height, width = frame_to_process.shape[:2]
                    scale_factor = width / float(process_width)
                    process_height = int(height / scale_factor)
                    
                    small_frame = cv2.resize(frame_to_process, (process_width, process_height))
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # 1. Yuzlarni aniqlash (Kichik kadrda)
                    # Bu eng og'ir jarayon, endi u kichik rasmda bo'lgani uchun tezlashadi
                    small_face_locations = face_recognition.face_locations(rgb_small_frame)
                    
                    # Topilgan koordinatalarni asl o'lchamga qaytarish
                    face_locations = []
                    for (top, right, bottom, left) in small_face_locations:
                        top = int(top * scale_factor)
                        right = int(right * scale_factor)
                        bottom = int(bottom * scale_factor)
                        left = int(left * scale_factor)
                        face_locations.append((top, right, bottom, left))
                    
                    # 2. Pose (Tana) aniqlash
                    pose_results = face_system.pose.process(rgb_small_frame) # Kichik kadrda tezroq
                    pose_landmarks = None
                    if pose_results.pose_landmarks:
                        pose_landmarks = pose_results.pose_landmarks

                    # 3. Hands (Qo'l) aniqlash
                    # Qo'llarni aniqlash uchun asl kadrni ishlatamiz (aniqlik uchun)
                    # Lekin tezlik uchun kichik kadrni ishlatish ham mumkin. Keling, kichik kadrni sinab ko'ramiz.
                    hands_results = face_system.hands.process(rgb_small_frame)
                    hand_landmarks_list = []
                    
                    if hands_results.multi_hand_landmarks:
                        hand_landmarks_list = hands_results.multi_hand_landmarks
                    
                    # Telefonni aniqlash (YOLOv8) - Butun kadr bo'ylab
                    phone_detected, phone_confidence, hand_rects = face_system.detect_phone(frame_to_process)

                    face_names = []
                    backward_looking_status = []
                    all_landmarks = []
                    all_eye_data = []
                    all_mesh_landmarks = []
                    all_pose_data = []
                    
                    for face_location in face_locations:
                        # 3. Yuzni tanish (Asl kadrda, lekin aniqlangan joylarda)
                        name = face_system.identify_face(frame_to_process, face_location)
                        face_names.append(name)
                        
                        # 4. Bosh holatini aniqlash
                        is_looking_backward, landmarks, eye_data, mesh_landmarks, pose_data = face_system.detect_backward_looking(frame_to_process, face_location)
                        backward_looking_status.append(is_looking_backward)
                        all_landmarks.append(landmarks)
                        all_eye_data.append(eye_data)
                        all_mesh_landmarks.append(mesh_landmarks)
                        all_pose_data.append(pose_data)
                        
                        # Qoidabuzarliklarni yozish
                        if phone_detected:
                            face_system.log_cheat(name, "Phone detected")
                            face_system.record_violation(name, "phone")
                        
                        # Gaze violation logic with state machine
                        if face_system.process_gaze_state(name, is_looking_backward):
                            face_system.log_cheat(name, "Looking backward")
                            face_system.record_violation(name, "backward_look")

                    # Natijalarni yangilash
                    with shared_data["lock"]:
                        shared_data["results"] = {
                            "face_locations": face_locations,
                            "face_names": face_names,
                            "phone_detected": phone_detected,
                            "phone_confidence": phone_confidence,
                            "backward_looking_status": backward_looking_status,
                            "landmarks": all_landmarks,
                            "eye_data": all_eye_data,
                            "face_mesh_landmarks": all_mesh_landmarks,
                            "head_pose_data": all_pose_data,
                            "pose_landmarks": pose_landmarks,
                            "hand_landmarks": hand_landmarks_list,
                            "hand_rects": hand_rects
                        }
                        shared_data["processing_time"] = time() - start_proc
                except Exception as e:
                    print(f"[ERROR] Thread error: {e}")
            else:
                time_module.sleep(0.005) # CPU ni bo'shatish uchun kichik pauza

    # Threadni ishga tushirish
    process_thread = threading.Thread(target=processing_loop, args=(face_system, shared_data))
    process_thread.daemon = True
    process_thread.start()
    print("[INFO] Multithreading mode started (Asynchronous Processing)")

    # Oynani yaratish va sozlash (Fullscreen/Windowed)
    window_name = "Face Recognition System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Oynani maksimal darajada ochish (Windowed Fullscreen)
    try:
        # Windows uchun maxsus buyruq (oynani maximize qilish)
        # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        pass
    except:
        pass

    # Diagnostika uchun o'zgaruvchilar
    fps_counter = 0
    fps_start_time = time()
    current_fps = 0
    
    # Source FPS (Kamera tezligi) ni o'lchash uchun
    source_fps_counter = 0
    source_fps = 0
    last_frame_id = -1
    
    last_frame_arrival = time()
    debug_mode = True # Debug rejimi (nuqtalarni ko'rsatish)

    while True:
        # FPS Limitni qo'lda boshqarish
        if target_fps > 0:
            time_elapsed = time() - prev_frame_time
            if time_elapsed < 1./target_fps:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                cv2.waitKey(1)
                continue 
            prev_frame_time = time()

        # Kadrni Thread-dan o'qish (Endi bu bloklamaydi!)
        frame, frame_id = video_stream.read()
        
        if frame is None:
            # Agar kadr bo'lmasa (stream uzilgan yoki hali yuklanmagan)
            if video_stream.stopped:
                print("Video stream stopped.")
                break
            
            # Kutish rejimini ko'rsatish
            blank = np.zeros((target_height, target_width, 3), np.uint8)
            cv2.putText(blank, "WAITING...", (50, target_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(window_name, blank)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Yangi kadr kelganini tekshirish (Source FPS uchun)
        if frame_id != last_frame_id:
            source_fps_counter += 1
            last_frame_id = frame_id
            last_frame_arrival = time() # Kadr keldi
        else:
            # Agar kadr o'zgarmagan bo'lsa, CPU ni tejash uchun biroz kutamiz
            # Lekin UI qotib qolmasligi uchun juda kam
            time_module.sleep(0.001)

        # Kadrni processing threadga yuborish
        with shared_data["lock"]:
            shared_data["frame"] = frame
            shared_data["new_frame_available"] = True
            current_results = shared_data["results"]
            proc_time = shared_data["processing_time"]

        # Natijalarni chizish (current_results dan foydalanamiz)
        face_locations = current_results["face_locations"]
        face_names = current_results["face_names"]
        phone_detected = current_results["phone_detected"]
        phone_confidence = current_results["phone_confidence"]
        backward_looking_status = current_results["backward_looking_status"]
        landmarks_list = current_results.get("landmarks", [])
        eye_data_list = current_results.get("eye_data", [])
        mesh_landmarks_list = current_results.get("face_mesh_landmarks", [])
        head_pose_data_list = current_results.get("head_pose_data", [])
        pose_landmarks = current_results.get("pose_landmarks", None)
        hand_landmarks_list = current_results.get("hand_landmarks", [])
        hand_rects = current_results.get("hand_rects", [])

        # Tana (Pose) chizish
        if pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Qo'llar (Hands) chizish
        for hand_landmarks in hand_landmarks_list:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

        # Telefon aniqlangan qo'l sohalarini chizish
        for (hx1, hy1, hx2, hy2) in hand_rects:
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 3)
            cv2.putText(frame, "PHONE DETECTED", (hx1, hy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw grid (setka) on phone
            step = 20
            for x in range(hx1, hx2, step):
                cv2.line(frame, (x, hy1), (x, hy2), (0, 0, 255), 1)
            for y in range(hy1, hy2, step):
                cv2.line(frame, (hx1, y), (hx2, y), (0, 0, 255), 1)
            # Diagonal lines for mesh effect
            cv2.line(frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 1)
            cv2.line(frame, (hx2, hy1), (hx1, hy2), (0, 0, 255), 1)

        # Ro'yxatlar uzunligi mos kelishini ta'minlash
        if len(landmarks_list) != len(face_locations):
             landmarks_list = [None] * len(face_locations)
        if len(eye_data_list) != len(face_locations):
             eye_data_list = [None] * len(face_locations)
        if len(mesh_landmarks_list) != len(face_locations):
             mesh_landmarks_list = [None] * len(face_locations)
        if len(head_pose_data_list) != len(face_locations):
             head_pose_data_list = [None] * len(face_locations)

        for (top, right, bottom, left), name, is_looking_backward, landmarks, eye_info, mesh_landmarks, pose_data in zip(face_locations, face_names, backward_looking_status, landmarks_list, eye_data_list, mesh_landmarks_list, head_pose_data_list):
            
            # Foydalanuvchi bloklangan yoki yo'qligini tekshirish
            is_blocked = face_system.is_blocked(name)

            color = (0, 255, 0)  # Yuzni yashil rangda belgilash
            if is_blocked:
                color = (128, 0, 128)  # Bloklangan bo'lsa, to'q pushti rangda belgilash
            elif phone_detected:
                color = (0, 0, 255)  # Telefon topilgan bo'lsa, qizil rangda belgilash
            elif is_looking_backward:
                color = (0, 165, 255)  # Orqaga qarayotgan bo'lsa, to'q sariq rangda belgilash

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)  # Yuzga to'rtburchak chizish

            # Debug rejimi: Yuz nuqtalarini chizish
            if debug_mode:
                # To'liq meshni chizish
                if mesh_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=mesh_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=mesh_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                if landmarks is not None:
                    for point in landmarks:
                        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 255, 255), -1)
                    # Nuqtalarni tutashtiruvchi chiziqlar (vizualizatsiya uchun)
                    # Burun uchi
                    nose = (int(landmarks[0][0]), int(landmarks[0][1]))
                    # Ko'zlar
                    left_eye = (int(landmarks[2][0]), int(landmarks[2][1]))
                    right_eye = (int(landmarks[3][0]), int(landmarks[3][1]))
                    
                    cv2.line(frame, nose, left_eye, (255, 0, 0), 1)
                    cv2.line(frame, nose, right_eye, (255, 0, 0), 1)
                
                if eye_info is not None:
                    # Ko'z konturlarini chizish (Nuqtalar bilan)
                    for pt in eye_info["left_eye_points"]:
                        cv2.circle(frame, pt, 1, (255, 255, 0), -1)
                    for pt in eye_info["right_eye_points"]:
                        cv2.circle(frame, pt, 1, (255, 255, 0), -1)

                    # Ko'z qorachig'ini chizish (Katta sariq aylana va qizil markaz)
                    # Left Iris
                    cv2.circle(frame, eye_info["left_iris"], 4, (0, 255, 255), 1) # Yellow ring
                    cv2.circle(frame, eye_info["left_iris"], 2, (0, 0, 255), -1)  # Red dot
                    # Right Iris
                    cv2.circle(frame, eye_info["right_iris"], 4, (0, 255, 255), 1) # Yellow ring
                    cv2.circle(frame, eye_info["right_iris"], 2, (0, 0, 255), -1)  # Red dot

                # Gaze Direction Visualization (Axis)
                if pose_data is not None:
                    try:
                        rvec, tvec = pose_data
                        
                        # Calculate angles for display
                        rmat, _ = cv2.Rodrigues(rvec)
                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                        pitch = angles[0] * 360
                        yaw = angles[1] * 360
                        roll = angles[2] * 360
                        
                        # Axis length (Kattalashtirildi)
                        axis_len = 200 
                        axis_pts = np.float32([[0,0,0], [0,0,axis_len], [0,axis_len,0], [axis_len,0,0]]).reshape(-1,3)
                        
                        # Camera matrix (must match what was used in solvePnP)
                        h, w, _ = frame.shape
                        focal_length = 1 * w
                        cam_matrix = np.array([[focal_length, 0, h / 2], [0, focal_length, w / 2], [0, 0, 1]])
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)
                        
                        imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, cam_matrix, dist_matrix)
                        nose_pt = tuple(imgpts[0].ravel().astype(int))
                        
                        # Draw axes: Z=Blue (Forward), Y=Green (Down), X=Red (Right)
                        # Z-axis (Forward) - Eng muhimi, qalinroq
                        z_pt = tuple(imgpts[1].ravel().astype(int))
                        cv2.arrowedLine(frame, nose_pt, z_pt, (255,0,0), 4) 
                        
                        # Y-axis (Down)
                        y_pt = tuple(imgpts[2].ravel().astype(int))
                        cv2.line(frame, nose_pt, y_pt, (0,255,0), 2) 
                        
                        # X-axis (Right)
                        x_pt = tuple(imgpts[3].ravel().astype(int))
                        cv2.line(frame, nose_pt, x_pt, (0,0,255), 2) 
                        
                        # Display Angles (Graduslarni ko'rsatish)
                        angle_text = f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}"
                        cv2.putText(frame, angle_text, (nose_pt[0] + 10, nose_pt[1] - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                   
                    except Exception as e:
                        pass

            label = f"Name: {name}"
            if is_blocked:
                label += " - BLOCKED!"  # Bloklangan bo'lsa yozuv chiqarish
            else:
                if phone_detected:
                    label += f" - Phone ({phone_confidence:.1f}%)"  # Telefon aniqlandi deb yozish
                if is_looking_backward:
                    label += " - LOOKING BACKWARD!"  # Orqaga qarash haqida yozish
                
                # Qoidabuzarliklar sonini ko'rsatish
                if name in face_system.violations and name != "Unknown":
                    backward_count = face_system.violations[name]["backward_look"]
                    if backward_count > 0:
                        label += f" | Backward: {backward_count}/5"

            cv2.putText(frame, label, (left, top - 10),  # Yuzning ustiga yozuv qo'yish
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Rasm saqlash logikasi (faqat aniqlangan paytda)
        if (phone_detected or any(backward_looking_status)):
             current_time = time()
             if current_time - last_save_time > 120:
                 # Kim qoidabuzarlik qilganini topish (birinchi uchraganini olamiz)
                 violator_name = "Unknown"
                 if face_names:
                     violator_name = face_names[0]
                 
                 face_system.save_cheat_frame(frame, violator_name)
                 last_save_time = current_time
                 print(f"[INFO] Cheat image saved. Next save in 2 minutes.")

        # --- DIAGNOSTIKA MA'LUMOTLARI ---
        fps_counter += 1
        if time() - fps_start_time > 1.0:
            current_fps = fps_counter
            source_fps = source_fps_counter # Source FPS ni yangilash
            
            fps_counter = 0
            source_fps_counter = 0
            fps_start_time = time()
        
        # Ekranga diagnostika yozish
        cv2.putText(frame, f"App FPS: {current_fps} | Cam FPS: {source_fps}", (10, target_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Muammoni aniqlash
        status_text = "Status: OK"
        status_color = (0, 255, 0)
        
        if source_fps < 5:
            status_text = "Status: LOW SOURCE FPS (Internet/Camera Lag)"
            status_color = (0, 0, 255)
        elif current_fps < 10:
            status_text = "Status: LOW APP FPS (System Lag)"
            status_color = (0, 0, 255)
        elif proc_time > 0.2: # Agar qayta ishlash 200ms dan ko'p vaqt olsa
            status_text = "Status: CPU Bottleneck (Processing Slow)"
            status_color = (0, 165, 255)
            
        cv2.putText(frame, status_text, (10, target_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        # --------------------------------

        cv2.imshow(window_name, frame)  # Kadrni ko'rsatish

        # X tugmasi bosilganda chiqish
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            shared_data["running"] = False
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 'q' tugmasi bosilganda chiqish
            shared_data["running"] = False
            break
        elif key == ord('t'):  # 't' tugmasi bosilganda o'qitish rejimiga o'tish
            training_mode = not training_mode
            if training_mode:
                current_name = input("Enter name for new face: ")  # Yangi ismni olish
        elif key == ord('a') and training_mode:  # 'a' tugmasi bosilganda yuzni qo'shish
            success, message = face_system.add_face(frame, current_name)  # Yuzni qo'shish
            print(message)
            training_mode = False  # O'qitish rejimini tugatish
        elif key == ord('r'):  # 'r' tugmasi bosilganda bloklangan foydalanuvchilarni ko'rsatish
            print("[BLOCKED USERS]:", face_system.blocked_users)
        elif key == ord('c'):  # 'c' tugmasi bosilganda barcha qoidabuzarliklarni tozalash
            face_system.violations = {}
            face_system.blocked_users = []
            face_system.save_violations_database()
            print("[INFO] All violations and blocks cleared")
        elif key == ord('d'): # 'd' tugmasi bosilganda debug rejimini o'zgartirish
            debug_mode = not debug_mode
            print(f"[INFO] Debug mode: {'ON' if debug_mode else 'OFF'}")

    video_stream.stop()  # Kamerani yopish
    cv2.destroyAllWindows()  # Barcha oynalarni yopish

if __name__ == "__main__":
    import sys
    import traceback
    
    # Windows konsolida UTF-8 muammolarini oldini olish
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass

    try:
        main()  # Dastur boshlanishi
    except BaseException:
        # Xatolikni faylga yozish (konsol muammosi bo'lsa)
        with open("error_log.txt", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        
        # Konsolga chiqarishga urinish
        print("\n[CRITICAL ERROR] Dasturda xatolik yuz berdi!")
        print("Xatolik tafsilotlari 'error_log.txt' fayliga yozildi.")
        try:
            traceback.print_exc()
        except:
            print("Konsolga xatolikni chiqarib bo'lmadi.")
        input("Chiqish uchun Enter bosing...")
