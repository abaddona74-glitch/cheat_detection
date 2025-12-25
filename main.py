import cv2
import numpy as np
import tensorflow as tf
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
        print("[INFO] Tizim ishga tushmoqda...")  # Tizim ishga tushayotganini bildirish
        self.known_face_encodings = []  # Yuzlarning kodlari
        self.known_face_names = []  # Yuzlar bilan bog'liq nomlar
        self.load_face_database()  # Ma'lumotlar bazasini yuklash
        
        # Qoidabuzarliklar va bloklanganlar bazasini yaratish
        self.violations = {}  # Qoidabuzarliklar soni
        self.blocked_users = []  # Bloklangan foydalanuvchilar ro'yxati
        self.load_violations_database()  # Qoidabuzarliklar ma'lumotlar bazasini yuklash

        # FaceMesh modelini yaratish
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 3D bosh pozitsiyasini aniqlash uchun model nuqtalari
        self.model_points = np.array([  # Model points for 3D head pose estimation
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1)
        ], dtype=np.float32)

        # Telefonni aniqlash uchun mobil model (MobileNetV2)
        self.phone_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            alpha=1.0,
            include_top=True,
            weights='imagenet',
            pooling='avg'
        )

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

    # Qoidabuzarliklarni qayd qilish
    def record_violation(self, name, violation_type):
        if name == "Notanish":
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
                self.log_cheat(name, f"BLOKLANGAN: {violation_type} sababi bilan")
                print(f"[BLOKLANDI] {name} - {violation_type} tufayli bloklandi")
                
        self.save_violations_database()
        
    # Foydalanuvchi bloklangan yoki yo'qligini tekshirish
    def is_blocked(self, name):
        return name in self.blocked_users

    # Yuzlarni noqonuniy (cheat) holatlarini loglash
    def log_cheat(self, name, reason):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Hozirgi vaqtni olish
        with open("log.txt", "a") as f:
            f.write(f"[{now}] Ism: {name} | Sabab: {reason}\n")

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
        print(f"[INFO] Qoidabuzarlik tasvirini saqlash: {filename}")

    # Telefonni aniqlash
    def detect_phone(self, frame):
        img = cv2.resize(frame, (224, 224))  # Rasmni o'lchamini o'zgartirish
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Rasmni RGB formatiga o'zgartirish
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  # Modelga moslashtirish
        img = np.expand_dims(img, axis=0)  # O'lchamni kengaytirish

        preds = self.phone_model.predict(img, verbose=0)  # Model orqali prognoz qilish
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds)[0]  # Prognozni dekodlash

        # Telefonlarga oid tasniflar
        phone_related = ['cell_phone', 'cellular_telephone', 'mobile_phone',
                         'hand-held_computer', 'iPod', 'smartphone']

        for _, label, conf in decoded:
            if any(phone in label for phone in phone_related) and conf > 0.15:
                return True, conf * 100  # Telefon aniqlandi
        return False, 0.0  # Telefon aniqlanmadi

    # Yuzni qo'shish
    def add_face(self, frame, name):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Rasmni RGB formatiga o'zgartirish
        face_locations = face_recognition.face_locations(rgb_frame)  # Yuzni topish

        if not face_locations:  # Agar yuz topilmasa
            return False, "Yuz topilmadi"

        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]  # Yuzning kodini olish
        self.known_face_encodings.append(face_encoding)  # Yuzni ma'lumotlar bazasiga qo'shish
        self.known_face_names.append(name)  # Ismni ma'lumotlar bazasiga qo'shish

        self.save_face_database()  # Ma'lumotlar bazasini saqlash
        return True, "Yuz muvaffaqiyatli qo'shildi"  # Muvaffaqiyatli qo'shildi

    # Yuzni aniqlash
    def identify_face(self, frame, face_location):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Rasmni RGB formatiga o'zgartirish
        face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])[0]  # Yuz kodini olish

        name = "Notanish"  # Yuzni aniqlash uchun default nom
        if self.known_face_encodings:  # Agar ma'lumotlar bazasida yuzlar bo'lsa
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)  # Yuzlar orasidagi masofa
            if len(distances) > 0:
                best_match = np.argmin(distances)  # Eng yaqin yuzni topish
                if distances[best_match] < 0.5:  # Agar masofa kichik bo'lsa
                    name = self.known_face_names[best_match]  # Nomni olish
        return name  # Yuzning nomini qaytarish

    # Orqaga qarayotgan yuzni aniqlash
    def detect_backward_looking(self, frame, face_location):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Rasmni RGB formatiga o'zgartirish
            height, width = frame.shape[:2]  # Rasm o'lchamlari
            results = self.face_mesh.process(rgb_frame)  # FaceMesh modelini ishlatish
            if not results.multi_face_landmarks:  # Agar yuzning nuqtalari mavjud bo'lmasa
                return False

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
                return abs(yaw) > 50  # Agar orqaga qarayotgan bo'lsa
        except Exception as e:
            print(f"[ERROR] Bosh pozitsiyasini aniqlashda xatolik: {str(e)}")  # Xatolikni chiqarish
            return False
        return False

import threading
import time as time_module # time modulini qayta nomlaymiz, chunki pastda time() funksiyasi bor

# ... (boshqa importlar)

# Asosiy dastur
def main():
    print("[INFO] Yuz tanib olish tizimi ishga tushmoqda...")
    
    # Kamera tanlash oynasini chaqirish
    from camera_selector import select_camera
    settings = select_camera()
    
    source = settings["source"]
    resolution_str = settings["resolution"]
    fps_limit_str = settings["fps_limit"]
    
    if source is None:
        print("[INFO] Dastur to'xtatildi (kamera tanlanmadi).")
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

    print(f"[INFO] Kamera ishga tushmoqda... Manba: {source}")
    print(f"[INFO] Sozlamalar: {target_width}x{target_height}, FPS Limit: {fps_limit_str}")

    # AMD RX580 yoki boshqa GPU lardan foydalanish uchun OpenCL ni yoqish
    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            print(f"[INFO] GPU Optimizatsiya (OpenCL): YOQILDI ({cv2.ocl.useOpenCL()})")
            print(f"[INFO] Qurilma: {cv2.ocl.Device.getDefault().name()}")
        else:
            print("[INFO] OpenCL mavjud emas, CPU ishlatilmoqda.")
    except Exception as e:
        print(f"[WARNING] OpenCL sozlashda xatolik: {e}")

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
                print(f"[INFO] Hardware Acceleration (Decode): YOQILDI (Mode: {self.stream.get(cv2.CAP_PROP_HW_ACCELERATION)})")
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
    print("[INFO] Video oqim alohida thread-da ishga tushirilmoqda...")
    video_stream = VideoStream(source, target_width, target_height).start()
    
    # Biroz kutamiz, kamera o'nglanib olsin
    time_module.sleep(1.0)

    if video_stream.stopped:
        print(f"[ERROR] Kamera/Stream ochilmadi: {source}")
        return
    else:
        print("[INFO] Kamera muvaffaqiyatli ishga tushdi.")

    training_mode = False  # O'qitish rejimi
    current_name = ""  # Joriy ism
    last_save_time = 0  # Oxirgi saqlangan vaqti (sekundda)
    
    # FPS nazorati uchun
    prev_frame_time = 0

    # --- MULTITHREADING SETUP ---
    # Asosiy oqim (UI) va Hisoblash oqimi (Processing) ni ajratamiz.
    # Bu FPS ni oshiradi, chunki UI hisob-kitob tugashini kutib o'tirmaydi.
    
    shared_data = {
        "frame": None,
        "results": {
            "face_locations": [],
            "face_names": [],
            "phone_detected": False,
            "phone_confidence": 0.0,
            "backward_looking_status": []
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
                    
                    # 2. Telefonni aniqlash (Asl kadrda, chunki u o'zi resize qiladi)
                    phone_detected, phone_confidence = face_system.detect_phone(frame_to_process)
                    
                    face_names = []
                    backward_looking_status = []
                    
                    for face_location in face_locations:
                        # 3. Yuzni tanish (Asl kadrda, lekin aniqlangan joylarda)
                        name = face_system.identify_face(frame_to_process, face_location)
                        face_names.append(name)
                        
                        # 4. Bosh holatini aniqlash
                        is_looking_backward = face_system.detect_backward_looking(frame_to_process, face_location)
                        backward_looking_status.append(is_looking_backward)
                        
                        # Qoidabuzarliklarni yozish
                        if phone_detected:
                            face_system.log_cheat(name, "Telefon aniqlandi")
                            face_system.record_violation(name, "phone")
                        elif is_looking_backward:
                            face_system.log_cheat(name, "Orqaga qarayapti")
                            face_system.record_violation(name, "backward_look")

                    # Natijalarni yangilash
                    with shared_data["lock"]:
                        shared_data["results"] = {
                            "face_locations": face_locations,
                            "face_names": face_names,
                            "phone_detected": phone_detected,
                            "phone_confidence": phone_confidence,
                            "backward_looking_status": backward_looking_status
                        }
                        shared_data["processing_time"] = time() - start_proc
                except Exception as e:
                    print(f"[ERROR] Thread xatolik: {e}")
            else:
                time_module.sleep(0.005) # CPU ni bo'shatish uchun kichik pauza

    # Threadni ishga tushirish
    process_thread = threading.Thread(target=processing_loop, args=(face_system, shared_data))
    process_thread.daemon = True
    process_thread.start()
    print("[INFO] Multithreading rejimi ishga tushdi (Asynchronous Processing)")

    # Oynani yaratish va sozlash (Fullscreen/Windowed)
    window_name = "Yuz tanib olish tizimi"
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
                print("Video oqim to'xtadi.")
                break
            
            # Kutish rejimini ko'rsatish
            blank = np.zeros((target_height, target_width, 3), np.uint8)
            cv2.putText(blank, "KUTILMOQDA...", (50, target_height//2), 
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

        for (top, right, bottom, left), name, is_looking_backward in zip(face_locations, face_names, backward_looking_status):
            
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

            label = f"Ism: {name}"
            if is_blocked:
                label += " - BLOKLANGAN!"  # Bloklangan bo'lsa yozuv chiqarish
            else:
                if phone_detected:
                    label += f" - Telefon ({phone_confidence:.1f}%)"  # Telefon aniqlandi deb yozish
                if is_looking_backward:
                    label += " - ORQAGA QARAYAPTI!"  # Orqaga qarash haqida yozish
                
                # Qoidabuzarliklar sonini ko'rsatish
                if name in face_system.violations and name != "Notanish":
                    backward_count = face_system.violations[name]["backward_look"]
                    if backward_count > 0:
                        label += f" | Orqaga: {backward_count}/5"

            cv2.putText(frame, label, (left, top - 10),  # Yuzning ustiga yozuv qo'yish
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Rasm saqlash logikasi (faqat aniqlangan paytda)
        if (phone_detected or any(backward_looking_status)):
             current_time = time()
             if current_time - last_save_time > 120:
                 # Kim qoidabuzarlik qilganini topish (birinchi uchraganini olamiz)
                 violator_name = "Noma'lum"
                 if face_names:
                     violator_name = face_names[0]
                 
                 face_system.save_cheat_frame(frame, violator_name)
                 last_save_time = current_time
                 print(f"[INFO] Cheating rasmi saqlandi. Keyingi saqlash 2 daqiqadan so'ng bo'ladi.")

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
                current_name = input("Yangi yuz uchun ism kiriting: ")  # Yangi ismni olish
        elif key == ord('a') and training_mode:  # 'a' tugmasi bosilganda yuzni qo'shish
            success, message = face_system.add_face(frame, current_name)  # Yuzni qo'shish
            print(message)
            training_mode = False  # O'qitish rejimini tugatish
        elif key == ord('r'):  # 'r' tugmasi bosilganda bloklangan foydalanuvchilarni ko'rsatish
            print("[BLOKLANGAN FOYDALANUVCHILAR]:", face_system.blocked_users)
        elif key == ord('c'):  # 'c' tugmasi bosilganda barcha qoidabuzarliklarni tozalash
            face_system.violations = {}
            face_system.blocked_users = []
            face_system.save_violations_database()
            print("[INFO] Barcha qoidabuzarliklar va bloklashlar tozalandi")

    video_stream.stop()  # Kamerani yopish
    cv2.destroyAllWindows()  # Barcha oynalarni yopish

if __name__ == "__main__":
    main()  # Dastur boshlanishi
