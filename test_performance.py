import cv2
import threading
import time
import numpy as np

# --- OPTIMIZED VIDEO STREAM CLASS ---
class VideoStream:
    def __init__(self, src):
        print(f"[INFO] Ulanmoqda: {src}")
        # FFMPEG backendini majburlash
        self.stream = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        
        # Hardware Acceleration (Agar mavjud bo'lsa)
        try:
            self.stream.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            print(f"[INFO] HW Acceleration: {self.stream.get(cv2.CAP_PROP_HW_ACCELERATION)}")
        except: pass

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stopped = True
                break
            
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            
            # CPU ni bo'shatish (muhim)
            time.sleep(0.005)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- MAIN TEST ---
def main():
    # Ishonchli test link (Big Buck Bunny)
    TEST_URL = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"
    
    print("------------------------------------------------")
    print("PERFORMANCE TEST (Multithreading + HW Accel)")
    print("------------------------------------------------")
    
    vs = VideoStream(TEST_URL).start()
    time.sleep(2.0) # Buferlash uchun vaqt

    if vs.stopped:
        print("[XATO] Test videoni ochib bo'lmadi. Internetni tekshiring.")
        return

    fps_start = time.time()
    frames = 0
    
    print("[INFO] Test boshlandi. 'q' tugmasini bosib chiqing.")

    while True:
        frame = vs.read()
        
        if frame is None:
            continue

        # Ekranga chiqarish
        frames += 1
        if time.time() - fps_start > 1.0:
            fps = frames / (time.time() - fps_start)
            print(f"[STATUS] FPS: {fps:.2f}")
            frames = 0
            fps_start = time.time()

        cv2.putText(frame, "TEST MODE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Performance Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
