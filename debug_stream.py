import cv2
import os
import sys

# 1. Debug loglarini yoqish (batafsil ma'lumot olish uchun)
os.environ["OPENCV_VIDEOIO_DEBUG"] = "1"
os.environ["OPENCV_FFMPEG_DEBUG"] = "1"
os.environ["OPENCV_LOG_LEVEL"] = "VERBOSE"

def test_stream(url):
    print(f"\n--- TEST BOSHLANDI ---")
    print(f"OpenCV versiyasi: {cv2.__version__}")
    
    # FFmpeg borligini tekshirish
    build_info = cv2.getBuildInformation()
    if "FFMPEG:                      YES" in build_info:
        print("FFmpeg holati: MAVJUD (OK)")
    else:
        print("FFmpeg holati: MAVJUD EMAS (Bu muammo bo'lishi mumkin)")

    print(f"Sinayotgan URL: {url}")
    
    # 2. Videoni ochishga urinish
    # CAP_FFMPEG ni majburlab ishlatamiz
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("\n[XATOLIK] Oqim ochilmadi!")
        print("Mumkin bo'lgan sabablar:")
        print("1. Link eskirgan (token muddati tugagan).")
        print("2. Sayt Python-ni bloklayapti (User-Agent kerak).")
        print("3. Internet yoki VPN muammosi.")
    else:
        print("\n[MUVAFFAQIYAT] Oqim ochildi!")
        
        # Ma'lumotlarni o'qish
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video o'lchami: {width}x{height}")
        print(f"FPS: {fps}")
        
        # Birinchi kadrni o'qish
        ret, frame = cap.read()
        if ret:
            print("Birinchi kadr muvaffaqiyatli o'qildi.")
            cv2.imshow("Test Stream", frame)
            print("Kadrni ko'rish uchun istalgan tugmani bosing...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("[XATOLIK] Oqim ochildi, lekin kadr o'qib bo'lmadi.")
    
    cap.release()
    print("--- TEST TUGADI ---")

if __name__ == "__main__":
    # Bu yerga o'sha muammoli linkni qo'ying
    url = input("Linkni kiriting: ").strip()
    if url:
        test_stream(url)
