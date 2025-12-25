import cv2

def check_codec(source):
    print(f"Tekshirilmoqda: {source}")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("Ochib bo'lmadi!")
        return

    # Codec ma'lumotlarini olish
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_name = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    print(f"FOURCC Code: {fourcc}")
    print(f"Codec Name: {codec_name}")
    
    # Bir nechta kadr o'qib ko'ramiz
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"Kadr {i+1}: OK ({frame.shape})")
        else:
            print(f"Kadr {i+1}: Fail")
            
    cap.release()

if __name__ == "__main__":
    url = input("Linkni kiriting: ").strip()
    if url:
        check_codec(url)
