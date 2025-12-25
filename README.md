# 🚨 Imtihon Nazorati Tizimi (Cheat Detection System)

Ushbu loyiha imtihon jarayonlarida qoidabuzarliklarni (cheating) avtomatik aniqlash uchun mo‘ljallangan sun'iy intellekt tizimidir. Tizim kompyuter ko'rish (Computer Vision) texnologiyalaridan foydalanib, real vaqt rejimida talabalarni kuzatadi va shubhali holatlarni qayd etadi.

## ✨ Asosiy Imkoniyatlar

*   **👤 Yuzni Tanib Olish (Face Recognition):** Talabalarni yuzidan taniydi va ro'yxatga oladi.
*   **📱 Telefonni Aniqlash (Phone Detection):** Kadrdan mobil telefon aniqlansa, qoidabuzarlik sifatida belgilaydi.
*   **👀 Nigoh va Bosh Harakatini Kuzatish (Head Pose Estimation):** Talaba orqaga yoki yon tomonga qarasa, ogohlantirish beradi.
*   **🚫 Avtomatik Bloklash:** Qoidabuzarliklar soni belgilangan limitdan oshsa, foydalanuvchi "bloklanganlar" ro'yxatiga tushadi.
*   **📊 Hisobot va Tahlil:** Barcha qoidabuzarliklar ma'lumotlar bazasida saqlanadi va ularni vizual ko'rish imkoniyati mavjud.

## 🛠 Talablar

Loyihani ishga tushirish uchun quyidagi dasturlar o'rnatilgan bo'lishi kerak:

*   Python 3.8+
*   Visual Studio Build Tools (Windows uchun `dlib` kutubxonasini o'rnatishda kerak bo'lishi mumkin)

## 🚀 O‘rnatish

1.  **Loyihani yuklab oling:**
    ```bash
    git clone <repository-url>
    cd cheat_detection
    ```

2.  **Virtual muhit yarating (tavsiya etiladi):**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scriptsctivate
    # Linux/macOS:
    source venv/bin/activate
    ```

3.  **Kerakli kutubxonalarni o‘rnating:**
    ```bash
    pip install -r requirements.txt
    ```

## 🖥 Ishlatish

### 1. Tizimni ishga tushirish
Asosiy kuzatuv dasturini ishga tushirish uchun:
```bash
python main.py
```
Dastur ishga tushgach, kamera ochiladi va kuzatuv boshlanadi.

### 2. Boshqaruv tugmalari (Hotkeys)
Dastur ishlab turgan vaqtda quyidagi tugmalardan foydalanishingiz mumkin:

| Tugma | Vazifa |
| :--- | :--- |
| **`q`** | Dasturni to‘xtatish va chiqish. |
| **`t`** | **O‘qitish rejimi (Training Mode):** Yangi talabani bazaga qo'shish uchun. Ism kiritish so'raladi. |
| **`a`** | (O'qitish rejimida) Joriy kadrni yuz sifatida saqlash va bazaga qo'shish. |
| **`r`** | Bloklangan foydalanuvchilar ro‘yxatini konsolga chiqarish. |
| **`c`** | Barcha qoidabuzarliklar va bloklanganlar tarixini tozalash (Reset). |

### 3. Natijalarni ko'rish
Yig'ilgan ma'lumotlar va qoidabuzarliklar tarixini ko'rish uchun:
```bash
python open.py
```
Bu buyruq `pandasgui` orqali ma'lumotlarni jadval ko'rinishida ochadi.

## 📂 Loyiha Tuzilishi

*   `main.py`: Asosiy dastur kodi (kamera, aniqlash, logika).
*   `open.py`: Saqlangan ma'lumotlarni (`.pkl`) ko'rish uchun yordamchi dastur.
*   `face_database.pkl`: Ro'yxatdan o'tgan yuzlar bazasi.
*   `violations_database.pkl`: Qoidabuzarliklar tarixi.
*   `requirements.txt`: Kerakli kutubxonalar ro'yxati.

## ⚠️ Eslatma
Agar `dlib` yoki `face_recognition` o'rnatishda xatolik yuz bersa, avval `cmake` ni o'rnating va C++ kompilyatorlari mavjudligini tekshiring.
