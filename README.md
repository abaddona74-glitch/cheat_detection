# 🚨 Violation Detection System using Face Recognition

Bu loyiha real vaqt rejimida yuzni aniqlash va qoidabuzarliklarni (cheating) qayd qilish uchun mo‘ljallangan. Ma’lumotlar `.pkl` faylida saqlanadi va vizual tarzda ko‘rsatish uchun `pandasgui` interfeysi orqali ishlatiladi.

## 🧰 Texnologiyalar

Loyihada quyidagi asosiy kutubxonalar ishlatilgan:

- `opencv-python` – kamera va video oqimini boshqarish uchun
- `numpy` – massivlar bilan ishlash
- `tensorflow` – (agar model ishlatilgan bo‘lsa)
- `face_recognition` – yuzni aniqlash va tanib olish
- `mediapipe` – yuz nuqtalarini (landmarks) aniqlash
- `pandas` – ma’lumotlarni saqlash va qayta ishlash
- `pandasgui` – `.pkl` fayldagi DataFrame'ni GUI ko‘rinishida ko‘rsatish
- `pickle-mixin` – `.pkl` fayllarni o‘qish va saqlash uchun

## 🔧 O‘rnatish

Loyihani lokal kompyuterda ishga tushirish uchun quyidagilarni bajaring:

```bash
# Virtual muhit yaratish
python -m venv venv
# Virtual muhitni aktivlashtirish (Windows)
venv\Scripts\activate
# Kutubxonalarni o‘rnatish
pip install -r requirements.txt
# Asosiy dastur va vizual panelni ishga tushirish
python main.py
python open.py

⌨️ Klaviatura tugmalari (Hotkeys)
Tugma	Funksiya
q	Dasturni to‘xtatish va chiqish (kamera yopiladi)
t	O‘qitish (training) rejimini yoqish/o‘chirish
a	O‘qitish rejimida: kadrdan yangi yuzni bazaga qo‘shish
r	Bloklangan foydalanuvchilar ro‘yxatini ko‘rsatish (konsolda)
c	Barcha qoidabuzarlik va bloklanganlar ma’lumotlarini tozalash