import requests
import time

def check_url_access(url):
    print(f"\nTekshirilayotgan URL: {url[:50]}...")
    
    # 1. Oddiy so'rov (Python sifatida)
    try:
        print("\n1. Oddiy so'rov yuborilmoqda...")
        r = requests.get(url, stream=True, timeout=5)
        print(f"Status Code: {r.status_code}")
        if r.status_code == 200:
            print("Natija: Muvaffaqiyatli! (Bloklanmagan)")
        elif r.status_code == 403:
            print("Natija: 403 Forbidden (Sayt sizni blokladi)")
        else:
            print(f"Natija: {r.status_code}")
        r.close()
    except Exception as e:
        print(f"Xatolik: {e}")

    # 2. Browser sifatida so'rov (User-Agent bilan)
    try:
        print("\n2. Browser (Chrome) sifatida so'rov yuborilmoqda...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.earthcam.com/"
        }
        r = requests.get(url, headers=headers, stream=True, timeout=5)
        print(f"Status Code: {r.status_code}")
        if r.status_code == 200:
            print("Natija: Muvaffaqiyatli! (Demak, User-Agent kerak)")
        else:
            print(f"Natija: {r.status_code}")
        r.close()
    except Exception as e:
        print(f"Xatolik: {e}")

if __name__ == "__main__":
    url = input("Linkni kiriting: ").strip()
    if url:
        check_url_access(url)
