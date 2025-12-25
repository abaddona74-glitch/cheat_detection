import urllib.request
import ssl
import sys

# URL ni argument sifatida olish yoki hardcode qilish
url = "https://hd-auth.skylinewebcams.com/live.m3u8?a=ft64g1kujhlfgvef3mbini36u7"

if len(sys.argv) > 1:
    url = sys.argv[1]

print(f"Tekshirilmoqda: {url}")

try:
    # SSL xatolarini inkor qilish (ba'zan sertifikat muammosi bo'ladi)
    context = ssl._create_unverified_context()
    
    # So'rov yuborish (User-Agent va Referer qo'shamiz)
    req = urllib.request.Request(
        url, 
        data=None, 
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.skylinewebcams.com/',
            'Origin': 'https://www.skylinewebcams.com'
        }
    )
    
    with urllib.request.urlopen(req, context=context) as response:
        print(f"Status Code: {response.getcode()}")
        content = response.read().decode('utf-8')
        print("-" * 20)
        print("M3U8 FILE CONTENT:")
        print("-" * 20)
        print(content)
        print("-" * 20)
        
        # Agar ichida boshqa .m3u8 linklar bo'lsa, ularni ajratib ko'rsatamiz
        lines = content.split('\n')
        print("\n[INFO] Ichki oqimlar (Nested Streams):")
        for line in lines:
            if ".m3u8" in line:
                print(f" -> {line}")

except Exception as e:
    print(f"Xatolik yuz berdi: {e}")
