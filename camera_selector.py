import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import json
import os

CONFIG_FILE = "camera_config.json"

def list_available_cameras(max_cameras=3):
    """Mavjud kameralarni aniqlash."""
    available_cameras = []
    
    # 0-indexli kamerani tekshirish
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        available_cameras.append("Camera 0")
        cap.release()
    else:
        available_cameras.append("Camera not found")
    
    for i in range(1, max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(f"Camera {i}")
            cap.release()
    return available_cameras

def load_config():
    """Sozlamalarni yuklash."""
    default_config = {
        "source_type": "index", 
        "source_value": "0", 
        "ip_url": "http://192.168.1.70:8080/video",
        "file_path": ""
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                # Eski config fayllar bilan muammo bo'lmasligi uchun default qiymatlarni tekshiramiz
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except:
            return default_config
    return default_config

def save_config(config):
    """Sozlamalarni saqlash."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

def select_camera():
    """Kamera tanlash oynasini ko'rsatish va tanlangan manbani qaytarish."""
    root = tk.Tk()
    root.title("Kamera Sozlamalari")
    root.geometry("400x400") # Oyna o'lchamini kattalashtirdik
    
    # Oynani ekran markaziga joylashtirish
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 400) // 2
    y = (screen_height - 400) // 2
    root.geometry(f"400x400+{x}+{y}")

    config = load_config()
    result = {"source": None, "resolution": "1024x768 (Default)", "fps_limit": "Cheklovsiz"}

    # O'zgaruvchilar
    selected_option = tk.StringVar(value="camera")
    if config["source_type"] == "ip":
        selected_option.set("ip")
    elif config["source_type"] == "file":
        selected_option.set("file")
    
    camera_list = list_available_cameras()
    selected_camera = tk.StringVar()
    
    # Agar oldingi tanlangan kamera ro'yxatda bo'lsa, o'shani tanlaymiz
    prev_cam = f"Camera {config['source_value']}"
    if prev_cam in camera_list:
        selected_camera.set(prev_cam)
    else:
        selected_camera.set(camera_list[0])

    ip_url_var = tk.StringVar(value=config["ip_url"])
    file_path_var = tk.StringVar(value=config.get("file_path", ""))

    # UI Elementlari
    tk.Label(root, text="Kamera manbasini tanlang:", font=("Arial", 12, "bold")).pack(pady=10)

    # Radio tugmalar uchun frame
    frame = tk.Frame(root)
    frame.pack(pady=5, padx=20, fill="x")

    # Oddiy kamera opsiyasi
    rb_camera = tk.Radiobutton(frame, text="Oddiy Kamera", variable=selected_option, value="camera", command=lambda: toggle_inputs())
    rb_camera.grid(row=0, column=0, sticky="w")

    camera_combo = ttk.Combobox(frame, textvariable=selected_camera, values=camera_list, state="readonly")
    camera_combo.grid(row=1, column=0, padx=20, sticky="ew")

    # IP kamera opsiyasi
    rb_ip = tk.Radiobutton(frame, text="IP Kamera", variable=selected_option, value="ip", command=lambda: toggle_inputs())
    rb_ip.grid(row=2, column=0, sticky="w", pady=(10, 0))

    ip_entry = tk.Entry(frame, textvariable=ip_url_var)
    ip_entry.grid(row=3, column=0, padx=20, sticky="ew")

    # Video fayl opsiyasi
    rb_file = tk.Radiobutton(frame, text="Video Fayl", variable=selected_option, value="file", command=lambda: toggle_inputs())
    rb_file.grid(row=4, column=0, sticky="w", pady=(10, 0))

    file_frame = tk.Frame(frame)
    file_frame.grid(row=5, column=0, padx=20, sticky="ew")
    
    file_entry = tk.Entry(file_frame, textvariable=file_path_var)
    file_entry.pack(side="left", fill="x", expand=True)
    
    def browse_file():
        filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")])
        if filename:
            file_path_var.set(filename)

    file_btn = tk.Button(file_frame, text="...", command=browse_file, width=3)
    file_btn.pack(side="right", padx=(5, 0))

    # --- Yangi qo'shilgan qism: Sozlamalar (FPS va Sifat) ---
    settings_frame = tk.LabelFrame(root, text="Qo'shimcha Sozlamalar", padx=10, pady=10)
    settings_frame.pack(pady=10, padx=20, fill="x")

    # Sifat (Resolution)
    tk.Label(settings_frame, text="Sifat (Resolution):").grid(row=0, column=0, sticky="w")
    resolutions = [
        "1024x768 (Default)", 
        "256x144 (144p)", 
        "426x240 (240p)", 
        "640x360 (360p)", 
        "640x480 (480p 4:3)",
        "854x480 (480p 16:9)", 
        "1280x720 (720p)", 
        "1920x1080 (1080p)"
    ]
    res_var = tk.StringVar(value=config.get("resolution", "1024x768 (Default)"))
    res_combo = ttk.Combobox(settings_frame, textvariable=res_var, values=resolutions, state="readonly", width=25)
    res_combo.grid(row=0, column=1, padx=10, sticky="w")

    # FPS Limit
    tk.Label(settings_frame, text="FPS Limit:").grid(row=1, column=0, sticky="w", pady=5)
    fps_options = ["Cheklovsiz", "30 FPS", "15 FPS", "5 FPS"]
    fps_var = tk.StringVar(value=config.get("fps_limit", "Cheklovsiz"))
    fps_combo = ttk.Combobox(settings_frame, textvariable=fps_var, values=fps_options, state="readonly", width=20)
    fps_combo.grid(row=1, column=1, padx=10, sticky="w")
    # ---------------------------------------------------------

    def toggle_inputs():
        opt = selected_option.get()
        if opt == "camera":
            camera_combo.config(state="readonly")
            ip_entry.config(state="disabled")
            file_entry.config(state="disabled")
            file_btn.config(state="disabled")
        elif opt == "ip":
            camera_combo.config(state="disabled")
            ip_entry.config(state="normal")
            file_entry.config(state="disabled")
            file_btn.config(state="disabled")
        else: # file
            camera_combo.config(state="disabled")
            ip_entry.config(state="disabled")
            file_entry.config(state="normal")
            file_btn.config(state="normal")

    toggle_inputs()

    def on_submit():
        opt = selected_option.get()
        
        if opt == "camera":
            cam_str = selected_camera.get() # "Camera 0" or "Camera not found"
            if "not found" in cam_str:
                messagebox.showerror("Xatolik", "Kamera topilmadi!")
                return
            cam_idx = int(cam_str.split(" ")[1])
            result["source"] = cam_idx
            
            # Config yangilash
            config["source_type"] = "index"
            config["source_value"] = str(cam_idx)
            
        elif opt == "ip":
            url = ip_url_var.get().strip()
            if not url:
                messagebox.showerror("Xatolik", "IP manzil kiritilmadi!")
                return
            result["source"] = url
            
            # Config yangilash
            config["source_type"] = "ip"
            config["ip_url"] = url
            
        elif opt == "file":
            path = file_path_var.get().strip()
            if not path or not os.path.exists(path):
                messagebox.showerror("Xatolik", "Fayl tanlanmadi yoki mavjud emas!")
                return
            result["source"] = path
            
            # Config yangilash
            config["source_type"] = "file"
            config["file_path"] = path
        
        # Yangi sozlamalarni saqlash
        config["resolution"] = res_var.get()
        config["fps_limit"] = fps_var.get()

        result["resolution"] = config["resolution"]
        result["fps_limit"] = config["fps_limit"]
        
        save_config(config)
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=20)
    
    tk.Button(btn_frame, text="Boshlash", command=on_submit, bg="#4CAF50", fg="white", width=15).pack()

    root.mainloop()
    
    return result

if __name__ == "__main__":
    res = select_camera()
    print(f"Tanlangan sozlamalar: {res}")
