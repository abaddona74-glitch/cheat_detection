import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading

class Dashboard:
    def __init__(self, root, face_system, video_stream):
        self.root = root
        self.face_system = face_system
        self.video_stream = video_stream
        self.root.title("Cheat Detection - Dashboard")
        self.root.geometry("400x300")
        
        # Markazlashtirish
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - 400) // 2
        y = (screen_height - 300) // 2
        root.geometry(f"400x300+{x}+{y}")

        self.should_start = False

        # UI Elementlari
        title_label = ttk.Label(root, text="Control Panel", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=20)

        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=10, fill="x", padx=50)

        self.btn_register = ttk.Button(btn_frame, text="Register New Face", command=self.open_register_window)
        self.btn_register.pack(fill="x", pady=5)

        self.btn_database = ttk.Button(btn_frame, text="View Database", command=self.open_database_window)
        self.btn_database.pack(fill="x", pady=5)

        self.btn_start = ttk.Button(btn_frame, text="START DETECTION", command=self.start_detection)
        self.btn_start.pack(fill="x", pady=20)
        
        # Style for Start button
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Helvetica", 12, "bold"))
        self.btn_start.configure(style="Accent.TButton")

    def start_detection(self):
        self.should_start = True
        self.root.destroy()

    def open_database_window(self):
        db_window = tk.Toplevel(self.root)
        db_window.title("Registered Faces")
        db_window.geometry("300x400")
        
        listbox = tk.Listbox(db_window, font=("Helvetica", 12))
        listbox.pack(fill="both", expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(db_window, orient="vertical", command=listbox.yview)
        scrollbar.pack(side="right", fill="y")
        listbox.config(yscrollcommand=scrollbar.set)
        
        for name in self.face_system.known_face_names:
            listbox.insert("end", name)
            
        ttk.Button(db_window, text="Close", command=db_window.destroy).pack(pady=5)

    def open_register_window(self):
        reg_window = tk.Toplevel(self.root)
        reg_window.title("Register Face")
        reg_window.geometry("600x500")
        
        # Video Label
        video_label = ttk.Label(reg_window)
        video_label.pack(pady=10)
        
        # Input Frame
        input_frame = ttk.Frame(reg_window)
        input_frame.pack(pady=10)
        
        ttk.Label(input_frame, text="Name:").pack(side="left", padx=5)
        name_entry = ttk.Entry(input_frame)
        name_entry.pack(side="left", padx=5)
        
        status_label = ttk.Label(reg_window, text="Enter name and click Capture", foreground="blue")
        status_label.pack(pady=5)

        def capture_face():
            name = name_entry.get().strip()
            if not name:
                messagebox.showerror("Error", "Please enter a name!")
                return
            
            frame, _ = self.video_stream.read()
            if frame is None:
                messagebox.showerror("Error", "Could not read frame from camera!")
                return
                
            success, message = self.face_system.add_face(frame, name)
            if success:
                messagebox.showinfo("Success", f"Face registered for {name}!")
                reg_window.destroy()
            else:
                messagebox.showerror("Error", message)

        ttk.Button(reg_window, text="Capture & Save", command=capture_face).pack(pady=10)

        # Video Loop for Preview
        def update_preview():
            if not reg_window.winfo_exists():
                return
                
            frame, _ = self.video_stream.read()
            if frame is not None:
                # Resize for UI
                frame = cv2.resize(frame, (500, 375))
                # Convert to RGB
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                video_label.imgtk = imgtk
                video_label.configure(image=imgtk)
            
            reg_window.after(30, update_preview)

        update_preview()

def show_dashboard(face_system, video_stream):
    root = tk.Tk()
    app = Dashboard(root, face_system, video_stream)
    root.mainloop()
    return app.should_start
