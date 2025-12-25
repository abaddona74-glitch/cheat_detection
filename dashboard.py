import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import mediapipe as mp
import numpy as np
import time

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

        self.btn_database = ttk.Button(btn_frame, text="View Faces", command=self.open_database_window)
        self.btn_database.pack(fill="x", pady=5)

        self.btn_violations = ttk.Button(btn_frame, text="View Violations", command=self.open_violations_window)
        self.btn_violations.pack(fill="x", pady=5)

        self.btn_start = ttk.Button(btn_frame, text="START DETECTION", command=self.start_detection)
        self.btn_start.pack(fill="x", pady=20)
        
        # Style for Start button
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Helvetica", 12, "bold"))
        self.btn_start.configure(style="Accent.TButton")

    def start_detection(self):
        # Reset violations for a new session
        self.face_system.reset_session()
        
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
            
        def delete_face():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("Warning", "Please select a face to delete.")
                return
            
            index = selection[0]
            name = self.face_system.known_face_names[index]
            
            if messagebox.askyesno("Confirm", f"Are you sure you want to delete '{name}'?"):
                # Remove from lists
                del self.face_system.known_face_names[index]
                del self.face_system.known_face_encodings[index]
                
                # Save changes
                self.face_system.save_face_database()
                
                # Update UI
                listbox.delete(index)
                messagebox.showinfo("Success", f"Deleted '{name}' from database.")

        btn_frame = ttk.Frame(db_window)
        btn_frame.pack(pady=5)
        
        ttk.Button(btn_frame, text="Delete Selected", command=delete_face).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Close", command=db_window.destroy).pack(side="left", padx=5)

    def open_violations_window(self):
        v_window = tk.Toplevel(self.root)
        v_window.title("Violations Database")
        v_window.geometry("600x400")
        
        # Treeview for table
        columns = ("name", "phone", "backward", "blocked")
        tree = ttk.Treeview(v_window, columns=columns, show="headings")
        
        tree.heading("name", text="Name")
        tree.heading("phone", text="Phone Detections")
        tree.heading("backward", text="Looking Backward")
        tree.heading("blocked", text="Is Blocked?")
        
        tree.column("name", width=150)
        tree.column("phone", width=100)
        tree.column("backward", width=120)
        tree.column("blocked", width=80)
        
        tree.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(v_window, orient="vertical", command=tree.yview)
        scrollbar.place(relx=1.0, rely=0.0, relheight=1.0, anchor="ne")
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Populate data
        for name, data in self.face_system.violations.items():
            phone_count = data.get("phone", 0)
            backward_count = data.get("backward_look", 0)
            is_blocked = "YES" if name in self.face_system.blocked_users else "NO"
            
            tree.insert("", "end", values=(name, phone_count, backward_count, is_blocked))
            
        ttk.Button(v_window, text="Close", command=v_window.destroy).pack(pady=10)

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
                self.capture_step += 1
                self.captured_images += 1
                
                if self.capture_step > 2:
                    messagebox.showinfo("Success", f"Face registered successfully! (3 angles)")
                    reg_window.destroy()
                else:
                    # Keyingi bosqichga o'tish
                    pass
            else:
                messagebox.showerror("Error", message)

        ttk.Button(reg_window, text="Capture & Save", command=capture_face).pack(pady=10)

        # Drawing utils for Face Mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh

        # Auto-capture variables
        self.auto_capture_start_time = None
        self.countdown_active = False
        self.capture_step = 0 # 0: Front, 1: Left, 2: Right
        self.captured_images = 0

        # Video Loop for Preview
        def update_preview():
            if not reg_window.winfo_exists():
                return
                
            frame, _ = self.video_stream.read()
            if frame is not None:
                # Resize for UI
                frame = cv2.resize(frame, (500, 375))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Face Mesh Processing
                results = self.face_system.face_mesh.process(rgb_frame)
                
                status_text = "No face detected"
                status_color = "red"
                is_perfect = False
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Draw the mesh (tesselation) - "Cells"
                        # Color: #9aeffa -> BGR(250, 239, 154)
                        landmark_spec = mp_drawing.DrawingSpec(color=(250, 239, 154), thickness=1, circle_radius=1)
                        # Color: #9cdeff -> BGR(255, 222, 156)
                        connection_spec = mp_drawing.DrawingSpec(color=(255, 222, 156), thickness=1, circle_radius=1)

                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=landmark_spec,
                            connection_drawing_spec=connection_spec)
                        
                        # Draw contours (eyes, lips, face oval)
                        # Yondan qaraganda "Face Oval" chizig'i noto'g'ri ko'rinishi mumkin, shuning uchun uni o'chirib turamiz
                        # Faqat lablarni chizamiz (ko'zlarni olib tashladik)
                        for connection_type in [mp_face_mesh.FACEMESH_LIPS]:
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=face_landmarks,
                                connections=connection_type,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                        # Calculate face size for distance guidance
                        h, w, _ = frame.shape
                        x_min = min([lm.x for lm in face_landmarks.landmark])
                        x_max = max([lm.x for lm in face_landmarks.landmark])
                        face_width = x_max - x_min
                        
                        # Calculate Yaw (Rotation)
                        # 3D model points logic (simplified for dashboard)
                        face_3d = []
                        face_2d = []
                        img_h, img_w, img_c = frame.shape
                        
                        # Key landmarks for pose
                        key_points = [1, 33, 263, 61, 291, 199]
                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx in key_points:
                                x, y = int(lm.x * img_w), int(lm.y * img_h)
                                face_2d.append([x, y])
                                face_3d.append([x, y, lm.z])
                        
                        face_2d = np.array(face_2d, dtype=np.float64)
                        face_3d = np.array(face_3d, dtype=np.float64)

                        focal_length = 1 * img_w
                        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                             [0, focal_length, img_w / 2],
                                             [0, 0, 1]])
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)

                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                        rmat, jac = cv2.Rodrigues(rot_vec)
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                        
                        x_angle = angles[0] * 360
                        y_angle = angles[1] * 360 # Yaw
                        
                        # Step logic
                        if self.capture_step == 0: # Front
                            if face_width < 0.3:
                                status_text = "Too far! Come closer."
                                status_color = "#FFAA00"
                            elif face_width > 0.7:
                                status_text = "Too close! Move back."
                                status_color = "#FFAA00"
                            elif abs(y_angle) > 10:
                                status_text = "Look Straight Ahead"
                                status_color = "#FFAA00"
                            else:
                                status_text = "Perfect! Hold still (Front)."
                                status_color = "green"
                                is_perfect = True
                        elif self.capture_step == 1: # Left
                            if y_angle < 15: # Mirror effect: Looking Left is positive Yaw? No, let's check.
                                # Usually +Yaw is Right, -Yaw is Left. But with Mirror...
                                # Let's just say "Turn Head Left" and check for significant angle
                                status_text = "Turn Head RIGHT ->"
                                status_color = "#00AAFF"
                            else:
                                status_text = "Hold still (Right Side)."
                                status_color = "green"
                                is_perfect = True
                        elif self.capture_step == 2: # Right
                            if y_angle > -15:
                                status_text = "Turn Head LEFT <-"
                                status_color = "#00AAFF"
                            else:
                                status_text = "Hold still (Left Side)."
                                status_color = "green"
                                is_perfect = True

                # Auto-capture Logic
                name = name_entry.get().strip()
                if is_perfect and name:
                    if not self.countdown_active:
                        self.countdown_active = True
                        self.auto_capture_start_time = time.time()
                    
                    elapsed = time.time() - self.auto_capture_start_time
                    countdown = 3 - int(elapsed)
                    
                    if countdown > 0:
                        # Draw countdown
                        cv2.putText(frame, str(countdown), (220, 180), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 5)
                        # status_text = f"Capturing in {countdown}..." # Keep the instruction text
                    else:
                        # Capture!
                        self.countdown_active = False
                        self.auto_capture_start_time = None
                        
                        fresh_frame, _ = self.video_stream.read()
                        if fresh_frame is not None:
                             # Mirror the fresh frame too if needed, but video_stream already does it?
                             # Yes, video_stream.read() returns mirrored frame now.
                             
                             success, message = self.face_system.add_face(fresh_frame, name)
                             if success:
                                 self.capture_step += 1
                                 self.captured_images += 1
                                 
                                 if self.capture_step > 2:
                                     messagebox.showinfo("Success", f"Face registered successfully! (3 angles)")
                                     reg_window.destroy()
                                     return
                                 else:
                                     # Flash effect or sound could go here
                                     pass
                             else:
                                 messagebox.showerror("Error", message)
                                 self.countdown_active = False # Reset
                else:
                    self.countdown_active = False
                    self.auto_capture_start_time = None

                # Progress Indicator
                cv2.putText(frame, f"Step: {self.capture_step + 1}/3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Update status label
                status_label.config(text=status_text, foreground=status_color)

                # Convert to RGB for Tkinter
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
