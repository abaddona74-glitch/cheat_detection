import sys
import os

print(f"Python version: {sys.version}")

try:
    import mediapipe as mp
    print(f"Mediapipe imported successfully.")
    print(f"File: {mp.__file__}")
    
    # Versiyani tekshirish
    try:
        print(f"Version: {mp.__version__}")
    except:
        print("Version not found")
    
    # Explicit import urinib ko'rish
    try:
        import mediapipe.python.solutions as solutions
        print("Successfully imported mediapipe.python.solutions directly.")
        if not hasattr(mp, 'solutions'):
            print("Patching mp.solutions...")
            mp.solutions = solutions
    except ImportError as e:
        print(f"Failed to import mediapipe.python.solutions: {e}")

    if hasattr(mp, 'solutions'):
        print("mediapipe.solutions found.")
        try:
            face_mesh = mp.solutions.face_mesh
            print("mp.solutions.face_mesh found.")
        except AttributeError:
            print("mp.solutions.face_mesh NOT found.")
    else:
        print("ERROR: mediapipe.solutions NOT found even after attempted patch.")
        print(f"Dir(mp): {dir(mp)}")
        
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
