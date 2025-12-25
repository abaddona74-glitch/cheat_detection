import sys
import os

try:
    import mediapipe
    print(f"Mediapipe imported successfully.")
    print(f"File: {mediapipe.__file__}")
    print(f"Dir: {dir(mediapipe)}")
    
    if hasattr(mediapipe, 'solutions'):
        print("mediapipe.solutions found.")
    else:
        print("ERROR: mediapipe.solutions NOT found.")
        
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
