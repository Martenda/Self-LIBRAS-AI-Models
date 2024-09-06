import mediapipe as mp
import numpy as np
import cv2

def extract_landmarks(image, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(image_rgb)

    if hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0]  # Only take the first hand
        return np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark])
    
    return None
