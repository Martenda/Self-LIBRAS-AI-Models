import numpy as np
import cv2

def extract_landmarks(image, mp_hands, draw_landmarks_on_camera, mp_solutions):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hand_results = mp_hands.process(image_rgb)

    if hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0]  # Only take the first hand found on image

        # Drawing Hand Landmarks according to configuration
        if draw_landmarks_on_camera:
            mp_solutions.drawing_utils.draw_landmarks(
                image, hand_landmarks, mp_solutions.hands.HAND_CONNECTIONS)

        return np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark])

    return None
