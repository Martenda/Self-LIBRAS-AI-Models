import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from track_and_extract_landmarks import extract_landmarks
import global_params

def capture_camera():
    # Initialize video capture from the camera
    cap = cv2.VideoCapture(0)

    # Load the trained model
    model = load_model(global_params.MODEL_PATH)

    # Initialize MediaPipe Hand solution
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Extract hand landmarks
        hand_landmarks = extract_landmarks(image, hands)

        # If landmarks are found, reshape and make prediction
        if hand_landmarks is not None:
            hand_landmarks = hand_landmarks.reshape(1, *global_params.INPUT_SHAPE)
            prediction = model.predict(hand_landmarks)
            print(f'Predicted Sign: {prediction}')

        # Show the image on screen
        cv2.imshow('Live Camera', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
