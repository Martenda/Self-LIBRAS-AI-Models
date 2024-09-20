import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from track_and_extract_landmarks import extract_landmarks
import global_params
import numpy as np  # Import numpy to load class labels

def capture_camera():
    # Initialize video capture from the camera
    cap = cv2.VideoCapture(0)

    # Load the trained model
    model = load_model(global_params.SAVED_MODEL_PATH)

    # Load class labels
    class_labels = np.load('class_labels.npy')

    # Initialize MediaPipe Hand solution
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        # Extract hand landmarks
        hand_landmarks = extract_landmarks(image, hands)

        # If landmarks are found, reshape and make prediction
        if hand_landmarks is not None:
            hand_landmarks = hand_landmarks.reshape(1, *global_params.INPUT_SHAPE)
            prediction = model.predict(hand_landmarks)
            predicted_class_index = prediction.argmax()
            predicted_label = class_labels[predicted_class_index]
            confidence = prediction[0][predicted_class_index]
            
            # Printing the predicted sign and the confidence of the prediction
            print(f'Predicted Sign: {predicted_label} (Confidence: {confidence * 100:.1f}%)')

            # Display the prediction on the image
            cv2.putText(image, f'{predicted_label} ({confidence * 100:.1f}%)', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the image on screen
        cv2.imshow('Live Camera', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
