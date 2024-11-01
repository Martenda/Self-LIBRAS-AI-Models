import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from track_and_extract_landmarks import extract_landmarks
import global_params
import numpy as np  # Import numpy to load class labels

def capture_camera(draw_landmarks_on_camera):
    # Initialize video capture from the camera
    cap = cv2.VideoCapture(0)
    
    # Using the IP address of the phone's camera
    # ip_camera_url = 'http://192.168.1.65:8080/video'
    # cap = cv2.VideoCapture(ip_camera_url)

    # Load the trained model
    model = load_model(global_params.SAVED_MODEL_PATH)

    # Load class labels
    class_labels = np.load(global_params.SAVED_PRE_PROCESSED_DATA_DIR + 'class_labels.npy')

    # Initialize MediaPipe Hand solution
    mp_hands = mp.solutions.hands.Hands()

    # Create a named window with the ability to resize
    cv2.namedWindow('Live Camera', cv2.WINDOW_NORMAL)

    # Set a custom size for the window (deafult defined as 80% of the screen size)
    screen_width = 1920  # It can be adjusted based on screen resolution
    screen_height = 1080  # It can be adjusted based on screen resolution
    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)
    
    # Resize the window to the percentage of the screen size pre-defined
    cv2.resizeWindow('Live Camera', window_width, window_height)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)

        # Extract hand landmarks
        hand_landmarks = extract_landmarks(image, mp_hands, draw_landmarks_on_camera, mp.solutions)

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

        # Show the image on screen with the custom window size
        cv2.imshow('Live Camera', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
