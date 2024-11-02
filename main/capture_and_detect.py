import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from track_and_extract_landmarks import extract_landmarks
import global_params
import numpy as np  # Import numpy to load the class labels
import joblib
import time  # Import time to measure FPS

def capture_camera(draw_landmarks_on_camera):
    # Initialize video capture from the camera
    cap = cv2.VideoCapture(0)
    
    # Using the IP address of the phone's camera
    # ip_camera_url = 'http://192.168.1.65:8080/video'
    # cap = cv2.VideoCapture(ip_camera_url)

    # Load the trained model
    model = load_model(global_params.SAVED_MODEL_PATH)

    # Load models
    rf_model = joblib.load(global_params.SAVED_MODEL_DIR_PATH + 'random_forest_model.joblib')
    svm_model = joblib.load(global_params.SAVED_MODEL_DIR_PATH + 'svm_model.joblib')
    knn_model = joblib.load(global_params.SAVED_MODEL_DIR_PATH + 'knn_model.joblib')

    # Load label encoder
    label_encoder = joblib.load(global_params.SAVED_MODEL_DIR_PATH + 'label_encoder.joblib')

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

    # Initialize FPS variables
    prev_frame_time = 0
    new_frame_time = 0

    target_letter = ''

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Measure FPS timing
        new_frame_time = time.time()

        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)

        # Extract hand landmarks
        hand_landmarks = extract_landmarks(image, mp_hands, draw_landmarks_on_camera, mp.solutions)

        # If landmarks are found, reshape and make prediction
        if hand_landmarks is not None:
            # Predict the Sign using the pre-loaded Model
            hand_landmarks_reshaped = hand_landmarks.reshape(1, *global_params.INPUT_SHAPE)
            prediction = model.predict(hand_landmarks_reshaped)
            predicted_class_index = prediction.argmax()
            predicted_label = class_labels[predicted_class_index]
            confidence = prediction[0][predicted_class_index]
            
            # Flatten landmarks
            landmarks_flat = hand_landmarks.flatten().reshape(1, -1)
            
            # # Make predictions
            # rf_prediction = rf_model.predict(landmarks_flat)
            # svm_prediction = svm_model.predict(landmarks_flat)
            # knn_prediction = knn_model.predict(landmarks_flat)
            
            # Get prediction probabilities
            rf_probs = rf_model.predict_proba(landmarks_flat)
            svm_probs = svm_model.predict_proba(landmarks_flat)
            knn_probs = knn_model.predict_proba(landmarks_flat)

            # Get the predicted class indices
            rf_prediction = np.argmax(rf_probs, axis=1)
            svm_prediction = np.argmax(svm_probs, axis=1)
            knn_prediction = np.argmax(knn_probs, axis=1)

            # Get the confidence scores
            rf_confidence = np.max(rf_probs, axis=1)
            svm_confidence = np.max(svm_probs, axis=1) # SVM probability estimates are calculated using cross-validation and may not be as reliable as those from probabilistic models. Use them cautiously and consider validating the confidence scores.
            knn_confidence = np.max(knn_probs, axis=1)
            
            # Decode labels
            rf_label = label_encoder.inverse_transform(rf_prediction)[0]
            svm_label = label_encoder.inverse_transform(svm_prediction)[0]
            knn_label = label_encoder.inverse_transform(knn_prediction)[0]
            
            # Printing the predicted sign and the confidence of the prediction
            print(f'CNN: {predicted_label} (Confidence: {confidence * 100:.1f}%)')
            print(f'RF : {rf_label} (Confidence: {rf_confidence[0] * 100:.1f}%)')
            print(f'SVM: {svm_label} (Confidence: {svm_confidence[0] * 100:.1f}%)')
            print(f'KNN: {knn_label} (Confidence: {knn_confidence[0] * 100:.1f}%)')

            # Display the predictions on the image
            cv2.putText(image, f'CNN:',                             (5, 30),    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),   2)
            cv2.putText(image, f'{predicted_label}',                (85, 30),   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),   2)
            cv2.putText(image, f'({confidence * 100:.1f}%)',        (115, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),   2)
            # cv2.putText(image, f'CNN: {predicted_label} ({confidence * 100:.1f}%)', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'RF:',                              (5, 70),    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),   2)
            cv2.putText(image, f'{rf_label}',                       (85, 70),   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),   2)
            cv2.putText(image, f'({rf_confidence[0] * 100:.1f}%)',  (115, 70),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),   2)
            # cv2.putText(image, f'RF : {rf_label} ({rf_confidence[0] * 100:.1f}%)', (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, f'SVM:',                             (5, 110),   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(image, f'{svm_label}',                      (85, 110),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(image, f'({svm_confidence[0] * 100:.1f}%)', (115, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            # cv2.putText(image, f'SVM: {svm_label} ({svm_confidence[0] * 100:.1f}%)', (5, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(image, f'KNN:',                             (5, 150),   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),   2)
            cv2.putText(image, f'{knn_label}',                      (85, 150),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),   2)
            cv2.putText(image, f'({knn_confidence[0] * 100:.1f}%)', (115, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),   2)
            # cv2.putText(image, f'KNN: {knn_label} ({knn_confidence[0] * 100:.1f}%)', (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Check if any model predicts the target letter with at least 50% confidence
            message_y_position = 200  # Starting position for messages
            if predicted_label == target_letter and confidence >= 0.5:
                cv2.putText(image, 'Congrats! CNN predicted your letter!', (5, message_y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                message_y_position += 30
            if rf_label == target_letter and rf_confidence[0] >= 0.5:
                cv2.putText(image, 'Congrats! RF predicted your letter!', (5, message_y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                message_y_position += 30
            if svm_label == target_letter and svm_confidence[0] >= 0.5:
                cv2.putText(image, 'Congrats! SVM predicted your letter!', (5, message_y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                message_y_position += 30
            if knn_label == target_letter and knn_confidence[0] >= 0.5:
                cv2.putText(image, 'Congrats! KNN predicted your letter!', (5, message_y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                message_y_position += 30

        # Calculate FPS
        fps = 1 / (new_frame_time - prev_frame_time)
        # print(new_frame_time - prev_frame_time) # miliseconds of each processed frame
        prev_frame_time = new_frame_time
        # Display FPS on the image
        cv2.putText(image, f'FPS: {int(fps)}', (5, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the image on screen with the custom window size
        cv2.imshow('Live Camera', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
