import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from track_and_extract_landmarks import extract_landmarks
import global_params
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import joblib

# Handling the image processing asynchronously
async def process_frame(frame_data, model, class_labels, rf_model, svm_model, knn_model, label_encoder, mp_hands):
    # Convert the frame from IO Binary text to Base64 text
    frame = Image.open(BytesIO(base64.b64decode(frame_data.split(",")[1])))
    # Convert the frame to Numpy array
    frame_np = np.array(frame)
    # Convert the frame to OpenCV format
    frame_cv = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

    # Extract hand landmarks
    hand_landmarks = extract_landmarks(frame_cv, mp_hands, False, mp.solutions)

    # If landmarks are found, reshape and make prediction
    if hand_landmarks is not None:
        # Predict the Sign using the pre-loaded Model
        hand_landmarks = hand_landmarks.reshape(1, *global_params.INPUT_SHAPE)
        prediction = model.predict(hand_landmarks)
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
        # print(f'CNN: {predicted_label} (Confidence: {confidence * 100:.1f}%)')
        # print(f'RF : {rf_label} (Confidence: {rf_confidence[0] * 100:.1f}%)')
        # print(f'SVM: {svm_label} (Confidence: {svm_confidence[0] * 100:.1f}%)')
        # print(f'KNN: {knn_label} (Confidence: {knn_confidence[0] * 100:.1f}%)')

        return f"CNN: {predicted_label} ({confidence * 100:.1f}%) - RF: {rf_label} ({rf_confidence[0] * 100:.1f}%) - SVM: {svm_label} ({svm_confidence[0] * 100:.1f}%) - KNN: {knn_label} ({knn_confidence[0] * 100:.1f}%)"
        # return predicted_label
        # return f"{predicted_label} (CNN, Confiabilidade: {confidence * 100:.1f}%) - "+f"{rf_label} (RF, Confiabilidade: {rf_confidence[0] * 100:.1f}%) - "+f"{svm_label} (SVM, Confiabilidade: {svm_confidence[0] * 100:.1f}%) - "+f"{knn_label} (KNN, Confiabilidade: {knn_confidence[0] * 100:.1f}%)"
    
    return "Faça um sinal pra câmera ;)"

def capture_camera_online_websocket():
    # Creates the API App
    app = FastAPI()

    # Allow CORS for the specified origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://self-libras.vercel.app/"], # Restricting to Self LIBRAS webapp only ("https://self-libras.vercel.app/"), or "*" for all
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load the trained model
    model = load_model(global_params.SAVED_MODEL_PATH)
    # Load class labels
    class_labels = np.load(global_params.SAVED_MODEL_DIR_PATH + '../class_labels.npy')
    
    # Load models
    rf_model = joblib.load(global_params.SAVED_MODEL_DIR_PATH + 'random_forest_model.joblib')
    svm_model = joblib.load(global_params.SAVED_MODEL_DIR_PATH + 'svm_model.joblib')
    knn_model = joblib.load(global_params.SAVED_MODEL_DIR_PATH + 'knn_model.joblib')
    # Load label encoder
    label_encoder = joblib.load(global_params.SAVED_MODEL_DIR_PATH + 'label_encoder.joblib')

    # Initialize MediaPipe Hand solution
    mp_hands = mp.solutions.hands.Hands()

    # Defines the API App behavior, when calling '/live_camera' endpoint
    @app.websocket("/live_camera")
    # Handle live-camera frames sent from client
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()

    #     # Debugging
    #     minhadataehora = datetime.now().strftime('%H:%M:%S.%f')
    #     print(f'\nANTES Predicted sign: {minhadataehora}')

    #     # print('\nTesting if its reading the image...')
    #     minhadataehora = datetime.now().strftime('%H:%M:%S.%f')
    #     print(f'\nDEPOIS Predicted sign: {minhadataehora}')

    #     # frame_exibicao = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # PIL uses RGB; OpenCV uses BGR
    #     # cv2.imshow('Received Image', frame_exibicao)
    #     # # Wait for a keypress and close the display window
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()

        try:
            while True:
                # Receive frame data
                data = await websocket.receive_text()
                # Process the frame asynchronously
                result = await process_frame(data, model, class_labels, rf_model, svm_model, knn_model, label_encoder, mp_hands)
                # Send result back to client
                await websocket.send_text(result)
        except WebSocketDisconnect:
            print("Client disconnected")
        except Exception as e:
            print(f"Connection error: {e}")
    
    # Starts the API App
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
    # uvicorn.run("main:app", host="0.0.0.0", port=5000)
