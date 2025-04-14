<p align="center">
  <img src="https://github.com/user-attachments/assets/f4eb5ec2-cad1-40f9-aa40-8922779490e6" width="128">
</p>

<h1 align="center">Self LIBRAS - A.I. Models & API</h1>

A deep learning-powered backend for recognizing static LIBRAS (Brazilian Sign Language) signs in real-time. This backend processes images using advanced AI models, including CNN, LSTM, and classical machine learning algorithms, to provide high-accuracy sign classification.

This project was developed as part of a top-grade undergraduate thesis (TCC), and this research was scientifically validated with statistical rigor, establishing it as an effective learning tool with proven success.

<br>

<div align="center">
  <a href='https://opensource.org/license/mit'>
    <img src='https://img.shields.io/badge/License-MIT-C20018.svg?logo=opensourceinitiative&logoColor=FFFFFF'>
  </a>
  <a href='https://www.python.org/'>
    <img src='https://img.shields.io/badge/Python-3.9.12-31C754.svg?logo=python&logoColor=31C754'>
  </a>
  <a href='https://www.tensorflow.org/'>
    <img src='https://img.shields.io/badge/TensorFlow-2.17-FF7300.svg?logo=tensorflow'>
  </a>
  <a href='https://opencv.org/'>
    <img src='https://img.shields.io/badge/OpenCV-4.10-128EFF.svg?logo=opencv'>
  </a>
  <a href='https://mediapipe.dev/'>
    <img src='https://img.shields.io/badge/MediaPipe-0.10.14-0097A7.svg?logo=mediapipe'>
  </a>
</div>

<br>

<div align="center">
  <a href='https://scikit-learn.org/'>
    <img src='https://img.shields.io/badge/scikit_learn-1.5.2-F7931E.svg?logo=scikitlearn'>
  </a>
  <a href='https://fastapi.tiangolo.com/'>
    <img src='https://img.shields.io/badge/FastAPI-0.115.3-05988A.svg?logo=fastapi'>
  </a>
  <a href='https://websockets.readthedocs.io/'>
    <img src='https://img.shields.io/badge/WebSockets-13.1-FFD743.svg'>
  </a>
  <a href='https://www.uvicorn.org/'>
    <img src='https://img.shields.io/badge/Uvicorn-0.32.0-4051B5.svg?logo=gunicorn&logoColor=FFFFFF'>
  </a>
  <a href='https://numpy.org/'>
    <img src='https://img.shields.io/badge/NumPy-1.26.4-4D77CF.svg?logo=numpy'>
  </a>
  <a href='https://matplotlib.org/'>
    <img src='https://img.shields.io/badge/Matplotlib-3.9.2-FEA96F.svg'>
  </a>
</div>

## Production Link  
You'll love to try the final product in production! Check it out at: **[self-libras.vercel.app/](https://self-libras.vercel.app/)**

## Article and Presentation

Read the full monograph: [Article at the Institutional Repository](https://repositorio.udesc.br/entities/publication/c370637b-e1fd-4e3e-9a79-0e7fcd44db5e)

See the presentation: [Presentation to the Doctoral Committee.pdf](https://github.com/user-attachments/files/19609969/Apresentacao.para.Banca.-.Self.LIBRAS.vFinal.pdf)

## Frontend Repository

Link to the WebApp (frontend) repository: [github.com/Martenda/Self-LIBRAS-WebApp](https://github.com/Martenda/Self-LIBRAS-WebApp)

<!--
## Demo  
GIFs here...  
-->

## Screenshots  

Prototype (MVP):

<img src="https://github.com/user-attachments/assets/62bcb094-e173-4b05-be33-c7dcc08cb8dd" width="450">
<img src="https://github.com/user-attachments/assets/c95e45c2-2d7d-42aa-bf82-5fc13d818540" width="450">
<img src="https://github.com/user-attachments/assets/642c7749-7d7f-4e48-842d-04fcc644c7fe" width="450">
<img src="https://github.com/user-attachments/assets/9d82c91d-60ee-4d5f-b555-2d698850b0ed" width="450">
<img src="https://github.com/user-attachments/assets/1c6837bf-5bb2-4153-a8f3-8e62dc06c16f" width="450">
<img src="https://github.com/user-attachments/assets/e4507d03-2a08-421b-a882-9ba14f1c134a" width="450">

WebApp (Final Product):

<img src="https://github.com/user-attachments/assets/ed943c18-b6bf-400f-a35c-47e6255555dd" width="450">
<img src="https://github.com/user-attachments/assets/506e2d56-5e26-44d2-a5a5-d857a2647aee" width="450">
<img src="https://github.com/user-attachments/assets/003737b6-ce59-4087-9977-9c53a4198036" width="450">
<img src="https://github.com/user-attachments/assets/92585b29-1d7e-4725-ab2e-d83580cdc60a" width="450">
<img src="https://github.com/user-attachments/assets/55ec1957-d00e-4fd9-bf0c-779ce7296256" width="225">
<img src="https://github.com/user-attachments/assets/8adec190-0038-453c-ac67-304a1b680a3e" width="225">

## Features  
✔ **Real-time AI-powered sign recognition** using a combination of CNN and LSTM  
✔ **Classical ML models (RF, SVM, KNN)** integrated for robust sign detection  
✔ **WebSocket-based live inference** for fast video stream processing  
✔ **Data augmentation** for improved model generalization  
✔ **Efficient hand tracking** with MediaPipe  
✔ **Trained on a custom dataset** of LIBRAS static signs  

## Tech Stack  
- **Deep Learning:** TensorFlow, Keras  
- **Machine Learning:** Random Forest, SVM, KNN (Scikit-learn)  
- **Computer Vision:** OpenCV, MediaPipe  
- **Server Framework:** FastAPI  
- **Model Deployment:** WebSocket API for real-time predictions  

## AI Pipeline Overview  
1. **Hand Landmark Detection:** Uses MediaPipe to extract 3D hand landmarks.  
2. **Feature Extraction:** The extracted landmarks are processed into a structured format.  
3. **Model Prediction:**  
   - **Deep Learning (CNN + LSTM):** Classifies the sign using a trained model.  
   - **Machine Learning (RF, SVM, KNN):** Provides secondary verification.  
4. **Real-Time Feedback:** Returns predicted sign with confidence scores.  

## Usage  

### Installation  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/Martenda/Self-LIBRAS-AI-Models.git  
cd self-libras-ai-models  
pip install -r requirements.txt  
```  

### Execution  
Run the AI inference server:  
```bash
python capture_and_detect.py  
```  

### Deployment  
Deploy the AI backend as a WebSocket API:  
```bash
uvicorn websocket_capture_and_detect:capture_camera_online_websocket --host 0.0.0.0 --port 5000  
```  

### Model Training  
To train a new model from scratch using the dataset:  
```bash
python train_model.py  
```  

## API Endpoints  
- **`/live_camera` (WebSocket):** Processes live video frames for sign recognition.  

## Contact the Author  
- [Lucas Martendal on GitHub](https://github.com/Martenda)  
- [Lucas Martendal on LinkedIn](https://www.linkedin.com/in/lucas-martendal/)  
- [E-mail to lucasmartendal777@gmail.com](mailto:lucasmartendal777@gmail.com)  

## Feedback  
If you have any feedback, please feel free to reach me out at [lucasmartendal777@gmail.com](mailto:lucasmartendal777@gmail.com)   

## License  
This project is licensed by [The MIT License](https://opensource.org/license/mit).  

<p align="center">
  <img src="https://github.com/user-attachments/assets/547fca19-0830-492d-8240-7bea38f8e16b" width="250">
  <!--
  <img src="https://i.pinimg.com/originals/02/c9/08/02c90823e70f897ab8d64d2455e8553d.gif" width="250">
  -->
</p>

<p align="center">
<b>Thank you so much! (in LIBRAS 😆)</b>
<!--
Thank you! (in ASL)
-->
</p>
