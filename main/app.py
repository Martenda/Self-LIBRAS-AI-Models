from capture_and_detect import capture_camera
from train.train_model import train_model
from websocket_api.websocket_capture_and_detect import capture_camera_online_websocket

if __name__ == "__main__":

    print('\n' + 'Execution inittiated.' + '\n')
    
    # train_model()
    # capture_camera(draw_landmarks_on_camera=True)
    capture_camera_online_websocket()
    
    print('\n' + 'Execution successfully done.' + '\n')
