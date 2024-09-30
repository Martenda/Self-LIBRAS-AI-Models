from capture_and_detect import capture_camera
# from train import train.index_train
from train.train_model import train_model

if __name__ == "__main__":

    print('\n' + 'Execution inittiated.' + '\n')
    
    # train_model()
    capture_camera()
    
    print('\n' + 'Execution successfully done.' + '\n')
