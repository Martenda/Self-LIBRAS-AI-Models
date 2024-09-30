# # Image dimensions
# IMG_HEIGHT = 128
# IMG_WIDTH = 128

# # Batch size and number of epochs
# BATCH_SIZE = 32
EPOCHS = 50

# Paths to data and saved models
TRAIN_DATA_DIR = '../datasets/training'
VAL_DATA_DIR = '../datasets/test'
SAVED_PRE_PROCESSED_DATA_DIR = '../working_area/pre-processed_data_from_videos/'
SAVED_MODEL_PATH = '../models/libras_static_signs_model.keras'

SEQUENCE_LENGTH = 1    # Number of frames for static signs
HEIGHT, WIDTH = 21, 3  # 21 landmarks with 3 coordinates (x, y, z)
CHANNELS = 1           # Single channel for 2D CNN
INPUT_SHAPE = (HEIGHT, WIDTH, CHANNELS)  # Input shape for model
