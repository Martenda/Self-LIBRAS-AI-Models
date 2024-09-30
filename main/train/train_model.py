import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from train.build_model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint
import global_params
from track_and_extract_landmarks import extract_landmarks
import mediapipe as mp

def prepare_dataset(dataset_path):
    # Define file paths for the processed data
    processed_X_path = global_params.SAVED_PRE_PROCESSED_DATA_DIR + 'processed_X.npy'
    processed_y_path = global_params.SAVED_PRE_PROCESSED_DATA_DIR + 'processed_y.npy'

    # Check if the processed data files exist
    if os.path.exists(processed_X_path) and os.path.exists(processed_y_path):
        print('\nLoading processed data from files...')
        X = np.load(processed_X_path)
        y = np.load(processed_y_path)
        print('\nProcessed data loaded successfully.')
    else:
        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        
        X, y = [], []
        
        print('\nProcessing dataset...')
        
        # Iterate over each label (class)
        for label in os.listdir(dataset_path):
            label_path = os.path.join(dataset_path, label)
            # Iterate over each image file in the label directory
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                image = cv2.imread(img_path)
                landmarks = extract_landmarks(image, hands)
                
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(label)
        
        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Save the processed data to files
        np.save(processed_X_path, X)
        np.save(processed_y_path, y)
        print('\nProcessed data saved to files.')

    return X, y

def train_model():
    # Define paths
    train_data_dir = global_params.TRAIN_DATA_DIR
    val_data_dir = global_params.VAL_DATA_DIR
    
    X, y = prepare_dataset(train_data_dir)
    print('\nDataset prepared successfully.')
    
    # Ensure X has the correct shape
    X = X.reshape(X.shape[0], *global_params.INPUT_SHAPE)
    
    # Split the data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Convert labels to integers
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    # Save class labels
    import numpy as np
    np.save(global_params.SAVED_PRE_PROCESSED_DATA_DIR + 'class_labels.npy', label_encoder.classes_)
    
    # Convert labels to categorical (one-hot encoding)
    from tensorflow.keras.utils import to_categorical
    num_classes = len(label_encoder.classes_)
    y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)
    y_val_categorical = to_categorical(y_val_encoded, num_classes=num_classes)

    model = create_model(global_params.INPUT_SHAPE, num_classes)
    print('\nModel created successfully.')
    
    # Print the model summary
    model.summary()
    
    # Data augmentation for training data
    print('\nGenerating data augmentation...')
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2
    )
    print('\nData augmentation for training data prepared.')
    
    print('\nFitting training data generator...')
    train_datagen.fit(X_train)
    print('\nTraining data generator fitted successfully.')
    
    # No augmentation for validation data
    val_datagen = ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow(
        X_train,
        y_train_categorical,
        batch_size=32
    )
    val_generator = val_datagen.flow(
        X_val,
        y_val_categorical,
        batch_size=32
    )
    
    # Model training
    print('\nSetting up model checkpoint...')
    checkpoint = ModelCheckpoint(global_params.SAVED_MODEL_PATH, save_best_only=True)
    print('\nModel checkpoint set up.')
    
    print('\nTraining Model...')
    model.fit(
        train_generator,
        epochs=global_params.EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint]
    )
    print('\nModel trained successfully.')
    
    # Save the trained model
    print('\nSaving Model...')
    model.save(global_params.SAVED_MODEL_PATH)
    print('\nModel saved successfully.')
