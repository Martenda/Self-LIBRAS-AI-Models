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
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    
    X, y = [], []
    
    print('\nPreparing dataset...')
    
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            image = cv2.imread(img_path)
            landmarks = extract_landmarks(image, hands)
            
            if landmarks is not None:
                X.append(landmarks)
                y.append(label)

    return np.array(X), np.array(y)

def train_model():
    # Define paths
    train_data_dir = global_params.TRAIN_DATA_DIR
    val_data_dir = global_params.VAL_DATA_DIR
    
    X, y = prepare_dataset(train_data_dir)
    print('\nDataset prepared successfully.')
    
    X = X.reshape(X.shape[0], *global_params.INPUT_SHAPE)
    
    num_classes = len(np.unique(y))

    model = create_model(global_params.INPUT_SHAPE, num_classes)
    print('\nModel created successfully.')

    # Print the model summary
    model.summary()

    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2)
    datagen.fit(X)

    # Model training
    checkpoint = ModelCheckpoint(global_params.MODEL_PATH, save_best_only=True)
    model.fit(datagen.flow(X, y, batch_size=32), epochs=1, validation_split=0.2, callbacks=[checkpoint])
    print('\nModel trained successfully.')

    # Save the trained model
    model.save(global_params.SAVED_MODEL_PATH)
    print('\nModel saved successfully.')

    # # Image Data Generator for augmentation
    # train_datagen = ImageDataGenerator(
    #     rescale=1.0/255,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True
    # )
    
    # val_datagen = ImageDataGenerator(rescale=1.0/255)
    
    # # Load training data
    # train_generator = train_datagen.flow_from_directory(
    #     train_data_dir,
    #     target_size=(global_params.IMG_HEIGHT, global_params.IMG_WIDTH),
    #     batch_size=global_params.BATCH_SIZE,
    #     class_mode='categorical'
    # )
    
    # # Load validation data
    # val_generator = val_datagen.flow_from_directory(
    #     val_data_dir,
    #     target_size=(global_params.IMG_HEIGHT, global_params.IMG_WIDTH),
    #     batch_size=global_params.BATCH_SIZE,
    #     class_mode='categorical'
    # )
    
    # # Create and compile the model
    # model = create_model(global_params.IMG_HEIGHT, global_params.IMG_WIDTH, global_params.NUM_CLASSES)
    # print('\nModel created successfully.')

    # model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # print('\nModel compiled successfully.')

    # # Train the model
    # history = model.fit(
    #     train_generator,
    #     steps_per_epoch=train_generator.samples // global_params.BATCH_SIZE,
    #     validation_data=val_generator,
    #     validation_steps=val_generator.samples // global_params.BATCH_SIZE,
    #     epochs=global_params.EPOCHS
    # )
    # print('\nModel trained successfully.')

    # # Save the trained model
    # model.save(global_params.SAVED_MODEL_PATH)
    # print('\nModel saved successfully.')
