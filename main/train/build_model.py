from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Reshape

def create_model(input_shape, num_classes):
    model = Sequential()
    
    ### 2D CNN layers ###
    model.add(Conv2D(512, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 1)))  # Pooling only over height
    model.add(Dropout(0.25))
    
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 1)))  # Pooling only over height
    model.add(Dropout(0.25))
    
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 1)))  # Pooling only over height
    model.add(Dropout(0.25))
    ### - ###

    ### Reshaping data for the following layers ###
    # Flattening and Reshaping
    model.add(Flatten()) # Convert 2D to 1D vector
    num_features = model.output_shape[1] // 1

    # Reshape to make it suitable for LSTM layers
    model.add(Reshape((1, num_features)))
    ### - ###

    # Hint: Use ConvLSTM2D...
    ### LSTM layers ###    
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512))
    ### - ###

    ### Fully connected layers ###
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    ### - ###
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'precision'])
    
    return model
