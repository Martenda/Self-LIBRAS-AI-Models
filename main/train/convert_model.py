import tensorflowjs as tfjs
import tensorflow as tf
import global_params

# Load your Keras model
model = tf.keras.models.load_model('../' + global_params.SAVED_MODEL_PATH)

# Convert and save the model to TensorFlow.js format
tfjs.converters.save_keras_model(model, '../../models/JSON_models')


# pip install tensorflowjs
# tensorflowjs_converter --input_format=keras path/to/your/model.keras path/to/output/directory
