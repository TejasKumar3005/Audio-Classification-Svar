import tensorflow as tf
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
# Load the model
model = tf.keras.models.load_model('./models/conv2d_1st.h5', custom_objects={'STFT': STFT, "Magnitude": Magnitude, "ApplyFilterbank": ApplyFilterbank, "MagnitudeToDecibel": MagnitudeToDecibel})

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_quant_model = converter.convert()

# Save the quantized model to a .tflite file
# with open('quantized_model.tflite', 'wb') as f:
#     f.write(tflite_quant_model)


# Save the TFLite model
with open('model1.tflite', 'wb') as f:
    f.write(tflite_model)
    


