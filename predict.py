import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

# 1. Load the TFLite model globally so it stays in memory (Cache this in app.py if possible)
interpreter = Interpreter(model_path="deepfake_model.tflite")
interpreter.allocate_tensors()

# 2. Get the input and output node details from the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(image_path):
    # Get the required input shape for your specific model (e.g., 224x224 or 256x256)
    input_shape = input_details[0]['shape']
    target_height, target_width = input_shape[1], input_shape[2]
    
    # Read and preprocess the image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_width, target_height))
    
    # Convert to float32 array
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize the pixels (Important: check if your original model used /255.0 scaling)
    img_array = img_array / 255.0
    
    # Add the batch dimension so shape becomes (1, height, width, channels)
    input_data = np.expand_dims(img_array, axis=0)

    # 3. Feed the image into the TFLite model
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 4. Run the prediction
    interpreter.invoke()

    # 5. Extract the result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Assuming a binary classification (e.g., Fake vs Real)
    confidence = output_data[0][0]
    return confidence