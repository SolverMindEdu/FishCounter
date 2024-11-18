from tensorflow.keras.models import load_model
import json

try:
    model = load_model("Data/Images/Keras/converted_keras/keras_model.h5", compile=False)
except Exception as e:
    print(f"Error loading the model: {e}")n
    with open("model_config.json", "w") as config_file:
        model_config = model.to_json()
        config_file.write(model_config)
    print("Model configuration saved as model_config.json for manual editing.")
    exit()
