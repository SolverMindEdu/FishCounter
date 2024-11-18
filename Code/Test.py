import tensorflow as tf
import keras
import cv2
import numpy as np
import time

SAVED_MODEL_PATH = "Data/Images/SavedModel/converted_savedmodel/model.savedmodel"
LABELS_PATH = "Data/Images/SavedModel/converted_savedmodel/labels.txt"

np.set_printoptions(suppress=True)

try:
    model_layer = keras.layers.TFSMLayer(SAVED_MODEL_PATH, call_endpoint="serving_default")
    print("SavedModel loaded successfully using TFSMLayer.")
except Exception as e:
    print(f"Error loading the SavedModel: {e}")
    exit()

try:
    class_names = open(LABELS_PATH, "r").readlines()
    print("Labels loaded successfully.")
except FileNotFoundError:
    print(f"Error: Labels file not found at {LABELS_PATH}")
    exit()

fish_count = 0
last_detected_time = 1 
cooldown_seconds = 2 

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Frame will remain open. Use external triggers or terminate manually to stop.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame from camera.")
        continue
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    input_image = np.asarray(resized_frame, dtype=np.float32).reshape(1, 224, 224, 3)
    input_image = (input_image / 127.5) - 1 

    try:
        prediction = model_layer(input_image)
    
        if isinstance(prediction, dict):
            output_key = list(prediction.keys())[0]
            prediction = prediction[output_key]  # Don't call .numpy() if it's already an ndarray
        
        # At this point, `prediction` should be a NumPy array.
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]
        current_time = time.time()
        if index == 0: 
            if current_time - last_detected_time > cooldown_seconds:
                fish_count += 1  
                last_detected_time = current_time 
                print(f"Fish detected! Total count: {fish_count}")

            # Draw rectangle for "Fish"
            start_point = (50, 50) 
            end_point = (frame.shape[1] - 50, frame.shape[0] - 50)  # Bottom-right corner
            color = (0, 0, 255)  # Red color for rectangle
            thickness = 2  # Thickness of the rectangle
            cv2.rectangle(frame, start_point, end_point, color, thickness)

        # Overlay prediction and fish count on the frame
        label = f"{class_name}: {np.round(confidence_score * 100, 2)}%"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Fish Count: {fish_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    except Exception as pred_error:
        print(f"Error during prediction: {pred_error}")
        continue
    cv2.imshow("Live Prediction and Fish Count", frame)
    cv2.waitKey(1)
camera.release()
cv2.destroyAllWindows()
