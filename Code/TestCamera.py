import cv2

# Open the webcam (default index is 0; change if another camera is connected)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Using DirectShow backend for compatibility

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press ESC to close the window.")

# Display the webcam feed
while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # Show the live camera feed in a window
    cv2.imshow("Camera Feed", frame)

    # Exit when the ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
