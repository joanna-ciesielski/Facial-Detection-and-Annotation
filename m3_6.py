import cv2
import os

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to the cascade files
face_cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join(script_dir, 'haarcascade_eye.xml')

# Load the Haar cascades
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Verify that the cascade files have been loaded correctly
if face_cascade.empty():
    print(f"Error loading face cascade from {face_cascade_path}")
    exit()
else:
    print("Successfully loaded face cascade.")

if eye_cascade.empty():
    print(f"Error loading eye cascade from {eye_cascade_path}")
    exit()
else:
    print("Successfully loaded eye cascade.")

# Path to the image
image_path = os.path.join(script_dir, 'photo.png')

# Load the image
img = cv2.imread(image_path)
if img is None:
    print(f"Could not read image at {image_path}")
    exit()
else:
    print("Image loaded successfully.")

# Convert the image to grayscale for detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

# Check if any faces were detected
if len(faces) == 0:
    print("No faces detected in the image.")
else:
    print(f"Detected {len(faces)} face(s) in the image.")

# Loop over each face detected
for (x, y, w, h) in faces:
    # Draw a green circle around the face
    center_coordinates = (x + w // 2, y + h // 2)
    radius = int(round((w + h) * 0.25))
    cv2.circle(img, center_coordinates, radius, (0, 255, 0), 2)

    # Define the region of interest (ROI) for eyes within the face
    face_roi_gray = gray[y:y + h, x:x + w]
    face_roi_color = img[y:y + h, x:x + w]

    # Only detect eyes in the top half of the face (to avoid detecting mouth/nose)
    eye_region_gray = face_roi_gray[0:int(h * 0.5), :]
    eye_region_color = face_roi_color[0:int(h * 0.5), :]

    # Adjust the parameters for eye detection
    eyes = eye_cascade.detectMultiScale(eye_region_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

    # Filter eyes based on size and position (keeping only the two largest detections)
    if len(eyes) > 0:
        # Sort eyes by width (the larger the detection, the more likely it is an actual eye)
        eyes = sorted(eyes, key=lambda e: e[2], reverse=True)
        largest_eyes = []

        for (ex, ey, ew, eh) in eyes:
            # Only keep two largest detections
            if len(largest_eyes) < 2:
                largest_eyes.append((ex, ey, ew, eh))

        # Draw red bounding boxes around the top two largest detected eyes
        for (ex, ey, ew, eh) in largest_eyes:
            cv2.rectangle(eye_region_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

# Add text "this is me" to the image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'this is me', (10, img.shape[0] - 10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Save the modified image
output_image_path = os.path.join(script_dir, 'photo_annotated_filtered.png')
cv2.imwrite(output_image_path, img)
print(f"Annotated image saved at {output_image_path}")

# Display the image (optional)
cv2.imshow('Annotated Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
