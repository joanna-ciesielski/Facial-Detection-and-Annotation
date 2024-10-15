# Facial Detection and Image Annotation with OpenCV

## Overview
This project utilizes OpenCV to detect faces and annotate images with bounding boxes around the eyes. It includes image processing techniques to detect the region of interest (ROI) for eyes within a face using Haar cascades. The project is designed to annotate images with detected facial features.

## Features
- Detects faces and eyes using Haar cascades.
- Annotates the largest detected eyes with bounding boxes.
- Includes an optional display of annotated images.
- Saves the processed and annotated images to a specified file.

## Technologies Used
- **Python**
- **OpenCV**: For image processing and facial detection.
- **Haar Cascades**: Pre-trained classifiers for face and eye detection.

## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
    ```
2. Install the required dependencies:
    ```bash
    pip install opencv-python
    ```
3. Place an image named `photo.png` in the project directory.
4. Run the Python script:
    ```bash
    python m3_6.py
    ```

## Output
- The script will detect faces and eyes, annotate them on the image, and save the processed image as `photo_annotated_filtered.png`.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
