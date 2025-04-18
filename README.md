# Forest Fire Detection

This project leverages a Convolutional Neural Network (CNN) to detect forest fires from images, classifying them as **"Forest Fire"** or **"No Forest Fire"** with ~92% accuracy. Built using TensorFlow and deployed via a Streamlit app, it aims to enhance environmental safety by automating wildfire detection, reducing reliance on manual monitoring, and supporting proactive disaster prevention.

## Features

- **Image Classification:** Classifies uploaded images using a pre-trained CNN model.
- **Streamlit App:** User-friendly interface to upload images and view classification results with confidence levels.
- **Automation:** Eliminates manual fire detection, improving speed and scalability.
- **Environmental Impact:** Supports ecosystem preservation and public safety through early fire detection.

## Prerequisites

- Python 3.8+
- Git
- Git LFS (for handling large model files)
- Streamlit Cloud account (optional, for deploying the app)

## Installation

1. **Clone the Repository:**
    ```sh
    git clone https://github.com/Jay0073/Forest-Fire-Detection.git
    cd Forest-Fire-Detection
    ```
2. **Set Up Git LFS:**
    - Install Git LFS:
        ```sh
        git lfs install
        ```
    - Pull large files:
        ```sh
        git lfs pull
        ```
3. **Create a Virtual Environment** (recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4. **Install Dependencies:**
    ```sh
    pip install tensorflow streamlit pillow numpy
    ```

## Usage

- **Run the Streamlit App Locally:**
    ```sh
    streamlit run forest_fire_app.py
    ```
    Open the provided URL (e.g., [http://localhost:8501](http://localhost:8501)) in your browser.
    - Upload a `.jpeg` or `.png` image to classify it as "Forest Fire" or "No Forest Fire".
- **Test the Deployed App (if hosted):**
    - Visit the Streamlit App URL (replace with your deployed app link, e.g., Streamlit Cloud).
    - Upload an image to view classification results.

## Model Details

- The `Forest_fire_detection.keras` model expects `224x224` RGB images.
- Outputs binary classification with confidence scores.

## File Structure

- `Forest_fire_detection.keras` : Pre-trained CNN model (~254 MB, managed by Git LFS).
- `forest_fire_app.py` : Streamlit app for image classification.
- `.gitattributes` : Git LFS configuration for large files.

## Future Scope

- Enhance model accuracy with larger, diverse datasets.
- Add more fire-related classes (e.g., smoke detection).
- Integrate with real-time systems like drones, cameras, or satellites for fire monitoring.

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a new branch:
    ```sh
    git checkout -b feature-branch
    ```
3. Commit changes:
    ```sh
    git commit -m "Add feature"
    ```
4. Push to the branch:
    ```sh
    git push origin feature-branch
    ```
5. Open a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For feedback or inquiries, reach out via [GitHub Issues](https://github.com/Jay0073/Forest-Fire-Detection/issues) or contact [Jay0073](https://github.com/Jay0073).

---

Visit the project repository here: [Forest Fire Detection on GitHub](https://github.com/Jay0073/Forest-Fire-Detection)
