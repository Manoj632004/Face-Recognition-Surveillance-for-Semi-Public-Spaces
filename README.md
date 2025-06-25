# Face-Recognition-Surveillance-for-Semi-Public-Spaces

## Setup Instructions

### 1. Download Required Models

- **Face Detection 0200**  
  [Download from OpenVINO Model Zoo](https://docs.openvino.ai/2023.3/omz_models_model_face_detection_0200.html)

- **Landmarks Regression Retail 0009**  
  [Download from OpenVINO Model Zoo](https://docs.openvino.ai/2023.3/omz_models_model_landmarks_regression_retail_0009.html#outputs)

- **Face Recognition ResNet100 ArcFace (ONNX)**  
  [Download and convert to OpenVINO IR format](https://docs.openvino.ai/2023.3/omz_models_model_face_recognition_resnet100_arcface_onnx.html#download-a-model-and-convert-it-into-openvino-ir-format)

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```
### 3. Run the python file
```bash
python retina.py
```

