# Installation and Setup Guide

## Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 8GB+ RAM

## Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/Saherasultana19/Skin_cancer_detection.git
cd Skin_cancer_detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download ISIC Dataset**
- Visit: https://www.isic-archive.com/
- Download the ISIC 2019 or 2020 dataset
- Organize data in the following structure:
```
ISIC_data/
├── MEL/
├── NV/
├── BCC/
├── AKIEC/
├── BKL/
├── DF/
└── VASC/
```

## Usage

### Training
```python
from main import SkinCancerDetectionPipeline

pipeline = SkinCancerDetectionPipeline(data_dir='./ISIC_data')
pipeline.train(epochs=50, batch_size=32)
pipeline.save_models()
```

### Inference
```python
import cv2
from main import SkinCancerDetectionPipeline

pipeline = SkinCancerDetectionPipeline(data_dir='./ISIC_data')
pipeline.load_models()

image = cv2.imread('skin_lesion.jpg')
results = pipeline.predict(image)

print(f"Class: {results['class_name']}")
print(f"Confidence: {results['confidence']:.2%}")
```

## Model Details

- **UNet**: For segmentation (256x256 input)
- **ResNet50**: For 7-class classification (MEL, NV, BCC, AKIEC, BKL, DF, VASC)
- **Grad-CAM**: For explainability and visualization

## Classes
- MEL: Melanoma
- NV: Nevus
- BCC: Basal Cell Carcinoma
- AKIEC: Actinic Keratosis
- BKL: Benign Keratosis-like Lesion
- DF: Dermatofibroma
- VASC: Vascular Lesion
