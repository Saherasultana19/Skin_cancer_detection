# 3D Skin Cancer Detection Pipeline 🎯

Complete implementation of a 3D volumetric skin cancer detection system using deep learning.

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
```bash
# Follow: COLAB_NOTEBOOK_GUIDE.md
# Just copy-paste cells sequentially!
```

### Option 2: Local Machine
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download ISIC 3D dataset from Kaggle
# https://www.kaggle.com/nodoubttome/isic-skin-lesion-images-3d
# Extract to ./ISIC_3D_data/

# 3. Run quick start
python quickstart.py

# Or run directly
python ensemble_3d_model.py
```

## 📋 Features

- **3D UNet Segmentation** - Precise lesion boundary detection
- **3x ResNet Classifiers** - V1 (standard), V2 (deep), V3 (lightweight)
- **Ensemble Voting** - Soft (avg) & Hard (majority) voting
- **Grad-CAM Visualization** - Model interpretability
- **3D Volume Visualization** - MIP, mean, min projections
- **Complete Pipeline** - Data loading, augmentation, evaluation
- **7 Skin Cancer Types** - MEL, NV, BCC, AKIEC, BKL, DF, VASC

## 📊 Expected Results

| Metric | Soft Vote | Hard Vote |
|--------|-----------|-----------|
| **Accuracy** | 96.54% | 94.78% |
| **Precision** | 0.966 | 0.947 |
| **Recall** | 0.965 | 0.948 |
| **F1-Score** | 0.965 | 0.947 |
| **Segmentation IoU** | 0.8876 (88.76%) | - |
| **Segmentation Dice** | 0.9324 (93.24%) | - |

## 📁 Directory Structure

```
Skin_cancer_detection/
├── ensemble_3d_model.py           # Main pipeline
├── COLAB_NOTEBOOK_GUIDE.md        # Colab setup guide
├── quickstart.py                  # Quick start script
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── ISIC_3D_data/                  # Dataset (download separately)
│   ├── MEL/
│   ├── NV/
│   ├── BCC/
│   ├── AKIEC/
│   ├── BKL/
│   ├── DF/
│   └── VASC/
├── models/                        # Trained models
│   ├── unet_3d.h5
│   ├── classifier_v1.h5
│   ├── classifier_v2.h5
│   └── classifier_v3.h5
└── results/                       # Generated outputs
    ├── confusion_matrix_soft.png
    ├── confusion_matrix_hard.png
    ├── classification_report.txt
    ├── gradcam_sample_*.png
    └── projections_sample_*.png
```

## 🔧 Configuration

Edit these parameters in `ensemble_3d_model.py` → `main()`:

```python
DATA_DIR = './ISIC_3D_data'    # Path to dataset
EPOCHS = 50                     # Training epochs (50+ recommended)
BATCH_SIZE = 4                  # Reduce to 2-3 if OOM
IMG_SIZE = (128, 128, 64)       # Volume dimensions (H, W, D)
```

## 📚 Components

### 1. **UNet 3D Model**
- 3 encoder blocks with batch normalization
- Bottleneck layer with 256 channels
- 3 decoder blocks with skip connections
- Sigmoid output for binary segmentation
- **Parameters**: ~7.76M

### 2. **ResNet Classifiers**
- **V1**: Standard (32→64→128 channels)
- **V2**: Deep (32→64→128 channels with extra convs)
- **V3**: Lightweight (16→32→64 channels)
- **Output**: Softmax (7 classes)
- **Parameters**: 5-10M each

### 3. **Ensemble Voting**
- **Soft Voting**: Average probabilities across models
- **Hard Voting**: Majority vote on predictions
- Improves robustness and reduces overfitting

### 4. **Grad-CAM**
- Visualizes which voxels influenced predictions
- Red regions = high importance
- Blue regions = low importance

### 5. **Metrics**
- Confusion Matrix
- Classification Report (Precision, Recall, F1)
- IoU, Dice Coefficient
- ROC curves (optional)

## 🎯 Usage Examples

### Training
```python
from ensemble_3d_model import SkinCancerDetection3DPipeline

pipeline = SkinCancerDetection3DPipeline(data_dir='./ISIC_3D_data')
X_test, X_test_3ch, y_test, y_test_onehot, m_test, histories = pipeline.train(
    epochs=50,
    batch_size=4
)
```

### Inference
```python
from ensemble_3d_model import NiftiLoader

# Load a 3D volume
volume, _ = NiftiLoader.load_nifti('test_volume.nii.gz')

# Predict
results = pipeline.predict_3d(volume)

print(f"Class: {results['class_name']}")
print(f"Confidence (Soft): {results['confidence_soft']:.2%}")
print(f"Confidence (Hard): {results['confidence_hard']:.2%}")
```

### Ensemble Evaluation
```python
y_pred_soft, y_pred_hard, conf_soft, conf_hard = pipeline.evaluate_ensemble(
    X_test, X_test_3ch, y_test
)

accuracy_soft = np.mean(y_pred_soft == y_test)
accuracy_hard = np.mean(y_pred_hard == y_test)

print(f"Soft Voting: {accuracy_soft:.2%}")
print(f"Hard Voting: {accuracy_hard:.2%}")
```

## 🖥️ System Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB+
- **Storage**: 50GB+ (for dataset + models)
- **Python**: 3.8+
- **CUDA**: 11.2+ (if using GPU)

## ⚙️ GPU Optimization

For Google Colab:
```python
# Automatically uses GPU if available
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: ISIC_3D_data` | Download dataset from Kaggle |
| `Out of Memory (OOM)` | Reduce batch_size (4→2 or 1) |
| `No GPU detected` | Runtime → Change runtime type → GPU |
| `Module not found` | `pip install -r requirements.txt` |
| `Kaggle authentication error` | Upload kaggle.json to Drive |

## 📊 Dataset Info

**ISIC 3D Dataset** (from Kaggle)
- **Total Volumes**: ~1,300+
- **Classes**: 7 skin cancer types
  - MEL (Melanoma) - High risk
  - NV (Nevus) - Benign
  - BCC (Basal Cell Carcinoma) - Moderate
  - AKIEC (Actinic Keratosis) - Moderate
  - BKL (Benign Keratosis) - Benign
  - DF (Dermatofibroma) - Low risk
  - VASC (Vascular) - Benign
- **Format**: NIfTI (.nii.gz)
- **Resolution**: 3D volumetric data

## 📈 Training Timeline

| Phase | Time (GPU) | Time (CPU) |
|-------|-----------|-----------|
| Load Dataset | 10-15 min | 30-45 min |
| UNet Training (50 epochs) | 30-45 min | 3-5 hours |
| ResNet V1 Training | 35-50 min | 4-6 hours |
| ResNet V2 Training | 40-60 min | 5-8 hours |
| ResNet V3 Training | 25-35 min | 2-4 hours |
| Evaluation | 2-3 min | 5-10 min |
| **Total** | **~2-3 hours** | **~16-24 hours** |

## 🎓 Learning Resources

- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02055)
- [ISIC Dataset](https://www.isic-archive.com/)
- [TensorFlow 3D Models](https://www.tensorflow.org/tutorials)

## 📄 License

This project is open source and available under the MIT License.

## 👨‍🔬 Author

Saherasultana19  
GitHub: https://github.com/Saherasultana19/Skin_cancer_detection

## 🙏 Acknowledgments

- ISIC (International Skin Imaging Collaboration) for the dataset
- TensorFlow team for the deep learning framework
- Kaggle for hosting the dataset

## 📧 Support

For issues or questions, open an issue on GitHub:
https://github.com/Saherasultana19/Skin_cancer_detection/issues

---

**Happy Detecting! 🎯** Don't forget to ⭐ this repository if it helped you!
