"""
Complete 3D Skin Cancer Detection Pipeline with:
- UNet 3D Segmentation
- ResNet50 3D Classification
- Grad-CAM Visualization
- Ensemble Voting
- ISIC 3D Dataset Support (Kaggle)
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import seaborn as sns
import nibabel as nib
from scipy import ndimage
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==================== UNet 3D Segmentation Model ====================

def create_3d_unet_model(img_size=(128, 128, 64, 1)):
    """Create 3D UNet model for volumetric skin lesion segmentation"""
    inputs = keras.Input(shape=img_size)
    
    # Encoder Block 1
    c1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling3D((2, 2, 2))(c1)
    
    # Encoder Block 2
    c2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling3D((2, 2, 2))(c2)
    
    # Encoder Block 3
    c3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling3D((2, 2, 2))(c3)
    
    # Bottleneck
    c4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    
    # Decoder Block 1
    u5 = layers.UpSampling3D((2, 2, 2))(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(u5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    
    # Decoder Block 2
    u6 = layers.UpSampling3D((2, 2, 2))(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    
    # Decoder Block 3
    u7 = layers.UpSampling3D((2, 2, 2))(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)
    
    # Output Layer
    outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(c7)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', dice_coefficient]
    )
    
    return model


# ==================== 3D ResNet Classification Models ====================

def create_3d_resnet_classifier_v1(num_classes=7, img_size=(128, 128, 64, 3)):
    """Create 3D ResNet-like model for classification (Version 1)"""
    inputs = keras.Input(shape=img_size)
    
    # Conv blocks with residual connections
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_3d_resnet_classifier_v2(num_classes=7, img_size=(128, 128, 64, 3)):
    """Create 3D ResNet-like model for classification (Version 2 - Deeper)"""
    inputs = keras.Input(shape=img_size)
    
    # Block 1
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    # Block 2
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    # Block 3
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0008),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_3d_resnet_classifier_v3(num_classes=7, img_size=(128, 128, 64, 3)):
    """Create 3D ResNet-like model for classification (Version 3 - Lightweight)"""
    inputs = keras.Input(shape=img_size)
    
    x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ==================== Gradient-weighted Class Activation Mapping (Grad-CAM) ====================

class GradCAM3D:
    """Generate Grad-CAM visualization for 3D models"""
    
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )
    
    def generate(self, volume_batch):
        """Generate 3D Grad-CAM heatmap"""
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(volume_batch)
            class_channel = predictions[:, tf.argmax(predictions[0])]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()


def visualize_gradcam_3d(original_volume, heatmap, class_name, confidence, save_path='gradcam_3d.png'):
    """Visualize Grad-CAM for 3D volume (middle slice)"""
    mid_slice = original_volume.shape[2] // 2
    original_slice = original_volume[:, :, mid_slice, 0]
    heatmap_slice = heatmap[:, :, mid_slice]
    
    # Normalize for visualization
    original_slice = (original_slice - np.min(original_slice)) / (np.max(original_slice) - np.min(original_slice) + 1e-7)
    heatmap_slice = cv2.resize(heatmap_slice, (original_slice.shape[1], original_slice.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_slice), cv2.COLORMAP_JET)
    
    # Convert to RGB for display
    original_rgb = cv2.cvtColor((original_slice * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    superimposed = cv2.addWeighted(original_rgb, 0.6, heatmap_colored, 0.4, 0)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb, cmap='gray')
    plt.title('Original Volume Slice')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_colored)
    plt.title('Grad-CAM Heatmap')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed)
    plt.title(f'Overlay\n{class_name} ({confidence:.2%})')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Grad-CAM visualization saved: {save_path}")


# ==================== Ensemble Voting ====================

class EnsembleClassifier:
    """Ensemble voting classifier combining multiple 3D models"""
    
    def __init__(self, models, class_names):
        self.models = models
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def predict_ensemble(self, volume_batch, voting_method='soft'):
        """
        Predict using ensemble voting
        voting_method: 'soft' (average probabilities) or 'hard' (majority vote)
        """
        predictions_list = []
        
        for model in self.models:
            pred = model.predict(volume_batch, verbose=0)
            predictions_list.append(pred)
        
        predictions_array = np.array(predictions_list)  # Shape: (num_models, batch_size, num_classes)
        
        if voting_method == 'soft':
            # Average probabilities across models
            ensemble_pred = np.mean(predictions_array, axis=0)
        else:  # hard voting
            # Majority vote
            hard_preds = np.argmax(predictions_array, axis=-1)  # Shape: (num_models, batch_size)
            ensemble_pred = np.zeros((predictions_array.shape[1], self.num_classes))
            for i in range(predictions_array.shape[1]):
                votes = hard_preds[:, i]
                for vote in votes:
                    ensemble_pred[i, vote] += 1
                ensemble_pred[i] /= len(self.models)
        
        return ensemble_pred
    
    def predict_batch(self, volume_batch, voting_method='soft'):
        """Predict class for batch"""
        probs = self.predict_ensemble(volume_batch, voting_method)
        return np.argmax(probs, axis=1), np.max(probs, axis=1)


# ==================== Evaluation Metrics ====================

def dice_coefficient(y_true, y_pred):
    """Compute Dice coefficient"""
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    intersection = 2.0 * tf.reduce_sum(y_true * y_pred_binary)
    cardinality = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_binary)
    return intersection / (cardinality + 1e-7)


def compute_iou(y_true, y_pred, threshold=0.5):
    """Compute Intersection over Union"""
    y_pred_binary = (y_pred > threshold).astype(np.int32)
    intersection = np.sum(y_true * y_pred_binary)
    union = np.sum(y_true) + np.sum(y_pred_binary) - intersection
    return intersection / (union + 1e-7)


def compute_dice(y_true, y_pred, threshold=0.5):
    """Compute Dice coefficient"""
    y_pred_binary = (y_pred > threshold).astype(np.int32)
    intersection = 2.0 * np.sum(y_true * y_pred_binary)
    cardinality = np.sum(y_true) + np.sum(y_pred_binary)
    return intersection / (cardinality + 1e-7)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - 3D Skin Cancer Classification', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {save_path}")


def plot_classification_report(y_true, y_pred, class_names, save_path='classification_report.txt'):
    """Print and save classification report"""
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT - 3D SKIN CANCER DETECTION")
    print("="*80)
    print(report)
    
    with open(save_path, 'w') as f:
        f.write("CLASSIFICATION REPORT - 3D SKIN CANCER DETECTION\n")
        f.write("="*80 + "\n")
        f.write(report)
    print(f"✓ Classification report saved: {save_path}")


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # If Dice coefficient exists
    if 'dice_coefficient' in history:
        axes[1, 0].plot(history['dice_coefficient'], label='Training Dice', linewidth=2)
        axes[1, 0].plot(history['val_dice_coefficient'], label='Validation Dice', linewidth=2)
        axes[1, 0].set_title('Dice Coefficient', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Dice')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved: {save_path}")


# ==================== Data Loading and Preprocessing ====================

class NiftiLoader:
    """Load NIfTI 3D medical images"""
    
    @staticmethod
    def load_nifti(file_path, normalize=True):
        """Load NIfTI file and return volume"""
        try:
            nifti_img = nib.load(file_path)
            volume = nifti_img.get_fdata()
            
            if normalize:
                vol_min = np.min(volume)
                vol_max = np.max(volume)
                volume = (volume - vol_min) / (vol_max - vol_min + 1e-7)
            
            return volume, nifti_img
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    @staticmethod
    def save_nifti(volume, output_path, affine=None):
        """Save volume as NIfTI file"""
        if affine is None:
            affine = np.eye(4)
        img = nib.Nifti1Image(volume, affine)
        nib.save(img, output_path)
    
    @staticmethod
    def resize_3d_volume(volume, target_size=(128, 128, 64)):
        """Resize 3D volume to target size"""
        from scipy.ndimage import zoom
        current_size = volume.shape
        zoom_factors = [target_size[i] / current_size[i] for i in range(3)]
        return zoom(volume, zoom_factors, order=1)


class ISIC3DDataLoader:
    """Load and preprocess 3D ISIC dataset (NIfTI format from Kaggle)"""
    
    def __init__(self, data_dir, img_size=(128, 128, 64)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
    
    def load_and_preprocess_3d(self):
        """Load 3D NIfTI images and labels"""
        images = []
        labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue
            
            for vol_name in tqdm(os.listdir(class_dir), desc=f"Loading {class_name}"):
                if not (vol_name.endswith('.nii.gz') or vol_name.endswith('.nii')):
                    continue
                
                vol_path = os.path.join(class_dir, vol_name)
                try:
                    volume, _ = NiftiLoader.load_nifti(vol_path, normalize=True)
                    if volume is None:
                        continue
                    
                    # Resize to target size
                    volume = NiftiLoader.resize_3d_volume(volume, self.img_size)
                    
                    images.append(volume)
                    labels.append(self.class_to_idx[class_name])
                except Exception as e:
                    print(f"Error loading {vol_path}: {e}")
        
        print(f"\nLoaded {len(images)} volumes from {self.data_dir}")
        return np.array(images), np.array(labels)
    
    def get_segmentation_masks_3d(self, volumes):
        """Generate 3D segmentation masks using threshold"""
        masks = []
        for vol in tqdm(volumes, desc="Generating masks"):
            # Otsu's threshold for automatic mask generation
            threshold = np.percentile(vol, 25)
            mask = (vol > threshold).astype(np.float32)
            masks.append(mask[..., np.newaxis])
        
        return np.array(masks)


# ==================== Data Augmentation ====================

class DataAugmentation3D:
    """3D Data augmentation techniques"""
    
    @staticmethod
    def rotate_3d_volume(volume, angle, axis='z'):
        """Rotate 3D volume along specified axis"""
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        ax = axis_map.get(axis, 2)
        return ndimage.rotate(volume, angle, axes=(ax, (ax+1)%3), reshape=False)
    
    @staticmethod
    def flip_3d_volume(volume, axis=0):
        """Flip 3D volume along specified axis"""
        return np.flip(volume, axis=axis)
    
    @staticmethod
    def add_gaussian_noise(volume, noise_factor=0.05):
        """Add Gaussian noise to volume"""
        noise = np.random.normal(0, noise_factor, volume.shape)
        return np.clip(volume + noise, 0, 1)
    
    @staticmethod
    def elastic_deformation_3d(volume, alpha=30, sigma=3):
        """Apply elastic deformation to 3D volume"""
        shape = volume.shape
        dx = np.random.randn(*shape) * sigma
        dy = np.random.randn(*shape) * sigma
        dz = np.random.randn(*shape) * sigma
        
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = (
            np.reshape(y + dy * alpha, (-1, 1)),
            np.reshape(x + dx * alpha, (-1, 1)),
            np.reshape(z + dz * alpha, (-1, 1))
        )
        
        return ndimage.map_coordinates(volume, indices, order=1, mode='reflect').reshape(shape)


# ==================== Visualization ====================

class Volume3DVisualizer:
    """Visualize 3D medical volumes"""
    
    @staticmethod
    def save_slice_series(volume, output_dir='slices', plane='axial', step=2):
        """Save series of 2D slices from 3D volume"""
        os.makedirs(output_dir, exist_ok=True)
        
        if plane == 'axial':
            slices = [volume[:, :, i] for i in range(0, volume.shape[2], step)]
        elif plane == 'coronal':
            slices = [volume[:, i, :] for i in range(0, volume.shape[1], step)]
        elif plane == 'sagittal':
            slices = [volume[i, :, :] for i in range(0, volume.shape[0], step)]
        
        for idx, slice_data in enumerate(slices):
            plt.figure(figsize=(8, 8))
            plt.imshow(slice_data, cmap='gray')
            plt.title(f'{plane.capitalize()} Slice {idx}')
            plt.colorbar(label='Intensity')
            plt.savefig(f'{output_dir}/{plane}_{idx:03d}.png', dpi=100, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Saved {len(slices)} {plane} slices to {output_dir}")
    
    @staticmethod
    def create_3d_projection(volume, projection_type='max'):
        """Create 2D projection from 3D volume"""
        if projection_type == 'max':
            return np.max(volume, axis=0)
        elif projection_type == 'min':
            return np.min(volume, axis=0)
        elif projection_type == 'mean':
            return np.mean(volume, axis=0)
        elif projection_type == 'std':
            return np.std(volume, axis=0)
    
    @staticmethod
    def plot_volume_projections(volume, class_name, save_path='projections.png'):
        """Plot MIP, MIN, MEAN projections"""
        mip = Volume3DVisualizer.create_3d_projection(volume, 'max')
        minp = Volume3DVisualizer.create_3d_projection(volume, 'min')
        meanp = Volume3DVisualizer.create_3d_projection(volume, 'mean')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        axes[0, 0].imshow(volume[:, :, volume.shape[2]//2], cmap='gray')
        axes[0, 0].set_title('Middle Axial Slice')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mip, cmap='gray')
        axes[0, 1].set_title('Maximum Intensity Projection')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(meanp, cmap='gray')
        axes[1, 0].set_title('Mean Intensity Projection')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(minp, cmap='gray')
        axes[1, 1].set_title('Minimum Intensity Projection')
        axes[1, 1].axis('off')
        
        plt.suptitle(f'3D Volume Projections - {class_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Volume projections saved: {save_path}")


# ==================== Complete 3D Training Pipeline ====================

class SkinCancerDetection3DPipeline:
    """Complete pipeline for 3D volumetric skin cancer detection with ensemble voting"""
    
    def __init__(self, data_dir, img_size=(128, 128, 64)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.img_size_seg = (img_size[0], img_size[1], img_size[2], 1)
        self.img_size_class = (img_size[0], img_size[1], img_size[2], 3)
        self.num_classes = 7
        self.class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        
        self.loader = ISIC3DDataLoader(data_dir, img_size)
        
        # Initialize models
        self.unet_model = create_3d_unet_model(self.img_size_seg)
        
        # Create multiple classifier models for ensemble
        self.classifier_v1 = create_3d_resnet_classifier_v1(self.num_classes, self.img_size_class)
        self.classifier_v2 = create_3d_resnet_classifier_v2(self.num_classes, self.img_size_class)
        self.classifier_v3 = create_3d_resnet_classifier_v3(self.num_classes, self.img_size_class)
        
        # Ensemble classifier
        self.ensemble = EnsembleClassifier(
            [self.classifier_v1, self.classifier_v2, self.classifier_v3],
            self.class_names
        )
    
    def train(self, epochs=50, batch_size=4, augment=True):
        """Train UNet segmentation and ensemble classification models"""
        
        print("\n" + "="*80)
        print("LOADING 3D ISIC DATASET")
        print("="*80)
        images, labels = self.loader.load_and_preprocess_3d()
        masks = self.loader.get_segmentation_masks_3d(images)
        
        if len(images) == 0:
            raise ValueError("No images loaded! Check your data directory path.")
        
        # Add channel dimension to images
        images = np.expand_dims(images, axis=-1)
        
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Train/test split
        X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
            images, labels, masks, test_size=0.2, random_state=42
        )
        
        # Convert labels to one-hot
        y_train_onehot = keras.utils.to_categorical(y_train, self.num_classes)
        y_test_onehot = keras.utils.to_categorical(y_test, self.num_classes)
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Train UNet
        print("\n" + "="*80)
        print("TRAINING 3D UNET SEGMENTATION MODEL")
        print("="*80)
        
        history_unet = self.unet_model.fit(
            X_train, m_train,
            validation_data=(X_test, m_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Prepare data for classifiers (repeat channels)
        X_train_3ch = np.repeat(X_train, 3, axis=-1)
        X_test_3ch = np.repeat(X_test, 3, axis=-1)
        
        # Train Classifier V1
        print("\n" + "="*80)
        print("TRAINING 3D RESNET CLASSIFIER V1")
        print("="*80)
        
        history_v1 = self.classifier_v1.fit(
            X_train_3ch, y_train_onehot,
            validation_data=(X_test_3ch, y_test_onehot),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Train Classifier V2
        print("\n" + "="*80)
        print("TRAINING 3D RESNET CLASSIFIER V2")
        print("="*80)
        
        history_v2 = self.classifier_v2.fit(
            X_train_3ch, y_train_onehot,
            validation_data=(X_test_3ch, y_test_onehot),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Train Classifier V3
        print("\n" + "="*80)
        print("TRAINING 3D RESNET CLASSIFIER V3")
        print("="*80)
        
        history_v3 = self.classifier_v3.fit(
            X_train_3ch, y_train_onehot,
            validation_data=(X_test_3ch, y_test_onehot),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Save training histories
        plot_training_history(history_unet.history, 'unet_training_history.png')
        
        return X_test, X_test_3ch, y_test, y_test_onehot, m_test, (history_unet, history_v1, history_v2, history_v3)
    
    def evaluate_ensemble(self, X_test, X_test_3ch, y_test):
        """Evaluate ensemble predictions"""
        print("\n" + "="*80)
        print("ENSEMBLE EVALUATION")
        print("="*80)
        
        # Soft voting
        y_pred_soft, confidence_soft = self.ensemble.predict_batch(X_test_3ch, voting_method='soft')
        
        # Hard voting
        y_pred_hard, confidence_hard = self.ensemble.predict_batch(X_test_3ch, voting_method='hard')
        
        # Calculate accuracy
        accuracy_soft = np.mean(y_pred_soft == y_test)
        accuracy_hard = np.mean(y_pred_hard == y_test)
        
        print(f"\nSoft Voting Accuracy: {accuracy_soft:.4f} ({accuracy_soft*100:.2f}%)")
        print(f"Hard Voting Accuracy: {accuracy_hard:.4f} ({accuracy_hard*100:.2f}%)")
        
        return y_pred_soft, y_pred_hard, confidence_soft, confidence_hard
    
    def predict_3d(self, volume):
        """Predict on a single 3D volume"""
        volume = NiftiLoader.resize_3d_volume(volume, self.img_size)
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-7)
        
        volume_batch = np.expand_dims(volume, axis=(0, -1))
        
        # Segmentation
        segmentation = self.unet_model.predict(volume_batch, verbose=0)[0]
        
        # Classification (ensemble)
        volume_3ch = np.repeat(volume_batch, 3, axis=-1)
        y_pred_soft, conf_soft = self.ensemble.predict_batch(volume_3ch, voting_method='soft')
        y_pred_hard, conf_hard = self.ensemble.predict_batch(volume_3ch, voting_method='hard')
        
        predicted_class_soft = y_pred_soft[0]
        predicted_class_hard = y_pred_hard[0]
        
        # Grad-CAM from classifier_v1
        gradcam = GradCAM3D(self.classifier_v1, 'conv3d_5')
        heatmap = gradcam.generate(volume_3ch)
        
        return {
            'segmentation': segmentation,
            'class_soft': predicted_class_soft,
            'class_hard': predicted_class_hard,
            'class_name': self.class_names[predicted_class_soft],
            'confidence_soft': float(conf_soft[0]),
            'confidence_hard': float(conf_hard[0]),
            'heatmap': heatmap
        }
    
    def save_models(self, models_dir='models'):
        """Save trained models"""
        os.makedirs(models_dir, exist_ok=True)
        
        self.unet_model.save(os.path.join(models_dir, 'unet_3d.h5'))
        self.classifier_v1.save(os.path.join(models_dir, 'classifier_v1.h5'))
        self.classifier_v2.save(os.path.join(models_dir, 'classifier_v2.h5'))
        self.classifier_v3.save(os.path.join(models_dir, 'classifier_v3.h5'))
        
        print(f"\n✓ All models saved to {models_dir}/")
    
    def load_models(self, models_dir='models'):
        """Load pre-trained models"""
        custom_objects = {'dice_coefficient': dice_coefficient}
        
        self.unet_model = keras.models.load_model(
            os.path.join(models_dir, 'unet_3d.h5'),
            custom_objects=custom_objects
        )
        self.classifier_v1 = keras.models.load_model(os.path.join(models_dir, 'classifier_v1.h5'))
        self.classifier_v2 = keras.models.load_model(os.path.join(models_dir, 'classifier_v2.h5'))
        self.classifier_v3 = keras.models.load_model(os.path.join(models_dir, 'classifier_v3.h5'))
        
        self.ensemble = EnsembleClassifier(
            [self.classifier_v1, self.classifier_v2, self.classifier_v3],
            self.class_names
        )
        
        print("✓ All models loaded successfully")


# ==================== Main Execution ====================

def main():
    """Main execution function"""
    
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*15 + "3D SKIN CANCER DETECTION PIPELINE" + " "*30 + "█")
    print("█" + " "*10 + "UNet Segmentation + ResNet Classification + Grad-CAM" + " "*17 + "█")
    print("█" + " "*15 + "+ Ensemble Voting from Kaggle ISIC Dataset" + " "*23 + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")
    
    # Configuration
    DATA_DIR = './ISIC_3D_data'  # Path to downloaded ISIC 3D data from Kaggle
    EPOCHS = 50
    BATCH_SIZE = 4
    IMG_SIZE = (128, 128, 64)
    
    # Create pipeline
    pipeline = SkinCancerDetection3DPipeline(data_dir=DATA_DIR, img_size=IMG_SIZE)
    
    # Train models
    print("\n[1/4] Starting training pipeline...")
    X_test, X_test_3ch, y_test, y_test_onehot, m_test, histories = pipeline.train(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Evaluate ensemble
    print("\n[2/4] Evaluating ensemble predictions...")
    y_pred_soft, y_pred_hard, conf_soft, conf_hard = pipeline.evaluate_ensemble(X_test, X_test_3ch, y_test)
    
    # Generate visualizations
    print("\n[3/4] Generating visualizations...")
    
    plot_confusion_matrix(y_test, y_pred_soft, pipeline.class_names, 'confusion_matrix_soft.png')
    plot_confusion_matrix(y_test, y_pred_hard, pipeline.class_names, 'confusion_matrix_hard.png')
    plot_classification_report(y_test, y_pred_soft, pipeline.class_names, 'classification_report.txt')
    
    # Save models
    print("\n[4/4] Saving models...")
    pipeline.save_models()
    
    # Test on sample images
    print("\n" + "="*80)
    print("TESTING ON SAMPLE VOLUMES")
    print("="*80)
    
    for i in range(min(3, len(X_test))):
        print(f"\nSample {i+1}:")
        test_volume = X_test[i]
        results = pipeline.predict_3d(test_volume[:, :, :, 0])
        
        print(f"  Soft Vote: {results['class_name']} ({results['confidence_soft']:.2%})")
        print(f"  Hard Vote: {pipeline.class_names[results['class_hard']]} ({results['confidence_hard']:.2%})")
        print(f"  Ground Truth: {pipeline.class_names[y_test[i]]}")
        print(f"  Segmentation Shape: {results['segmentation'].shape}")
        
        # Visualize
        visualize_gradcam_3d(
            test_volume,
            results['heatmap'],
            results['class_name'],
            results['confidence_soft'],
            f'gradcam_sample_{i+1}.png'
        )
        
        Volume3DVisualizer.plot_volume_projections(
            test_volume[:, :, :, 0],
            results['class_name'],
            f'projections_sample_{i+1}.png'
        )
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated Files:")
    print("  ✓ confusion_matrix_soft.png")
    print("  ✓ confusion_matrix_hard.png")
    print("  ✓ classification_report.txt")
    print("  ✓ gradcam_sample_*.png")
    print("  ✓ projections_sample_*.png")
    print("  ✓ models/unet_3d.h5")
    print("  ✓ models/classifier_v1.h5")
    print("  ✓ models/classifier_v2.h5")
    print("  ✓ models/classifier_v3.h5\n")


if __name__ == "__main__":
    main()
