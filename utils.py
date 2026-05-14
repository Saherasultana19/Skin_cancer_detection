import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import nibabel as nib
from scipy import ndimage

class EvaluationMetrics:
    """Evaluate model performance"""
    
    @staticmethod
    def compute_iou(y_true, y_pred, threshold=0.5):
        """Compute Intersection over Union for segmentation (2D/3D)"""
        y_pred_binary = (y_pred > threshold).astype(np.int32)
        intersection = np.sum(y_true * y_pred_binary)
        union = np.sum(y_true) + np.sum(y_pred_binary) - intersection
        return intersection / (union + 1e-7)
    
    @staticmethod
    def compute_dice(y_true, y_pred, threshold=0.5):
        """Compute Dice coefficient (2D/3D)"""
        y_pred_binary = (y_pred > threshold).astype(np.int32)
        intersection = 2.0 * np.sum(y_true * y_pred_binary)
        cardinality = np.sum(y_true) + np.sum(y_pred_binary)
        return intersection / (cardinality + 1e-7)
    
    @staticmethod
    def compute_hausdorff_distance(y_true, y_pred, threshold=0.5):
        """Compute Hausdorff distance for 3D segmentation"""
        y_pred_binary = (y_pred > threshold).astype(np.int32)
        # Distance transform
        dist_true = ndimage.distance_transform_edt(~y_true.astype(bool))
        dist_pred = ndimage.distance_transform_edt(~y_pred_binary.astype(bool))
        # Hausdorff distance
        return max(np.max(np.min(dist_pred[y_true > 0], axis=0)),
                   np.max(np.min(dist_true[y_pred_binary > 0], axis=0)))
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
    
    @staticmethod
    def print_classification_report(y_true, y_pred, class_names):
        """Print detailed classification report"""
        print(classification_report(y_true, y_pred, target_names=class_names))


class DataAugmentation:
    """Data augmentation techniques for 2D and 3D images"""
    
    @staticmethod
    def rotate_image(image, angle):
        """Rotate 2D image"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    @staticmethod
    def rotate_3d_volume(volume, angle, axis='z'):
        """Rotate 3D volume along specified axis (x, y, or z)"""
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        ax = axis_map.get(axis, 2)
        return ndimage.rotate(volume, angle, axes=(ax, (ax+1)%3), reshape=False)
    
    @staticmethod
    def flip_image(image, direction='horizontal'):
        """Flip 2D image"""
        if direction == 'horizontal':
            return cv2.flip(image, 1)
        elif direction == 'vertical':
            return cv2.flip(image, 0)
        return image
    
    @staticmethod
    def flip_3d_volume(volume, axis=0):
        """Flip 3D volume along specified axis"""
        return np.flip(volume, axis=axis)
    
    @staticmethod
    def add_noise(image, noise_factor=0.1):
        """Add Gaussian noise to 2D/3D image"""
        noise = np.random.normal(0, noise_factor, image.shape)
        return np.clip(image + noise, 0, 1)
    
    @staticmethod
    def adjust_brightness(image, factor):
        """Adjust brightness for 2D image"""
        if image.ndim == 2:
            return np.clip(image * factor, 0, 1)
        else:
            hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            v = np.clip(v * factor, 0, 255).astype(np.uint8)
            hsv = cv2.merge((h, s, v))
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.0
    
    @staticmethod
    def elastic_deformation_3d(volume, alpha=30, sigma=3):
        """Apply elastic deformation to 3D volume for augmentation"""
        shape = volume.shape
        dx = np.random.randn(*shape) * sigma
        dy = np.random.randn(*shape) * sigma
        dz = np.random.randn(*shape) * sigma
        
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy * alpha, (-1, 1)), \
                  np.reshape(x + dx * alpha, (-1, 1)), \
                  np.reshape(z + dz * alpha, (-1, 1))
        
        return ndimage.map_coordinates(volume, indices, order=1, mode='reflect').reshape(shape)


class Volume3DVisualizer:
    """Visualize 3D medical volumes"""
    
    @staticmethod
    def save_slice_series(volume, output_dir='slices', plane='axial', step=1):
        """Save series of 2D slices from 3D volume"""
        import os
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
            plt.colorbar()
            plt.savefig(f'{output_dir}/{plane}_{idx:03d}.png')
            plt.close()
    
    @staticmethod
    def create_3d_projection(volume, projection_type='max'):
        """Create 2D projection from 3D volume (MIP, MIN, MEAN)"""
        if projection_type == 'max':
            return np.max(volume, axis=0)
        elif projection_type == 'min':
            return np.min(volume, axis=0)
        elif projection_type == 'mean':
            return np.mean(volume, axis=0)
        elif projection_type == 'std':
            return np.std(volume, axis=0)


class NiftiLoader:
    """Load NIfTI 3D medical images"""
    
    @staticmethod
    def load_nifti(file_path, normalize=True):
        """Load NIfTI file and return volume"""
        nifti_img = nib.load(file_path)
        volume = nifti_img.get_fdata()
        
        if normalize:
            volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-7)
        
        return volume, nifti_img
    
    @staticmethod
    def save_nifti(volume, output_path, affine=None):
        """Save volume as NIfTI file"""
        if affine is None:
            affine = np.eye(4)
        img = nib.Nifti1Image(volume, affine)
        nib.save(img, output_path)
    
    @staticmethod
    def resize_3d_volume(volume, target_size=(256, 256, 256)):
        """Resize 3D volume to target size"""
        from scipy.ndimage import zoom
        current_size = volume.shape
        zoom_factors = [target_size[i] / current_size[i] for i in range(3)]
        return zoom(volume, zoom_factors, order=1)


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

# ==================== 3D UNet Segmentation Model ====================

def create_3d_unet_model(img_size=(128, 128, 64, 1)):
    """Create 3D UNet model for volumetric skin lesion segmentation"""
    inputs = keras.Input(shape=img_size)
    
    # Encoder
    c1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling3D((2, 2, 2))(c1)
    
    c2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling3D((2, 2, 2))(c2)
    
    c3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling3D((2, 2, 2))(c3)
    
    # Bottleneck
    c4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(c4)
    
    # Decoder
    u5 = layers.UpSampling3D((2, 2, 2))(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(c5)
    
    u6 = layers.UpSampling3D((2, 2, 2))(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.UpSampling3D((2, 2, 2))(u6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(c7)
    
    outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(c7)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def create_3d_resnet_classifier(num_classes=7, img_size=(128, 128, 64, 3)):
    """Create 3D ResNet-like model for volumetric classification"""
    inputs = keras.Input(shape=img_size)
    
    # 3D Convolutional blocks
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    # Global pooling and dense layers
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


# ==================== Grad-CAM for 3D ====================

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


# ==================== Data Preprocessing for 3D ====================

class ISIC3DDataLoader:
    """Load and preprocess 3D ISIC dataset (NIfTI format)"""
    
    def __init__(self, data_dir, img_size=(128, 128, 64)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
    
    def load_and_preprocess_3d(self):
        """Load 3D NIfTI images and labels"""
        images = []
        labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            for vol_name in tqdm(os.listdir(class_dir), desc=f"Loading {class_name}"):
                if not vol_name.endswith('.nii.gz') and not vol_name.endswith('.nii'):
                    continue
                
                vol_path = os.path.join(class_dir, vol_name)
                try:
                    volume, _ = NiftiLoader.load_nifti(vol_path, normalize=True)
                    volume = NiftiLoader.resize_3d_volume(volume, self.img_size)
                    
                    images.append(volume)
                    labels.append(self.class_to_idx[class_name])
                except Exception as e:
                    print(f"Error loading {vol_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def get_segmentation_masks_3d(self, volumes):
        """Generate or load 3D segmentation masks"""
        masks = []
        for vol in volumes:
            # Threshold-based mask
            mask = (vol > np.percentile(vol, 25)).astype(np.float32)
            masks.append(mask[..., np.newaxis])
        
        return np.array(masks)


# ==================== 3D Training Pipeline ====================

class SkinCancerDetection3DPipeline:
    """Complete pipeline for 3D volumetric skin cancer detection"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_size = (128, 128, 64, 1)
        self.img_size_class = (128, 128, 64, 3)
        self.num_classes = 7
        self.loader = ISIC3DDataLoader(data_dir)
        
        self.unet_model = create_3d_unet_model(self.img_size)
        self.classifier_model = create_3d_resnet_classifier(self.num_classes, self.img_size_class)
    
    def train(self, epochs=50, batch_size=8):
        """Train both 3D segmentation and classification models"""
        
        print("Loading 3D ISIC dataset...")
        images, labels = self.loader.load_and_preprocess_3d()
        masks = self.loader.get_segmentation_masks_3d(images)
        
        # Add channel dimension
        images = np.expand_dims(images, axis=-1)
        
        # Train/test split
        X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
            images, labels, masks, test_size=0.2, random_state=42
        )
        
        # Convert labels to one-hot
        y_train_onehot = keras.utils.to_categorical(y_train, self.num_classes)
        y_test_onehot = keras.utils.to_categorical(y_test, self.num_classes)
        
        # Train 3D UNet
        print("\n=== Training 3D UNet Segmentation Model ===")
        self.unet_model.fit(
            X_train, m_train,
            validation_data=(X_test, m_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Train 3D Classifier
        print("\n=== Training 3D Classification Model ===")
        # Repeat channels for classifier (simulate RGB)
        X_train_3ch = np.repeat(X_train, 3, axis=-1)
        X_test_3ch = np.repeat(X_test, 3, axis=-1)
        
        self.classifier_model.fit(
            X_train_3ch, y_train_onehot,
            validation_data=(X_test_3ch, y_test_onehot),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return X_test, y_test, m_test
    
    def predict_3d(self, volume):
        """Predict on a single 3D volume"""
        volume = NiftiLoader.resize_3d_volume(volume, self.img_size[:-1])
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-7)
        volume_batch = np.expand_dims(volume, axis=(0, -1))
        
        # Segmentation
        segmentation = self.unet_model.predict(volume_batch)[0]
        
        # Classification
        volume_3ch = np.repeat(volume_batch, 3, axis=-1)
        classification = self.classifier_model.predict(volume_3ch)[0]
        predicted_class = np.argmax(classification)
        confidence = classification[predicted_class]
        
        return {
            'segmentation': segmentation,
            'class_idx': predicted_class,
            'class_name': self.loader.class_names[predicted_class],
            'confidence': float(confidence)
        }
    
    def save_models(self, unet_path='models/unet_3d.h5', classifier_path='models/classifier_3d.h5'):
        """Save trained 3D models"""
        os.makedirs('models', exist_ok=True)
        self.unet_model.save(unet_path)
        self.classifier_model.save(classifier_path)
        print(f"3D models saved to {unet_path} and {classifier_path}")
    
    def load_models(self, unet_path='models/unet_3d.h5', classifier_path='models/classifier_3d.h5'):
        """Load pre-trained 3D models"""
        self.unet_model = keras.models.load_model(unet_path)
        self.classifier_model = keras.models.load_model(classifier_path)
        print("3D models loaded successfully")


# ==================== Main Execution ====================

if __name__ == "__main__":
    # Initialize 3D pipeline
    pipeline = SkinCancerDetection3DPipeline(data_dir='./ISIC_3D_data')
    
    # Train 3D models (uncomment to train)
    # X_test, y_test, m_test = pipeline.train(epochs=50, batch_size=8)
    
    # Or load pre-trained models
    # pipeline.load_models()
    
    # Make predictions on test 3D volume
    # volume, _ = NiftiLoader.load_nifti('test_volume.nii.gz')
    # results = pipeline.predict_3d(volume)
    # print(f"Prediction: {results['class_name']}")
    # print(f"Confidence: {results['confidence']:.2%}")
    # 
    # # Visualize slices
    # visualizer = Volume3DVisualizer()
    # visualizer.save_slice_series(results['segmentation'], plane='axial')
