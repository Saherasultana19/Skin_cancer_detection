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

# ==================== UNet Segmentation Model ====================

def create_unet_model(img_size=(256, 256, 3)):
    """Create UNet model for skin lesion segmentation"""
    inputs = keras.Input(shape=img_size)
    
    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder
    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = layers.UpSampling2D((2, 2))(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = layers.UpSampling2D((2, 2))(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# ==================== ResNet Classification Model ====================

def create_resnet_classifier(num_classes=7, img_size=(256, 256, 3)):
    """Create ResNet50 model for multi-class classification"""
    base_model = ResNet50(input_shape=img_size, include_top=False, weights='imagenet')
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ==================== Grad-CAM Implementation ====================

class GradCAM:
    """Generate Grad-CAM visualization"""
    
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )
    
    def generate(self, img_array):
        """Generate Grad-CAM heatmap"""
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            class_channel = predictions[:, tf.argmax(predictions[0])]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()


def visualize_gradcam(original_img, heatmap, save_path='gradcam.png'):
    """Visualize Grad-CAM overlay"""
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Grad-CAM Heatmap')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed)
    plt.title('Overlay')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ==================== Data Preprocessing ====================

class ISICDataLoader:
    """Load and preprocess ISIC dataset"""
    
    def __init__(self, data_dir, img_size=(256, 256)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
    
    def load_and_preprocess(self):
        """Load images and labels from ISIC dataset"""
        images = []
        labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            for img_name in tqdm(os.listdir(class_dir), desc=f"Loading {class_name}"):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0
                    
                    images.append(img)
                    labels.append(self.class_to_idx[class_name])
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def get_segmentation_masks(self, images):
        """Generate or load segmentation masks"""
        masks = []
        for img in images:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
            masks.append(mask[..., np.newaxis])
        
        return np.array(masks)


# ==================== Training Pipeline ====================

class SkinCancerDetectionPipeline:
    """Complete pipeline for skin cancer detection"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_size = (256, 256, 3)
        self.num_classes = 7
        self.loader = ISICDataLoader(data_dir)
        
        self.unet_model = create_unet_model(self.img_size)
        self.classifier_model = create_resnet_classifier(self.num_classes, self.img_size)
    
    def train(self, epochs=50, batch_size=32):
        """Train both segmentation and classification models"""
        
        print("Loading ISIC dataset...")
        images, labels = self.loader.load_and_preprocess()
        masks = self.loader.get_segmentation_masks(images)
        
        # Train/test split
        X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
            images, labels, masks, test_size=0.2, random_state=42
        )
        
        # Convert labels to one-hot
        y_train_onehot = keras.utils.to_categorical(y_train, self.num_classes)
        y_test_onehot = keras.utils.to_categorical(y_test, self.num_classes)
        
        # Train UNet
        print("\n=== Training UNet Segmentation Model ===")
        self.unet_model.fit(
            X_train, m_train,
            validation_data=(X_test, m_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Train ResNet Classifier
        print("\n=== Training ResNet Classification Model ===")
        self.classifier_model.fit(
            X_train, y_train_onehot,
            validation_data=(X_test, y_test_onehot),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return X_test, y_test, m_test
    
    def predict(self, image):
        """Predict on a single image"""
        img = cv2.resize(image, (self.img_size[0], self.img_size[1]))
        img = img / 255.0
        img_batch = np.expand_dims(img, axis=0)
        
        # Segmentation
        segmentation = self.unet_model.predict(img_batch)[0]
        
        # Classification
        classification = self.classifier_model.predict(img_batch)[0]
        predicted_class = np.argmax(classification)
        confidence = classification[predicted_class]
        
        # Grad-CAM
        gradcam = GradCAM(self.classifier_model, 'conv5_block3_out')
        heatmap = gradcam.generate(img_batch)
        
        return {
            'segmentation': segmentation,
            'class_idx': predicted_class,
            'class_name': self.loader.class_names[predicted_class],
            'confidence': float(confidence),
            'heatmap': heatmap
        }
    
    def save_models(self, unet_path='models/unet.h5', classifier_path='models/classifier.h5'):
        """Save trained models"""
        os.makedirs('models', exist_ok=True)
        self.unet_model.save(unet_path)
        self.classifier_model.save(classifier_path)
        print(f"Models saved to {unet_path} and {classifier_path}")
    
    def load_models(self, unet_path='models/unet.h5', classifier_path='models/classifier.h5'):
        """Load pre-trained models"""
        self.unet_model = keras.models.load_model(unet_path)
        self.classifier_model = keras.models.load_model(classifier_path)
        print("Models loaded successfully")


# ==================== Main Execution ====================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = SkinCancerDetectionPipeline(data_dir='./ISIC_data')
    
    # Train models (uncomment to train)
    # X_test, y_test, m_test = pipeline.train(epochs=50, batch_size=32)
    
    # Or load pre-trained models
    # pipeline.load_models()
    
    # Make predictions on test image
    # test_image = cv2.imread('test_image.jpg')
    # results = pipeline.predict(test_image)
    # print(f"Prediction: {results['class_name']}")
    # print(f"Confidence: {results['confidence']:.2%}")
    # visualize_gradcam(test_image, results['heatmap'])
