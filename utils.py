import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

class EvaluationMetrics:
    """Evaluate model performance"""
    
    @staticmethod
    def compute_iou(y_true, y_pred, threshold=0.5):
        """Compute Intersection over Union for segmentation"""
        y_pred_binary = (y_pred > threshold).astype(np.int32)
        intersection = np.sum(y_true * y_pred_binary)
        union = np.sum(y_true) + np.sum(y_pred_binary) - intersection
        return intersection / (union + 1e-7)
    
    @staticmethod
    def compute_dice(y_true, y_pred, threshold=0.5):
        """Compute Dice coefficient"""
        y_pred_binary = (y_pred > threshold).astype(np.int32)
        intersection = 2.0 * np.sum(y_true * y_pred_binary)
        cardinality = np.sum(y_true) + np.sum(y_pred_binary)
        return intersection / (cardinality + 1e-7)
    
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
    """Data augmentation techniques"""
    
    @staticmethod
    def rotate_image(image, angle):
        """Rotate image"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    @staticmethod
    def flip_image(image, direction='horizontal'):
        """Flip image"""
        if direction == 'horizontal':
            return cv2.flip(image, 1)
        elif direction == 'vertical':
            return cv2.flip(image, 0)
        return image
    
    @staticmethod
    def add_noise(image, noise_factor=0.1):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_factor, image.shape)
        return np.clip(image + noise, 0, 1)
    
    @staticmethod
    def adjust_brightness(image, factor):
        """Adjust brightness"""
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v = np.clip(v * factor, 0, 255).astype(np.uint8)
        hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.0
