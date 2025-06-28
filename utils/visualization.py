import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from sklearn.metrics import confusion_matrix, classification_report

class VisualizationUtils:
    def __init__(self, config):
        self.config = config
        
    def plot_training_history(self, history_path=None, history_dict=None, save_path=None):
        """Plot training history from file or dictionary"""
        
        if history_dict is None:
            if history_path is None:
                history_path = os.path.join(self.config.RESULTS_DIR, 'training_history.json')
            
            with open(history_path, 'r') as f:
                training_info = json.load(f)
                history_dict = training_info['history']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        if 'accuracy' in history_dict and 'val_accuracy' in history_dict:
            axes[0, 0].plot(history_dict['accuracy'], label='Training Accuracy', marker='o')
            axes[0, 0].plot(history_dict['val_accuracy'], label='Validation Accuracy', marker='s')
            axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        if 'loss' in history_dict and 'val_loss' in history_dict:
            axes[0, 1].plot(history_dict['loss'], label='Training Loss', marker='o')
            axes[0, 1].plot(history_dict['val_loss'], label='Validation Loss', marker='s')
            axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in history_dict and 'val_precision' in history_dict:
            axes[1, 0].plot(history_dict['precision'], label='Training Precision', marker='o')
            axes[1, 0].plot(history_dict['val_precision'], label='Validation Precision', marker='s')
            axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in history_dict and 'val_recall' in history_dict:
            axes[1, 1].plot(history_dict['recall'], label='Training Recall', marker='o')
            axes[1, 1].plot(history_dict['val_recall'], label='Validation Recall', marker='s')
            axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
            return fig
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=['Normal', 'Pneumonia'], 
                             save_path=None, normalize=False):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return plt.gcf()
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """Plot ROC curve"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        return plt.gcf(), roc_auc
    
    def plot_sample_predictions(self, images, true_labels, predictions, 
                               class_names=['Normal', 'Pneumonia'], 
                               num_samples=8, save_path=None):
        """Plot sample predictions with images"""
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            # Display image
            axes[i].imshow(images[i], cmap='gray')
            
            # Create title with prediction info
            true_class = class_names[int(true_labels[i])]
            pred_class = class_names[int(predictions[i] > 0.5)]
            confidence = predictions[i][0] if predictions[i] > 0.5 else 1 - predictions[i][0]
            
            # Color code: green for correct, red for incorrect
            color = 'green' if true_class == pred_class else 'red'
            
            title = f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}'
            axes[i].set_title(title, color=color, fontweight='bold')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample predictions saved to {save_path}")
        
        return fig