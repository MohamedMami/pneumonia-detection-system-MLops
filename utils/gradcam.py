import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import os

class GradCAM:
    def __init__(self, model, img_height=224, img_width=224):
        self.model = model
        self.img_height = img_height
        self.img_width = img_width
    
    def make_gradcam_heatmap(self, img_array, last_conv_layer_name, pred_index=None):
        # Create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs], 
            [self.model.get_layer(last_conv_layer_name).output, self.model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def preprocess_image(self, img_path):
        """Preprocess image for model input"""
        img = image.load_img(img_path, target_size=(self.img_height, self.img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    
    def generate_gradcam(self, img_path, layer_name='conv4', class_idx=None):
        """
        Generate Grad-CAM for a single image
        
        Args:
            img_path: Path to the image file
            layer_name: Name of the convolutional layer to use
            class_idx: Class index for which to generate heatmap
        
        Returns:
            dict: Contains prediction, confidence, and heatmap
        """
        
        # Preprocess image
        img_array = self.preprocess_image(img_path)
        
        # Make prediction
        prediction = self.model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])
        
        # Generate heatmap
        heatmap = self.make_gradcam_heatmap(img_array, layer_name, class_idx)
        
        return {
            'prediction': 'Pneumonia' if confidence > 0.5 else 'Normal',
            'confidence': confidence * 100 if confidence > 0.5 else (1 - confidence) * 100,
            'raw_prediction': confidence,
            'heatmap': heatmap
        }
    
    def visualize_gradcam(self, img_path, heatmap, save_path=None, alpha=0.4):
        """
        Visualize Grad-CAM results
        
        Args:
            img_path: Path to original image
            heatmap: Grad-CAM heatmap
            save_path: Path to save visualization
            alpha: Transparency for overlay
        
        Returns:
            matplotlib figure
        """
        
        # Load original image
        img = image.load_img(img_path)
        img_array = image.img_to_array(img)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original X-ray', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        im1 = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
        overlay = alpha * heatmap_colored + (1 - alpha) * (img_array / 255.0)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay Visualization', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        return fig
    
    def generate_layer_comparison(self, img_path, layer_names, save_path=None):
        """
        Compare Grad-CAM across different layers
        
        Args:
            img_path: Path to image
            layer_names: List of layer names to compare
            save_path: Path to save comparison
        """
        
        img_array = self.preprocess_image(img_path)
        
        fig, axes = plt.subplots(2, len(layer_names), figsize=(5*len(layer_names), 10))
        if len(layer_names) == 1:
            axes = axes.reshape(-1, 1)
        
        # Load original image for overlay
        img = image.load_img(img_path)
        img_array_vis = image.img_to_array(img)
        
        for i, layer_name in enumerate(layer_names):
            try:
                heatmap = self.make_gradcam_heatmap(img_array, layer_name)
                
                # Heatmap
                im = axes[0, i].imshow(heatmap, cmap='jet')
                axes[0, i].set_title(f'{layer_name} - Heatmap', fontweight='bold')
                axes[0, i].axis('off')
                plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)
                
                # Overlay
                heatmap_resized = cv2.resize(heatmap, (img_array_vis.shape[1], img_array_vis.shape[0]))
                heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
                overlay = 0.4 * heatmap_colored + 0.6 * (img_array_vis / 255.0)
                
                axes[1, i].imshow(overlay)
                axes[1, i].set_title(f'{layer_name} - Overlay', fontweight='bold')
                axes[1, i].axis('off')
                
            except Exception as e:
                axes[0, i].text(0.5, 0.5, f'Error: {str(e)}', 
                               ha='center', va='center', transform=axes[0, i].transAxes)
                axes[0, i].axis('off')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Layer comparison saved to {save_path}")
        
        return fig
    
    def batch_gradcam_analysis(self, image_paths, layer_name='conv4', save_dir=None):
        """
        Perform Grad-CAM analysis on multiple images
        
        Args:
            image_paths: List of image paths
            layer_name: Convolutional layer name
            save_dir: Directory to save results
        
        Returns:
            List of analysis results
        """
        
        results = []
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for i, img_path in enumerate(image_paths):
            try:
                print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
                
                # Generate Grad-CAM
                result = self.generate_gradcam(img_path, layer_name)
                result['image_path'] = img_path
                result['image_name'] = os.path.basename(img_path)
                
                # Visualize and save
                if save_dir:
                    save_path = os.path.join(save_dir, f"gradcam_{i+1}_{result['image_name']}")
                    self.visualize_gradcam(img_path, result['heatmap'], save_path)
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                results.append({
                    'image_path': img_path,
                    'image_name': os.path.basename(img_path),
                    'error': str(e)
                })
        
        return results