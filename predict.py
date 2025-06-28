import os
import argparse
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from utils.gradcam import GradCAM
import matplotlib.pyplot as plt

def load_and_preprocess_image(img_path, img_height=224, img_width=224):
    """Load and preprocess a single image"""
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array, img

def predict_single_image(model, img_path, threshold=0.5):
    """Make prediction on a single image"""
    img_array, original_img = load_and_preprocess_image(img_path)
    
    # Make prediction
    prediction = model.predict(img_array, verbose=0)
    confidence = float(prediction[0][0])
    
    # Interpret prediction
    if confidence > threshold:
        predicted_class = 'Pneumonia'
        final_confidence = confidence * 100
    else:
        predicted_class = 'Normal'
        final_confidence = (1 - confidence) * 100
    
    return {
        'prediction': predicted_class,
        'confidence': final_confidence,
        'raw_score': confidence,
        'image': original_img
    }

def main():
    parser = argparse.ArgumentParser(description='Predict Pneumonia from X-ray Images')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to a single image for prediction')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Directory containing images for batch prediction')
    parser.add_argument('--gradcam', action='store_true',
                       help='Generate Grad-CAM visualizations')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Prediction threshold (default: 0.5)')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image_path and not args.image_dir:
        print("Error: Either --image_path or --image_dir must be provided")
        return
    
    if args.image_path and args.image_dir:
        print("Error: Provide either --image_path or --image_dir, not both")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    
    # Initialize Grad-CAM if requested
    if args.gradcam:
        gradcam = GradCAM(model)
        gradcam_dir = os.path.join(args.output_dir, 'gradcam')
        os.makedirs(gradcam_dir, exist_ok=True)
    
    # Single image prediction
    if args.image_path:
        print(f"Predicting for image: {args.image_path}")
        
        result = predict_single_image(model, args.image_path, args.threshold)
        
        print(f"\nPrediction Results:")
        print(f"Image: {os.path.basename(args.image_path)}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Raw Score: {result['raw_score']:.4f}")
        
        # Generate Grad-CAM if requested
        if args.gradcam:
            print("Generating Grad-CAM visualization...")
            gradcam_result = gradcam.generate_gradcam(args.image_path)
            
            # Save visualization
            save_path = os.path.join(gradcam_dir, f"gradcam_{os.path.basename(args.image_path)}")
            gradcam.visualize_gradcam(args.image_path, gradcam_result['heatmap'], save_path)
    
    # Batch prediction
    elif args.image_dir:
        print(f"Predicting for images in directory: {args.image_dir}")
        
        # Get all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(args.image_dir) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print("No image files found in the specified directory")
            return
        
        print(f"Found {len(image_files)} images")
        
        results = []
        
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(args.image_dir, img_file)
            
            try:
                result = predict_single_image(model, img_path, args.threshold)
                result['filename'] = img_file
                results.append(result)
                
                print(f"[{i+1}/{len(image_files)}] {img_file}: {result['prediction']} "
                      f"({result['confidence']:.2f}%)")
                
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue
        
        # Save results to CSV
        import pandas as pd
        
        df_results = pd.DataFrame([
            {
                'filename': r['filename'],
                'prediction': r['prediction'],
                'confidence': r['confidence'],
                'raw_score': r['raw_score']
            }
            for r in results
        ])
        
        csv_path = os.path.join(args.output_dir, 'batch_predictions.csv')
        df_results.to_csv(csv_path, index=False)
        print(f"\nBatch prediction results saved to: {csv_path}")
        
        # Generate Grad-CAM for batch if requested
        if args.gradcam:
            print("Generating Grad-CAM visualizations for batch...")
            image_paths = [os.path.join(args.image_dir, r['filename']) for r in results]
            gradcam.batch_gradcam_analysis(image_paths, save_dir=gradcam_dir)
        
        # Print summary
        pneumonia_count = sum(1 for r in results if r['prediction'] == 'Pneumonia')
        normal_count = len(results) - pneumonia_count
        
        print(f"\nBatch Prediction Summary:")
        print(f"Total images processed: {len(results)}")
        print(f"Pneumonia detected: {pneumonia_count}")
        print(f"Normal cases: {normal_count}")
        print(f"Average confidence: {np.mean([r['confidence'] for r in results]):.2f}%")
    
    print("\nPrediction completed successfully!")

if __name__ == '__main__':
    main()