import os
import argparse
from config import Config
from model.architecture import PneumoniaModel
from model.evaluator import ModelEvaluator
from utils.data_loader import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Evaluate Pneumonia Detection Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--weights_path', type=str, default=None,
                       help='Path to model weights (optional)')
    parser.add_argument('--detailed', action='store_true',
                       help='Generate detailed evaluation report')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    config.create_directories()
    
    print(f"Evaluating model: {args.model_path}")
    
    # Initialize data loader
    data_loader = DataLoader(
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        batch_size=config.BATCH_SIZE
    )
    
    # Create test data generator
    _, _, test_gen = data_loader.create_data_generators(
        config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR,
        augmentation=False  # No augmentation for evaluation
    )
    
    # Load model
    if args.model_path.endswith('.h5'):
        import tensorflow as tf
        model = tf.keras.models.load_model(args.model_path)
        print(f"Loaded complete model from {args.model_path}")
    else:
        # Load architecture and weights separately
        pneumonia_model = PneumoniaModel(
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH
        )
        
        # You would need to specify which architecture was used
        model = pneumonia_model.build_basic_cnn()
        model = pneumonia_model.compile_model()
        
        if args.weights_path:
            model.load_weights(args.weights_path)
            print(f"Loaded weights from {args.weights_path}")
        else:
            print("Warning: No weights loaded. Using random initialization.")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, config)
    
    # Evaluate model
    print("Starting evaluation...")
    results = evaluator.evaluate_model(test_gen)
    
    # Print summary
    evaluator.print_evaluation_summary()
    
    # Generate detailed report if requested
    if args.detailed:
        print("\nGenerating detailed report...")
        report_dir = evaluator.generate_detailed_report(test_gen)
        print(f"Detailed report saved to: {report_dir}")
    
    print("\nEvaluation completed successfully!")

if __name__ == '__main__':
    main()