import os
import argparse
from config import Config, DevelopmentConfig, ProductionConfig
from model.architecture import PneumoniaModel
from model.trainer import ModelTrainer
from utils.data_loader import DataLoader
from utils.visualization import VisualizationUtils

def main():
    parser = argparse.ArgumentParser(description='Train Pneumonia Detection Model')
    parser.add_argument('--config', type=str, default='development',
                       choices=['development', 'production'],
                       help='Configuration to use')
    parser.add_argument('--model_type', type=str, default='basic',
                       choices=['basic', 'deeper'],
                       help='Model architecture to use')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == 'development':
        config = DevelopmentConfig()
    else:
        config = ProductionConfig()
    
    # Create directories
    config.create_directories()
    
    print(f"Using {args.config} configuration")
    print(f"Training {args.model_type} model")
    
    # Initialize data loader
    data_loader = DataLoader(
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        batch_size=config.BATCH_SIZE
    )
    
    # Verify data structure
    print("Verifying data structure...")
    data_loader.verify_data_structure(config.DATA_DIR)
    
    # Create data generators
    print("Creating data generators...")
    train_gen, val_gen, test_gen = data_loader.create_data_generators(
        config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR
    )
    
    # Calculate class weights
    class_weights = data_loader.get_class_weights(train_gen)
    print(f"Class weights: {class_weights}")
    
    # Initialize model
    pneumonia_model = PneumoniaModel(
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH
    )
    
    # Build model
    if args.model_type == 'basic':
        model = pneumonia_model.build_basic_cnn()
    else:
        model = pneumonia_model.build_deeper_cnn()
    
    # Compile model
    model = pneumonia_model.compile_model(learning_rate=config.LEARNING_RATE)
    
    # Print model summary
    print("\nModel Architecture:")
    pneumonia_model.get_model_summary()
    
    # Initialize trainer
    trainer = ModelTrainer(model, config)
    
    # Train model
    if args.resume:
        print(f"Resuming training from {args.resume}")
        history = trainer.resume_training(args.resume, train_gen, val_gen)
    else:
        print("Starting training from scratch...")
        history = trainer.train(train_gen, val_gen, class_weights=class_weights)
    
    # Save training information
    trainer.save_training_info()
    
    # Save final model
    model_path = os.path.join(config.MODEL_SAVE_PATH, config.MODEL_NAME)
    pneumonia_model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Generate training visualizations
    viz = VisualizationUtils(config)
    history_plot_path = os.path.join(config.PLOTS_DIR, 'training_history.png')
    viz.plot_training_history(history_dict=history.history, save_path=history_plot_path)
    
    # Print training summary
    print("\nTraining Summary:")
    final_metrics = trainer.get_training_summary()
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTraining completed successfully!")

if __name__ == '__main__':
    main()