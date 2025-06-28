import tensorflow as tf
from tensorflow import keras
import os
import json
from datetime import datetime

class ModelTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.history = None
        self.callbacks = None
    
    def setup_callbacks(self, monitor='val_loss', patience=7):
        """Setup training callbacks"""
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.2,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(self.config.MODEL_SAVE_PATH, self.config.WEIGHTS_NAME),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            keras.callbacks.CSVLogger(
                os.path.join(self.config.RESULTS_DIR, 'training_log.csv'),
                append=True
            )
        ]
        
        self.callbacks = callbacks
        return callbacks
    
    def train(self, train_generator, val_generator, epochs=None, 
              class_weights=None, initial_epoch=0):
        """Train the model"""
        
        if epochs is None:
            epochs = self.config.EPOCHS
        
        if self.callbacks is None:
            self.setup_callbacks()
        
        # Calculate steps
        steps_per_epoch = train_generator.samples // train_generator.batch_size
        validation_steps = val_generator.samples // val_generator.batch_size
        
        print(f"Training Configuration:")
        print(f"- Epochs: {epochs}")
        print(f"- Steps per epoch: {steps_per_epoch}")
        print(f"- Validation steps: {validation_steps}")
        print(f"- Class weights: {class_weights}")
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            initial_epoch=initial_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=self.callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return self.history
    
    def save_training_info(self):
        """Save training information"""
        if self.history is None:
            print("No training history to save.")
            return
        
        # Save training history
        history_path = os.path.join(self.config.RESULTS_DIR, 'training_history.json')
        
        # Convert history to serializable format
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        # Add metadata
        training_info = {
            'model_config': {
                'img_height': self.config.IMG_HEIGHT,
                'img_width': self.config.IMG_WIDTH,
                'batch_size': self.config.BATCH_SIZE,
                'learning_rate': self.config.LEARNING_RATE,
                'epochs_trained': len(history_dict['loss'])
            },
            'training_date': datetime.now().isoformat(),
            'history': history_dict
        }
        
        with open(history_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"Training information saved to {history_path}")
    
    def resume_training(self, checkpoint_path, train_generator, val_generator, 
                       additional_epochs=10):
        """Resume training from a checkpoint"""
        
        # Load weights
        self.model.load_weights(checkpoint_path)
        print(f"Loaded weights from {checkpoint_path}")
        
        # Load previous history if exists
        history_path = os.path.join(self.config.RESULTS_DIR, 'training_history.json')
        initial_epoch = 0
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                training_info = json.load(f)
                initial_epoch = training_info['model_config']['epochs_trained']
                print(f"Resuming from epoch {initial_epoch}")
        
        # Continue training
        return self.train(
            train_generator, 
            val_generator, 
            epochs=initial_epoch + additional_epochs,
            initial_epoch=initial_epoch
        )
    
    def get_training_summary(self):
        """Get a summary of training results"""
        if self.history is None:
            return "No training history available."
        
        final_metrics = {}
        for metric in ['loss', 'accuracy', 'precision', 'recall', 'val_loss', 
                      'val_accuracy', 'val_precision', 'val_recall']:
            if metric in self.history.history:
                final_metrics[metric] = self.history.history[metric][-1]
        
        return final_metrics