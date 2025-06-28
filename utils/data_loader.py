import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


class DataLoader:
    def __init__(self, img_height=224, img_width=224, batch_size=32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

    def load_data(self, train_dir,val_dir,test_dir,augmentation=True):
        if augmentation:
            datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        val_test_generation = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True,
        )
        val_generator = val_test_generation.flow_from_directory(
            val_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False,
        )
        test_generator = val_test_generation.flow_from_directory(
            test_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False,
        )
        return train_generator, val_generator, test_generator
    
    def get_class_weights(self, train_generator):
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        # Get class indices
        class_indices = train_generator.class_indices
        classes = list(class_indices.keys())
        
        # Get all labels
        labels = train_generator.classes
        
        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        return dict(zip(np.unique(labels), class_weights))
    def verify_data_structure(self, data_dir):
        """Verify the data directory structure"""
        required_dirs = ['train', 'val', 'test']
        required_classes = ['NORMAL', 'PNEUMONIA']
        
        for split_dir in required_dirs:
            split_path = os.path.join(data_dir, split_dir)
            if not os.path.exists(split_path):
                raise FileNotFoundError(f"Directory {split_path} not found")
            
            for class_name in required_classes:
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    raise FileNotFoundError(f"Directory {class_path} not found")
                
                # Count images
                image_count = len([f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"{split_dir}/{class_name}: {image_count} images")
        
        print("Data structure verification completed successfully!")
        