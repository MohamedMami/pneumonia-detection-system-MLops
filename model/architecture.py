import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class PneumoniaModel:
    def __init__(self,img_height=224, img_width=224, num_classes=1):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes  
        self.model = None
        
    def build_basic_cnn(self):
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.img_height, self.img_width, 3,)),
            # first conv layer
            layers.Conv2D(32, (3, 3), activation='relu', name='conv1'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            # second conv layer
            layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            # third conv layer
            layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            # forth conv layer
            layers.Conv2D(256, (3, 3), activation='relu', name='conv4'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            # Flatten the output
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='sigmoid')
        ])
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        if self.model is None:
            raise ValueError("Model is not built yet. Call build_basic_cnn() first.")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return self.model
    def get_model_summary(self):
        if self.model is None:
            raise ValueError("Model is not built yet. Call build_basic_cnn() first.")
        return self.model.summary()
    def save_model(self, model_path):
        if self.model is None:
            raise ValueError("Model is not built yet. Call build_basic_cnn() first.")
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    def save_weights(self, weights_path):
        if self.model is None:
            raise ValueError("Model is not built yet. Call build_basic_cnn() first.")
        self.model.save_weights(weights_path)
        print(f"Weights saved to {weights_path}")
    def load_weights(self, weights_path):
        if self.model is None:
            raise ValueError("Model is not built yet. Call build_basic_cnn() first.")
        self.model.load_weights(weights_path)
        print(f"Weights loaded from {weights_path}")