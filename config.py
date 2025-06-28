import os 

class Config: 
    # Model parameters
    IMG_HIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    # Data paths
    DATA_DIR = 'data'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    # Model paths
    MODEL_SAVE_PATH= 'models'
    MODEL_NAME = 'pneumonia_detector.h5'
    WEIGHTS_NAME = 'best_pneumonia_model.h5'
    # Results
    RESULTS_DIR = 'results'
    PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
    
    # Create directories
    @classmethod
    def create_directories(cls):
        for directory in [cls.MODEL_SAVE_PATH, cls.RESULTS_DIR, cls.PLOTS_DIR]:
            os.makedirs(directory, exist_ok=True)
            
    