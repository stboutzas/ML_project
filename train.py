#%%

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

import random
import time
from pathlib import Path
    
SEED = 42 
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

TRAIN_DIR = 'files/FaceMaskDataset/train'
TEST_DIR = 'files/FaceMaskDataset/test'
# MODELS_DIR = 'files/trained_models'
# LOGS_DIR = 'files/logs'
# if 'google.colab' in sys.modules:
#     TRAIN_DIR = 'FaceMaskDataset/train'
#     TEST_DIR = 'FaceMaskDataset/test'
PATH = Path('files/check2')
EPOCHS = 50
NAME = f'bigger_{EPOCHS}'
MODELS_DIR = PATH / 'trained_models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = PATH / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def create_train_val_generators(data_dir, target_size=(224, 224), batch_size=32, val_split=0.2):
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.15,
        validation_split=val_split,  # 20% for validation
        #preprocessing_function=lambda x: np.mean(x, axis=-1, keepdims=True),  # grayscale
        
    )

    # Training generator (80% of data, WITH augmentation)
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=SEED
    )

    # Validation generator 
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=SEED
    )

    return train_generator, val_generator

#TODO json config object - maybe
# def create_model(input_shape=(224, 224, 3), filters=32, size=3, pooling=2, stride=1, dense=128, dropout=0.2, name=None):

#     model = Sequential([
#         Input(shape=input_shape),
#         Conv2D(filters, (size,size)),
#         BatchNormalization(),
#         Activation('relu'),
#         MaxPooling2D(pooling,stride),
#         Conv2D(filters*2, (size,size)),
#         BatchNormalization(),
#         Activation('relu'),
#         MaxPooling2D(pooling,stride),
#         Flatten(),
#         Dense(dense, activation='relu'),  # BatchNormalization layers ???
#         Dropout(dropout),
#         Dense(1, activation='sigmoid')
#     ])

#     if name is None:
#         name = f'model_{filters}_{size}_{pooling}_{stride}_{dense}_{str(dropout).replace('.', '')}'
#     return model, name

def set_callbacks(name, metric='val_accuracy'):
  # directory for models
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(exist_ok=True)

    # directory for TensorBoard
    logs_dir = Path(LOGS_DIR)
    logs_dir.mkdir(exist_ok=True)

    if metric == 'val_accuracy':
        mode = 'max'  # Higher accuracy is better
    elif metric == 'val_loss':
        mode = 'min'  # Lower loss is better

    # ====== Checkpoint
    checkpoint = ModelCheckpoint(
        filepath=str(models_dir / f'{name}.keras'),
        monitor=metric, #TODO check val_loss
        mode=mode,
        save_best_only=True,
        verbose=1
    )

    # ===== Logs for TensorBoard
    tensorboard = TensorBoard(
        log_dir=str(logs_dir / f'{name}_{time.strftime("%Y%m%d_%H%M%S")}'),
        histogram_freq=0, #set this to 1 to check - crashes colab
        write_graph=False,
        write_images=False
    )

    # ===== EarlyStopping 
    early_stop = EarlyStopping(
        monitor=metric, #TODO check val_loss
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    return [tensorboard, early_stop, checkpoint]

# ===================== TRAIN =================================

models_dir = Path(MODELS_DIR)
models_dir.mkdir(exist_ok=True)
logs_dir = Path(LOGS_DIR)
logs_dir.mkdir(exist_ok=True)
metric = 'val_loss'
if metric == 'val_accuracy':
    mode = 'max'  # Higher accuracy is better
elif metric == 'val_loss':
    mode = 'min'  # Lower loss is better

train_generator, val_generator = create_train_val_generators(TRAIN_DIR, batch_size=16, val_split=0.2)


# configs = [

#     {'filters': 8, 'size': 5, 'pooling': 2, 'stride': 2, 'dense': 128, 'dropout': 0.2,},
#     {'filters': 8, 'size': 5, 'pooling': 2, 'stride': 2, 'dense': 128, 'dropout': 0.5,},
#     {'filters': 16, 'size': 5, 'pooling': 2, 'stride': 2, 'dense': 128, 'dropout': 0.5,},
#     {'filters': 32, 'size': 5, 'pooling': 2, 'stride': 2, 'dense': 512, 'dropout': 0.5,},
# ]

#for config in configs:

#config = configs[0]
# model, name = create_model(
#     input_shape=(224, 224, 3),
#     filters=config['filters'],
#     size=config['size'],
#     pooling=config['pooling'],
#     stride=2,
#     dense=config['dense'],
#     dropout=config['dropout']
# )

model = Sequential([
        Input(shape=(224, 224, 3)),
        Conv2D(32, (5,5), strides=(2,2), padding='same'),
        #BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(3,2),
        Conv2D(64, (5,5), strides=(2,2), padding='same'),
        #BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(3,2),
        Flatten(),
        Dropout(0.5),
        Dense(1024),  
        #BatchNormalization(),
        Activation('relu'),
        Dense(1, activation='sigmoid')
    ])

#model, name = create_model()

model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

#model.summary()

callbacks = set_callbacks(NAME, metric='val_accuracy') #val_loss

history = model.fit(
    train_generator,                # <-- THE GENERATOR
    epochs=EPOCHS,
    validation_data=val_generator,  # Validation generator
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    #callbacks=[checkpoint, early_stop, tensorboard]
    callbacks=callbacks
)


# %%
