#%%
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

import time
from pathlib import Path

TRAIN_DIR = 'FaceMaskDataset/FaceMaskDataset/train'
TEST_DIR = 'FaceMaskDataset/FaceMaskDataset/test'
EPOCHS = 100

#%%
def create_train_val_generators(data_dir, target_size=(224, 224), batch_size=32, val_split=0.2):
    """
    Creates train and validation generators with augmentation.
    """
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.15,
        validation_split=val_split  # 20% for validation
    )
    
    # Training generator (80% of data, WITH augmentation)
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    # Validation generator (20% of data, NO augmentation)
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

#%%
#TODO json config object - maybe
def create_model(input_shape=(224, 224, 3)):
    """
    create_model
    """
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

#%%
def set_callbacks():
  # Create directory for models
    models_dir = Path('trained_models')
    models_dir.mkdir(exist_ok=True)
    
    # Create logs directory for TensorBoard
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # 1. ModelCheckpoint - Saves best model
    checkpoint = ModelCheckpoint(
        filepath=str(models_dir / 'best.keras'),
        monitor='val_accuracy', #TODO check val_loss
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    # 2. TensorBoard - Logs training metrics
    tensorboard = TensorBoard(
        log_dir=str(logs_dir / f'{time.strftime("%Y%m%d_%H%M%S")}'),
        histogram_freq=0, #set this to 1 to check - crashes colab
        write_graph=True,
        write_images=False
    )
    
    # 3. EarlyStopping - Prevents overfitting
    early_stop = EarlyStopping(
        monitor='val_accuracy', #TODO check val_loss
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    return [checkpoint, tensorboard, early_stop]
    

#%%
# ===================== TRAIN =================================

train_generator, val_generator = create_train_val_generators(TRAIN_DIR)

model = create_model()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#model.summary()

callbacks = set_callbacks()

history = model.fit(
    train_generator,                # <-- THE GENERATOR
    epochs=EPOCHS,
    validation_data=val_generator,  # Validation generator
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=callbacks       
)


# %%
