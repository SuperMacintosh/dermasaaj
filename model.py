import numpy as np
# from tensorflow import keras
from keras import Model, Sequential, layers
from keras.callbacks import EarlyStopping
import tensorflow as tf
import os
from preprocessing import data_augmentation


def initialize_model(num_classes:int,
                     kernel_size:int=3,
                     val_dropout:float=0.2,
                     img_height:int=os.getenv('IMAGE_HEIGHT'),
                     img_width:int=os.getenv('IMAGE_WITH')
                     ) -> Model:
    """
    Initialize adequate CNN model
    """
    model = Sequential([
    data_augmentation(img_height,img_width, val_rotation=0.1, val_zoom=0.1),
    layers.Rescaling(1./255),
    layers.Conv2D(16, kernel_size, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, kernel_size, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, kernel_size, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(val_dropout),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
    ])
    print(f'✅ Model initialized')
    return model

def compile_model(model: Model) -> Model:
    """
    Compile the defined CNN model
    """
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    print(f'✅ Model compiled')
    return model

def train_model(
        model: Model,
        X: np.ndarray,
        validation_data,
        batch_size=os.getenv('BATCH_SIZE'),
        epochs:int=20,
        patience=2,
        verbose:int=0
        ) :

    """
    Fit the model and return a tuple (fitted_model, history)
    """
    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=verbose
    )

    history = model.fit(
    X,
    validation_data=validation_data,
    epochs=epochs,
    callbacks=[es]
    )
    print(f'✅ Model succesfully trained through {len(history.epoch)} epochs')
    return model, history

def evaluate_model(
        model: Model,
        X: np.ndarray,
        batch_size=os.getenv('BATCH_SIZE')
    ) :
    """
    Evaluate trained model performance on the dataset
    """

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        batch_size=batch_size,
        verbose=0,
        return_dict=True
    )

    print(f'✅ Model evaluated, [Loss, Accuracy]: [{round(metrics[0], 2)},{round(metrics[1], 2)}')

    return metrics
