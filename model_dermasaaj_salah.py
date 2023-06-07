import numpy as np
from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
import tensorflow as tf
from preprocessing import data_augmentation


def initialize_model(input_shape: tuple, num_classes:int, kernel_size:int=3, val_dropout:float=0.2) -> Model:
    """
    Initialize adeQUATE CNN model
    """
    model = Sequential([
    data_augmentation,
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
    return model

def compile_model(model: Model) -> Model:
    """
    Compile the defined CNN model
    """
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=2,
        validation_data=None,
        ) -> np.Tuple[Model, dict]:

    """
    Fit the model and return a tuple (fitted_model, history)
    """
    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
    X,
    validation_data=validation_data,
    epochs=10,
    callbacks=[es]
    )

    return model, history

def evaluate_model(
        model: Model,
        X: np.ndarray,
        batch_size=64
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

    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

    return metrics
