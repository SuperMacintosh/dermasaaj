import numpy as np
# from tensorflow import keras
from keras import Model, Sequential, layers
from keras.callbacks import EarlyStopping,CSVLogger, ModelCheckpoint
import tensorflow as tf
import os
from dermaflow.logic.preprocessing import data_augmentation
from dermaflow.params import *
from keras.applications.densenet import DenseNet201

def initialize_model(num_classes:int,
                    model_type:str=MODEL_TYPE,
                    kernel_size:int=3,
                    val_dropout:float=0.2,
                    img_height:int=IMAGE_HEIGHT,
                    img_width:int=IMAGE_WIDTH,
                    input_shape:tuple=None,
                    ) -> Model:
    """
    Initialize adequate CNN model function of model_type
    """
    if str.upper(model_type) == 'DERMA':
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
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
        ])
    elif str.upper(model_type) == 'DENSENET201':
        model = DenseNet201(
                            include_top=False,
                            weights="imagenet",
                            input_shape=input_shape,
                            classifier_activation="softmax"
                            )
        model.trainable = False
        model = Sequential([
        model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
        ])


    else:
        print(f'\n❌ Unknown model type')

    print(f'✅ Model initialized')

    return model

def compile_model(model: Model, model_type:str=MODEL_TYPE) -> Model:
    """
    Compile the defined CNN model
    """
    if os.upper(model_type) == 'DERMA':
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        opt='adam'
    elif  os.upper(model_type) == 'DENSENET201':
        loss='categorical_crossentropy'
        opt =tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=opt,
              loss=loss,
              metrics=['accuracy'])

    print(f'✅ Model compiled')
    return model

def train_model(
        model: Model,
        X: np.ndarray,
        validation_data,
        model_type:str=MODEL_TYPE,
        batch_size=BATCH_SIZE,
        epochs:int=20,
        patience:int=2,
        verbose:int=0
    ) :

    """
    Fit the model and return a tuple (fitted_model, history)
    """
    if os.upper(model_type) == 'DERMA':
        monitor="val_loss"
        mode='max'
    elif os.upper(model_type) == 'DENSENET201':
        monitor = 'val_accuracy'
        mode='max'

    es = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        restore_best_weights=True,
        verbose=verbose
    )
    csv_logger = CSVLogger('training.log')

    checkpoint_callback = ModelCheckpoint(
                filepath=LOCAL_CHECKPOINT_PATH,
                save_weights_only=True,
                monitor=monitor,
                mode=mode,
                save_best_only=True
                )

    history = model.fit(X,
                        model_type,
                        validation_data,
                        epochs,
                        callbacks=[es, csv_logger,checkpoint_callback]
    )
    print(f'✅ Model succesfully trained through {len(history.epoch)} epochs')
    return model, history

def evaluate_model(
        model: Model,
        X: np.ndarray,
        batch_size=BATCH_SIZE
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

    print(f'✅ Model evaluated, [Loss, Accuracy]: [{round(metrics["loss"], 2)},{round(metrics["accuracy"], 2)}]')

    return metrics
