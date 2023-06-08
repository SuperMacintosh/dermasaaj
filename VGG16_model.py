import matplotlib.pyplot as plt
import pathlib
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

# download dataset from GCP to keras if not yet downloaded

cache_path = pathlib.Path("../root/.keras/datasets/data")
if not cache_path.is_dir():
  dataset_url = "https://storage.googleapis.com/derma-data/raw_data/archive.zip"
  data_dir = tf.keras.utils.get_file(
                                    origin=dataset_url,
                                      extract=True,
                                      archive_format	='zip')
  data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 128
img_width = 128

train_ds = tf.keras.utils.image_dataset_from_directory('../root/.keras/datasets/data/train'
                                        ,batch_size=batch_size
                                        ,label_mode='categorical'
                                        ,image_size=(img_height,img_width))

test_ds = tf.keras.utils.image_dataset_from_directory('../root/.keras/datasets/data/test'
                                        ,batch_size=batch_size
                                        ,label_mode='categorical'
                                        ,image_size=(img_height,img_width))

val_ds = tf.keras.utils.image_dataset_from_directory('../root/.keras/datasets/data/valid'
                                        ,batch_size=batch_size
                                        ,label_mode='categorical'
                                        ,image_size=(img_height,img_width))

# Load VGG16 model
from tensorflow.keras.applications.vgg16 import VGG16

def load_model():

    model = VGG16(weights="imagenet", include_top=False, input_shape=x[0].shape)

    return model

# Number of parameters
model = load_model()
model.summary()

# deactivate the training of the VGG16 parameters
def set_nontrainable_layers(model):

    # Set the first layers to be untrainable
    model.trainable = False

    return model

# check params of VGG16 are non-trainable
model = set_nontrainable_layers(model)
model.summary()

#Chain pre-trained layers of VGG16 with our flattening and dense layers
from tensorflow.keras import layers, models

class_names = train_ds.class_names
num_classes = len(class_names)

def add_last_layers(model):
    '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''

    base_model = set_nontrainable_layers(model)
    flatten_layer = layers.Flatten()
    dense_layer = layers.Dense(500, activation='relu')
    prediction_layer = layers.Dense(num_classes, activation='softmax')

    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer,
        prediction_layer
    ])

    return model

# Number of parameters for our customized VGG16
model = add_last_layers(model)
model.summary()

# Build a full customized VGG16 and compile it
from tensorflow.keras import optimizers

def build_model():

    model = load_model()
    model = add_last_layers(model)

    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

# VGG16 required preproc_input of X
from tensorflow import keras
from keras.applications.vgg16 import preprocess_input

def new_preproc(X,y):
  return preprocess_input(X),y

train_preproc_ds = train_ds.map(new_preproc)
val_preproc_ds = val_ds.map(new_preproc)
test_preproc_ds = test_ds.map(new_preproc)

# Fit the model
from keras.callbacks import EarlyStopping

model = build_model()

es = EarlyStopping(monitor = 'val_accuracy',
                   mode = 'max',
                   patience = 5,
                   verbose = 1,
                   restore_best_weights = True)

history = model.fit(train_preproc_ds,
                    validation_data=val_preproc_ds,
                    epochs=50,
                    batch_size=batch_size,
                    callbacks=[es]
                    )

# Plot the accuracy
def plot_history(history, title='', axs=None, exp_name=""):
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history.history['loss'], label='train' + exp_name)
    ax1.plot(history.history['val_loss'], label='val' + exp_name)
    #ax1.set_ylim(0., 2.2)
    ax1.set_title('loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train accuracy'  + exp_name)
    ax2.plot(history.history['val_accuracy'], label='val accuracy'  + exp_name)
    #ax2.set_ylim(0.25, 1.)
    ax2.set_title('Accuracy')
    ax2.legend()
    return (ax1, ax2)

plot_history(history)

# Evaluate the model
res_vgg = model.evaluate(test_preproc_ds)
test_accuracy_vgg = res_vgg[-1]

print(f"test_accuracy_vgg = {round(test_accuracy_vgg,2)*100} %")
