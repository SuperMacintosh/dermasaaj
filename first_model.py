import matplotlib.pyplot as plt
import pathlib
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow import keras

'''First model built following the Tensorflow tutorial available
at https://www.tensorflow.org/tutorials/images/classification?hl=fr'''

# download dataset from GCP to keras, if not yet downloaded

cache_path = pathlib.Path("../root/.keras/datasets/data")
if not cache_path.is_dir():
  dataset_url = "https://storage.googleapis.com/derma-data/raw_data/archive.zip"
  data_dir = tf.keras.utils.get_file(
                                    origin=dataset_url,
                                      extract=True,
                                      archive_format	='zip')
  data_dir = pathlib.Path(data_dir)

# Parameters definition
batch_size = 32
img_height = 128
img_width = 128

# Create train, validation and test datasets
train_ds = tf.keras.utils.image_dataset_from_directory('../root/.keras/datasets/data/train'
                                        ,batch_size=batch_size
                                        ,image_size=(img_height,img_width))

valid_ds = tf.keras.utils.image_dataset_from_directory('../root/.keras/datasets/data/valid'
                                        ,batch_size=batch_size
                                        ,image_size=(img_height,img_width))

test_ds = tf.keras.utils.image_dataset_from_directory('../root/.keras/datasets/data/test'
                                        ,batch_size=batch_size
                                        ,image_size=(img_height,img_width))

#Percentage of train, valida and test datasets vs. total dataset

total_nb = len(np.concatenate([i for x, i in train_ds], axis=0)) + len(np.concatenate([i for x, i in valid_ds], axis=0)) + len(np.concatenate([i for x, i in test_ds], axis=0))

percentage_train = round(len(np.concatenate([i for x, i in train_ds], axis=0)) / total_nb,2)
percentage_test = round(len(np.concatenate([i for x, i in test_ds], axis=0)) / total_nb, 2)
percentage_valid = round(len(np.concatenate([i for x, i in valid_ds], axis=0)) / total_nb, 2)

print(f'the train dataset represents {percentage_train} % of the total dataset')
print(f'the validation dataset represents {percentage_valid} % of the total dataset')
print(f'the test dataset represents {percentage_test}% of the total dataset')

# See the names of the classes
class_names = train_ds.class_names
print(f'The names of the classes are {class_names}')

# Have a look at the 9 first images of the train dataset
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# Data configuration for better performances
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Model creation
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)), #Data standardization - normaliser les valeurs pour qu'elles soient dans la plage [0, 1]
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes) #output layer
])

#Model compilation
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model summary
model.summary()

# Train the model
epochs=15
history = model.fit(
  train_ds,
  validation_data=valid_ds,
  epochs=epochs
)

#evaluate with test dataset
res = model.evaluate(test_ds)
print(f'the model evaluation score [loss, accuracy] with the test dataset is {res}')

#Fit results visualization
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Data augmentation to reduce overfitting and improve model perormances
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# Visualization of augmented images
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")

# New model creation with dropout method and data augmentation to reduce overfitting
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

#New model compilation
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Model summary
model.summary()

# Train model
epochs = 25
history = model.fit(
  train_ds,
  validation_data=valid_ds,
  epochs=epochs
)

#evaluate with test dataset
res2 = model.evaluate(test_ds)
print(f'the model evaluation score [loss, accuracy] with the test dataset is {res2}')

# Visualize results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Predict with new data
mole_URL = 'https://www.centre-chirurgie-dermatologique.fr/app/uploads/2021/12/melanome-centre-chirurgie-dermatologique.jpg.webp'
mole_path = tf.keras.utils.get_file('Suspicious_mole', origin=mole_URL)

img = tf.keras.utils.load_img(
    mole_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
