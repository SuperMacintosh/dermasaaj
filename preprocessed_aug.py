# Import packages
import glob
import cv2
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def augmentation(min_rotation, max_rotation):

    # Rotation range
    rotation_range = random.randint(min_rotation,max_rotation)


    datagen = ImageDataGenerator(
        featurewise_center = False,
        featurewise_std_normalization = False,
        rotation_range = rotation_range,
        width_shift_range = 0.02,
        height_shift_range = 0.02,
        vertical_flip = True,
        horizontal_flip = True,
        zoom_range = (0.8, 1.2),
        fill_mode='nearest'
        )
    return datagen


def generated_data(datagen,image_content):

    #expand dims
    datagen.fit(image_content)

    #Augmentation of the image
    X_augmented = datagen.flow(image_content, shuffle=False, batch_size=1)

    #Reduce dims
    X_aug = np.squeeze(X_augmented[0], axis=0)
    X_aug = X_aug.astype('uint8')


    return X_aug

def data_augmentation_classes(folder_class, augmented_folder,factor_mult:int):
    for file in glob.glob(f'{folder_class}/*.jpg'):
        #load the image
        img = cv2.imread(file)
        img = np.expand_dims(img, axis=0)
        write_path='/'.join(file.split('/')[2:]).rstrip('.jpg')

        for i in range(factor_mult):
            X_aug=generated_data(img, i+1)
            #save the data
            cv2.imwrite(f'{augmented_folder}/{write_path}_{i}.jpg', X_aug)
