import tensorflow as tf
import os
from tensorflow import keras

def initialize_dataset_from_file(url_file_name,extract:bool=False,archive_format:str=None):
    """
    Load tensorflow dataset and return local data path
    """
    dataset_url = url_file_name
    data_dir = tf.keras.utils.get_file(
                                   origin=url_file_name,
                                   extract=extract,
                                   archive_format  = archive_format)
    return os.path.split(data_dir)[0]

def get_split_image_data(parent_path, child_path, img_height, img_width,batch_size:int=32 ):
    """
    get from targeted tensor path and return all existing files
    """
    path=os.path.join(parent_path + f'/{child_path}')
    bloc = tf.keras.utils.image_dataset_from_directory(
    path,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    return bloc


def data_augmentation(img_height:int,img_width:int):
    result = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal",
                        input_shape=(img_height,
                                    img_width,
                                    3)),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ]
    )
    return result
