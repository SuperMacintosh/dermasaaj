from tensorflow import keras


def data_augmentation(img_height:int,img_width:int):
    result = keras.Sequential(
    [
        keras.layerslayers.RandomFlip("horizontal",
                        input_shape=(img_height,
                                    img_width,
                                    3)),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ]
    )
    return result
