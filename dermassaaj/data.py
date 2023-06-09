import pandas as pd
import tensorflow as tf
import numpy as np
from dermasaaj.params import DATA_PATH
from PIL import Image
import pathlib

def return_features(path,df):
    '''
    Fonction qui retourne tous les features du df one hot encoded sous forme
    de numpy array
    '''
    #TODO CATCH ERROR
    # try : .loc
    # Except Raise ERROR
    id_image=path.split('/')[-1]
    id_image=id_image[:-4]
    ligne = df.loc[df['image'] == id_image]
    features = ligne.drop('image', axis=1)
    features_array = np.array(features.values.flatten(), dtype=float)
    return features_array


class Dataloader():
    def __init__(self):
        pass
        # Load csv
        self.metadata =  pd.read_csv(f"{DATA_PATH}/aug_ISIC2019_metadata.csv")
        self.classes =  {"MEL":0
                        ,"NV":1
                        ,"BCC":2
                        ,"AK":3
                        ,"BKL":4
                        ,"DF":5
                        ,"VASC":6
                        ,"SCC":7
                        ,"UNK":8}



    def create_generator(self
                         ,resize = (128,128) # New shape ()
                         ,fold = "train" # Can be "train","valid","test"
                         ):
        def gen():
            w,h=resize
            path = pathlib.Path(f"{DATA_PATH}/{fold}")
            files = []
            n_classes =  len(list(path.glob("*")))
            for sub_dir in  path.glob("*") :
                files += list(sub_dir.glob("*.jpg"))

            print(f"Found {len(files)} files belonging to {n_classes} classe{'s' if n_classes>0 else ''}")
            np.random.shuffle(files) # Inplace methode
            for path in files :
                #print(path)
                *other , class_ ,filename_or  = str(path).split("/")
                #filename = filename_or.split(".")[0]

                #TODO Catch ERROR
                assert class_ in self.classes.keys()
                ohe = 8*[0]
                ohe[self.classes["AK"]]=1
                #filename = filename.split("_")[0] if "sample" in filename else filename
                #print(filename)

                # if not filename in self.metadata.index :
                #     ERRORS.append((filename_or,filename))
                #     continue

                #search = re.search("\d+",filename)
                #if search :

                #id_ = (re.search("\d+",filename).group())

                # RETURN X_img, X2 (metadata), Y
                img =  np.asarray(Image.open(path))
                X_meta = return_features(str(path),self.metadata)
                # yield ( [X_img,X_meta], y_cat)
                yield (np.resize(img,(w,h,3)) ,X_meta),np.array(ohe)

                #else :
                #   print(f"ERROR !!!\ncouldn't parse Id at file {filename_or}")

        return gen


    def create_dataset(self
                       ,fold = "train"
                       ,resize= (128,128)
                       ,batch_size=32):
        w,h=resize
        return  tf.data.Dataset.from_generator(self.create_generator(fold=fold,resize=resize)
                                    ,output_signature=(((tf.TensorSpec(shape=(w,h,3), dtype=tf.int8)
                                                        ,tf.TensorSpec(shape=(12),dtype=tf.int8))
                                                        ,tf.TensorSpec(shape=(8),dtype=tf.int8))
                                                       )
                                    ).batch(batch_size)





#################################################################################
#UNITEST CLASS Dataloader
if __name__=="__main__":
    from keras.layers import Input,Flatten,Concatenate,Dense,AveragePooling2D,Conv2D,MaxPool2D
    from keras.models import Model
    loader= Dataloader()
    ds_train = loader.create_dataset("train")
    ds_val = loader.create_dataset("valid")
    def initialize_model():
        input_cnn = Input(shape=(128,128,3))
        input_meta = Input(shape=(12))

        output_cnn = Conv2D(10,3)(input_cnn)
        output_cnn = MaxPool2D()(output_cnn)
        output_cnn = Conv2D(10,3)(output_cnn)
        output_cnn = MaxPool2D()(output_cnn)
        flat = Flatten()(output_cnn)

        output_dense = Dense(3)(input_meta)

        combined = Concatenate()([flat, output_dense])
        y = Dense(8,activation="softmax")(combined)
        return Model(inputs=[input_cnn,input_meta],outputs=y)
    model = initialize_model()
    model.summary()
    model.compile(optimizer="adam",loss="categorical_crossentropy")
    model.fit(ds_train,epochs=2
              ,validation_data=ds_val)
