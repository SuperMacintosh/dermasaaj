import pandas as pd
import numpy as np





def return_features(path,df):
    #fonction qui retourne tous les features du df one hot encoded sous forme
    #de numpy array
    id_image=path.split('/')[-1]

    id_image=id_image[:-4]

    ligne = df.loc[df['image'] == id_image]
    #age = ligne['age_approx'].values[1]


    features = ligne.drop('image', axis=1)

    features_array = np.array(features.values.flatten(), dtype=float)
    return features_array
