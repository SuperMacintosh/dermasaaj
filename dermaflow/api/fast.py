from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dermaflow.logic.registry import load_model
import tensorflow as tf
from params import *


app = FastAPI()

app.state.model=load_model()

# Allowing all middleware is optional, but good practice for dev purposes

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# end point

@app.get("/predict")

def predict(
        image_url: str,             # url/path image
        candidate_gender: int,      # 0,1 : male, female
        candidate_age: int,         # [0, 110] : month & year of birthday
        candidate_anatom_site: int  # [0,8]
    ):

    img = tf.keras.utils.load_img(image_url, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    result=app.state.model.predict(img_array,candidate_gender,candidate_age,candidate_anatom_site)
    # score = tf.nn.softmax(result[0])

    return{'This image most likely belongs to ': int(result[0])}

@app.get("/")
def root():
    return {'Dermasaaj': 'Here we are'}
