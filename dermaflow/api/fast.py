from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dermaflow.logic.registry import load_model
import tensorflow as tf
from dermaflow.params import *
from pydantic import BaseModel
# import tempfile
import numpy as np
# import io, import base64, import imageio
import cv2
from starlette.responses import Response


class Item(BaseModel):
    img: str
    candidate_gender: int | None = None
    candidate_age: int | None = None
    candidate_anatom_site: int | None = None


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

# Predict end point
@app.get("/predict")

def predict(
        image_url: str,             # url/path image
        candidate_gender: int,      # 0,1 : male, female
        candidate_age: int,         # [0, 110] : month & year of birthday
        candidate_anatom_site: int  # [0,8]
    ):
    if 0:
        img = tf.keras.utils.load_img(image_url, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        result=app.state.model.predict(img_array,candidate_gender,candidate_age,candidate_anatom_site)
    # score = tf.nn.softmax(result[0])
    else:
        result=[-9999,0]

    return{'✅ This image most likely belongs to ': int(result[0])}

@app.post("/upload_items")

async def post(
    image: UploadFile = File(...),
    age: int = Form(...),
    gender: int = Form(...),
):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(cv2_img.shape)

     # tf.convert_to_tensor(img)

    if 0:

        img = tf.keras.utils.load_img(fp, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        result=app.state.model.predict(img_array,candidate_gender,candidate_age,candidate_anatom_site)
    # score = tf.nn.softmax(result[0])
    else:

        return{'age': age,'gender': gender, 'img_shape': cv2_img.shape}


@app.post('/test')

async def receive_image(img: UploadFile=File(...),
                        age: int = Form(...),
                        sexe: int = Form(...)
                        ):
    ### Receiving and decoding the image
    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)
    print(nparr.shape)
    print(nparr)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    ### Encoding and responding with the image
    im = cv2.imencode('.png', cv2_img)[1] # extension depends on which format is sent from Streamlit
    # return Response(content=im.tobytes(), media_type="image/png")
    return{'age': age,'gender': sexe, 'img_shape': cv2_img.shape}

@app.get("/")
def root():
    return {'Dermasaaj - new': ' ✅ Here we are !!! '}
