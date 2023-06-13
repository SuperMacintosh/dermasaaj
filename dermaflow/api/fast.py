from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dermaflow.logic.registry import load_model
import tensorflow as tf
from dermaflow.params import *
import numpy as np
import cv2
from keras.applications.densenet import preprocess_input


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
@app.post("/predict")


def predict(img_file: UploadFile=File(...),
                        candidate_age: int = Form(...),
                        candidate_gender: int = Form(...),
                        candidate_anatom_site: int = Form(...)
                        ):
    ### Receiving and decoding the image
    contents = img_file.read()
    nparr = np.fromstring(contents, np.uint8)
    img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
    img_array= preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)
    # result=app.state.model.predict(img_array,candidate_gender,candidate_age,candidate_anatom_site)
    result=app.state.model.predict(img_array)

    return{f'✅ This image most likely belongs to {CLASS_NAMES[np.argmax(result)]} with a probability of {round(100*np.max(result),2)}%'}

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
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    ### Encoding and responding with the image
    im = cv2.imencode('.png', cv2_img)[1] # extension depends on which format is sent from Streamlit
    # return Response(content=im.tobytes(), media_type="image/png")
    return{'age': age,'gender': sexe, 'img_shape': cv2_img.shape}

@app.get("/")
def root():
    return {'Dermasaaj - new': ' ✅ Here we are !!! '}
