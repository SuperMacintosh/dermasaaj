from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dermaflow.logic.registry import load_model
import tensorflow as tf
from dermaflow.params import *
import numpy as np
import cv2
from keras.applications.densenet import preprocess_input
from dermaflow.logic.model import compile_model

app = FastAPI()

# model=load_model()

model = tf.keras.models.load_model('DenseNet121_best_model.keras', compile=False)
model = compile_model(model, MODEL_TYPE)
app.state.model=model

# Allowing all middleware is optional, but good practice for dev purposes

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/test2")

async def predict_test(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()
    return {'message': '✅ Api reached '}

# Predict end point
@app.post("/predict_cnn")

async def predict_cnn(img: UploadFile=File(...)):
    ### Receiving and decoding the image

    if app.state.model == None:
        return {'message': f"\n❌ No model found where asked"}

    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
    img_array=cv2.resize(img_array, (IMAGE_HEIGHT, IMAGE_WIDTH))
    img_array= preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)
    result=app.state.model.predict(img_array)
    msg=f'✅ This image most likely belongs to {CLASS_NAMES[np.argmax(result)]} with a probability of {round(100*np.max(result),2)}%'

    return {'message': msg}


@app.post("/predict_full")

async def predict_full(img_file: UploadFile=File(...),
                        candidate_age: int = Form(...),
                        candidate_gender: int = Form(...),
                        candidate_anatom_site: int = Form(...)
                        ):
    ### Receiving and decoding the image
    contents = await img_file.read()
    nparr = np.fromstring(contents, np.uint8)
    img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
    img_array= preprocess_input(img_array)
    img_array = tf.expand_dims(img_array, 0)
    result=app.state.model.predict(img_array,candidate_gender,candidate_age,candidate_anatom_site)
    msg=f'✅ This image most likely belongs to {CLASS_NAMES[np.argmax(result)]} with a probability of {round(100*np.max(result),2)}%'
    return {'message': msg}


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
