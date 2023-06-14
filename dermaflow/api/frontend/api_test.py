import streamlit as st
from PIL import Image
import requests
from dotenv import load_dotenv
import json
import cv2
import numpy as np

url = "http://localhost:8000"

img_file_buffer = st.file_uploader('Upload an image')

if img_file_buffer is not None:

  # col1, col2 = st.columns(2)

  # with col1:
    ### Display the image user uploaded
    st.image(Image.open(img_file_buffer), caption="Here's the image you uploaded ☝️")

  # with col2:
    with st.spinner("Wait for it..."):
      ### Get bytes from the file buffer
      img = img_file_buffer.getvalue()
      res = requests.post(url + "/predict_cnn", files={'img': img})

      if res.status_code == 200:
        data=json.loads(res.text)
        text=data['message']
        st.text(f'Prediction ☝️ \n{text}')
      else:
        st.text(f'\n❌ Prediction failed - status code : {res.status_code}')
