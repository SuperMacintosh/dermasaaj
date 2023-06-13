import requests
import cv2
from dermaflow.params import *
import base64
import streamlit as st

"""
url='http://localhost:8000/test/'
fp='./raw_data/preproc_data/valid/MEL/ISIC_0000163.jpg'
img = cv2.imread(fp).tobytes()
img=base64.b64encode(img).decode()
data={'file_content':img,'candidate_gender': 1, 'candidate_age': 25}
response=requests.post(url, json=data)
# print(response.content)
"""
"""
url='http://localhost:8000'
img_file_buffer = st.file_uploader('Upload an image')
img_bytes = img_file_buffer.getvalue()
res = requests.post(url + "/upload_image", files={'img': img_bytes, 'candidate_gender': 1, 'candidate_age': 25})
"""

url = "http://localhost:8000/upload_items"
r = requests.post(
    url,
    files={"image": ("filename", open('./raw_data/preproc_data/valid/MEL/ISIC_0000163.jpg', "rb"))},
    data={"candidate_age": 25, "candidate_gender": 1},
)
print(r.content)
