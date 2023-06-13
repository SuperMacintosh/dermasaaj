import streamlit as st
import requests
import datetime
import pandas as pd
from PIL import Image
from st_pages import Page, show_pages, add_page_title
from tempfile import NamedTemporaryFile
import numpy as np
import base64


st.set_page_config(
    page_title="Votre assistant cutané",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",

)
#pages = st.source_util.get_pages('app.py')
show_pages(
    [
        Page("app.py", "Lesion prediction", ":mag:"),
        Page("pages/faq.py", "FAQ", ":question:"),
    ]
)






#st.title("Dermasaaj votre assistant cutané")
def add_logo():
    st.markdown(
        """
        <style>
            .font {

                background-repeat: no-repeat;
                padding-top: 20px;
                background-position: 20px 20px;
            }

        </style>
        """,
        unsafe_allow_html=True,
    )
add_logo()
image = Image.open(r'logo.png') #Brand logo image (optional)

#Create two columns with different width
col1, col2 = st.columns( [0.8, 0.18])
with col1:               # To display the header text using css style
    st.markdown(""" <style> .font {
    font-size:32px ; font-family: 'Cooper Black'; font-weight: bold;color: #FF9633;}
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Dermasaaj votre assistant cutané</p>', unsafe_allow_html=True)

with col2:               # To display brand logo
    st.image(image,  width=150)

age = st.number_input('Votre âge ?', min_value=0, max_value=100, step=1)

#year = st.selectbox('Année de naissance', range(1920, 2021))

sex = st.radio('Votre sexe', ('Homme', 'Femme'),horizontal=True)


lesion = st.selectbox('Localisation de la lésion', ['Anterior torso','head/neck','lateral torso','lower extremity','oral/genital','palms/soles','posterior torso','upper extremity','other'])

st.set_option('deprecation.showfileUploaderEncoding', False)

###########################################

uploaded_file = st.file_uploader("Choisissez une photo de votre lésion", type=['png', 'jpg'] )



if uploaded_file is not None:



    #image = Image.open(uploaded_file)
    st.image(uploaded_file, width=400,caption='Lésion cutanée')

    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)
    #st.write(image_uploaded)

#############################################################

col1, col2,col3,col4,col5= st.columns(5)
if col3.button('Valider'):
    # print is visible in the server output, not in the page
    #response = requests.get(url,params)
    #response=response.json()
    #st.write(f"{round(response['proba'],3)} pourcent de chance d'être en bonne santé")

    st.write(bytes_data)

    pass
else:
    pass
params = {
    #'sex': sex,
    #'age': age,
    #'lesion':lesion
}

url="https://www.gscloudderamsaj.com"
#response = requests.get(url,params)
#response=response.json()
#st.write(f"Le prix de la course est de  {round(response['fare'],3)} $")


#headers = {
 #   "Authorization" : "xxx",
  #  "Content-Type": "application/json"
#}


#res = requests.post(url, files={'img': bytes_data}, data=params, headers=headers)
#if res.status_code == 200:
 #   st.write(res.content)
#else:
 #   st.markdown("**Oops**, something went wrong 😓 Please try again.")
  #  print(res.status_code, res.content)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
