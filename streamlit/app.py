import streamlit as st
import requests
from PIL import Image
from st_pages import Page, show_pages, add_page_title
from pathlib import Path


st.set_page_config(
    page_title="Votre assistant cutané",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",

)



path = Path(__file__).parents[1]

#pages = st.source_util.get_pages('app.py')
show_pages(
    [
        Page("streamlit/app.py", "Lesion prediction", ":mag:"),
        Page("streamlit/pages/faq.py", "FAQ", ":question:"),
    ]
)



def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(logo.png);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "My Company Name";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
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
image = Image.open("/app/dermasaaj/streamlit/dermacare-logo.png") #Brand logo image (optional)
#st.sidebar.image("illlustr3.png", use_column_width=True)

#Create two columns with different width
col1, col2 = st.columns( [0.8, 0.12])
with col1:               # To display the header text using css style
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; font-weight: 700;color: #FF9633;}
    #colorbis #ff7412
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Dermacare votre assistant cutané</p>', unsafe_allow_html=True)

with col2:               # To display brand logo
    st.image(image,  width=150)

age = st.number_input('Votre âge ?', min_value=0, max_value=100, step=1)


sex = st.radio('Votre sexe', ('Homme', 'Femme'),horizontal=True)


lesion = st.selectbox('Localisation de la lésion', ['Anterior torso','head/neck','lateral torso','lower extremity','oral/genital','palms/soles','posterior torso','upper extremity','other'])

st.set_option('deprecation.showfileUploaderEncoding', False)

###########################################

uploaded_file = st.file_uploader("Choisissez une photo de votre lésion", type=['png', 'jpg'] )



if uploaded_file is not None:



    #image = Image.open(uploaded_file)
    st.image(uploaded_file, width=400,caption='Lésion cutanée')

    bytes_data = uploaded_file.getvalue()



#############################################################

col1, col2,col3,col4,col5= st.columns(5)
if col3.button('Valider'):
    #params = {
    #'sex': sex,
    #'age': age,
    #'lesion':lesion
    #}
    url=https://api-kpmfnijgja-ew.a.run.app
    res = requests.post(url, files={'img': bytes_data}, headers=headers)
    if res.status_code == 200:
        st.write(res.content)
        response=res.json()
        #st.write(f"{round(response['proba'],3)} pourcent de chance d'être en bonne santé")
        st.write(response['message'])

    else:
        st.markdown("**Oops**, something went wrong 😓 Please try again.")
        print(res.status_code, res.content)



    pass
else:
    pass




#headers = {
 #   "Authorization" : "xxx",
  #  "Content-Type": "application/json"
#}





hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
