import streamlit as st
from st_pages import Page, show_pages, add_page_title

add_page_title('FAQ')

st.markdown('**Les diagnostics sont-ils fiables ?**')

st.write('Dermasaaj ne remplace pas un avis médical')


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
