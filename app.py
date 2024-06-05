import streamlit as st
from PIL import Image

#Trick to not init function multitime
if "model" not in st.session_state:
    print("INIT MODEL")
    from src.model import Model
    st.session_state.model = Model()
    print("DONE INIT MODEL")

st.set_page_config(page_title="VQA", layout="wide")
hide_menu_style = """
<style>
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html= True)


image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp", ])

if image:
    bytes_data = image.getvalue()
    with open("test.png", "wb") as f:
        f.write(bytes_data)
    f.close()
    st.session_state.image = "test.png"

if 'image' in st.session_state:
    st.image(st.session_state.image)
    question = st.text_input("Question: ")
    if question:
        answer = st.session_state.model.inference(st.session_state.image, question)
        st.write(f"Answer: {answer}")
        