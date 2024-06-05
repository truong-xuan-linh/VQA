import glob
import streamlit as st

from streamlit_image_select import image_select

#Trick to not init function multitime
# if "model" not in st.session_state:
#     print("INIT MODEL")
#     from src.model import Model
#     st.session_state.model = Model()
#     print("DONE INIT MODEL")

st.set_page_config(page_title="VQA", layout="wide")
hide_menu_style = """
<style>
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html= True)

mapper = {
        "images\\000000000008.jpg": "A",
        "images\\000000000012.jpg": "B",
        "images\\000000000016.jpg": "C",
        "images\\000000000019.jpg": "D",
        "images\\000000000181.jpg": "E"
}

image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp", ])
example = image_select("Examples", glob.glob("images/*.jpg"))

if image:
    bytes_data = image.getvalue()
    with open("test.png", "wb") as f:
        f.write(bytes_data)
    f.close()
    st.session_state.image = "test.png"
    st.session_state.question = ""
else:
    st.session_state.question = mapper[example]
    st.session_state.image = example

if 'image' in st.session_state:
    st.image(st.session_state.image)
    question = st.text_input("Question: ", value=st.session_state.question)
#     if question:
#         answer = st.session_state.model.inference(st.session_state.image, question)
#         st.write(f"Answer: {answer}")
        