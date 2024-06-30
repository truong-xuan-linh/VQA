import glob
import streamlit as st

from streamlit_image_select import image_select
import streamlit.components.v1 as components

# Trick to not init function multitime
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
st.markdown(hide_menu_style, unsafe_allow_html=True)

mapper = {
    "images/000000000645.jpg": "Đây là đâu",
    "images/000000000661.jpg": "Tốc độ tối đa trên đoạn đường này là bao nhiêu",
    "images/000000000674.jpg": "Còn bao xa nữa là tới Huế",
    "images/000000000706.jpg": "Cầu này dài bao nhiêu",
    "images/000000000777.jpg": "Chè khúc bạch giá bao nhiêu",
}

image = st.file_uploader(
    "Choose an image file",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)
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

if "image" in st.session_state:
    st.image(st.session_state.image)
    question = st.text_input("**Question:** ", value=st.session_state.question)
    visualize = True
    if question:
        answer, text_attention_html, images_visualize = (
            st.session_state.model.inference(
                st.session_state.image, question, visualize
            )
        )
        st.write(f"**Answer:** {answer}")

        if visualize:
            st.write("**Explanation**")
            col1, col2 = st.columns([1, 2])
            # st.markdown(text_attention_html, unsafe_allow_html=True)
            with col1:
                st.write("*Text Attention*")
                components.html(text_attention_html, height=960, scrolling=True)

            with col2:
                st.write("*Image Attention*")
                for image_visualize in images_visualize:
                    st.image(image_visualize)
