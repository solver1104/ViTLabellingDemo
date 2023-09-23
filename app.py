import streamlit as st
import torch
import imageio as iio
import open_clip
from PIL import Image

CLIP_MODEL =   ['ViT-B-32', 'laion2b_e16']

@st.cache_resource
def load_model_components(CLIP_MODEL):
    return open_clip.create_model_and_transforms(CLIP_MODEL[0], pretrained=CLIP_MODEL[1])

@st.cache_resource
def load_tokenizer_components(CLIP_MODEL):
    return open_clip.get_tokenizer(CLIP_MODEL[0])

model, _, preprocess = load_model_components(CLIP_MODEL)
tokenizer = load_tokenizer_components(CLIP_MODEL)

PROMPT = "a photo of a "

st.title("Labelling Demonstration")
uploaded_file = st.file_uploader("Upload Image for Inference", type= ['png', 'jpg'])
labels = st.text_area("Possible Labels to Select From (Put each label in a new line)").split("\n")

if st.button('Run Inference'):
    if uploaded_file is not None and len(labels) != 0:
        image = iio.imread(uploaded_file, pilmode='RGB')
        img = preprocess(Image.fromarray(image.astype('uint8'), 'RGB')).unsqueeze(0)
        text = tokenizer([PROMPT + x for x in labels])

        image_features = model.encode_image(img)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        st.image(image)
        st.text(f"Predicted Class: {labels[torch.argmax(text_probs, dim = -1).item()]}")
    else:
        e = RuntimeError('Upload a file and possible labels.')
        st.exception(e)
