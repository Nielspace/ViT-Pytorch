import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from PIL import Image
import torch
from torchvision import transforms
import torch
import streamlit as st

from embeddings import Embeddings
from attention_block import Block
from linear import Mlp
from attention import Attention
from transformer import VisionTransformer, Transformer, Encoder

from config import Config

config = Config()

st.set_option("deprecation.showfileUploaderEncoding", False)
st.title("Bird Image Classifier")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type="jpg")


def predict(image):
    """Return top 5 predictions ranked by highest probability.
    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """

    model = torch.load("metadata/models/model.pth", map_location=torch.device("cpu"))

    # transform the input image through resizing, normalization
    transform = transforms.Compose(
        [
            transforms.Resize(config.IMG_SIZE),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # load the image, pre-process it, and make predictions
    img = Image.open(image)
    x = transform(img)
    x = torch.unsqueeze(x, 0)
    model.eval()
    logits, attn_w = model(x)

    with open("metadata/classes.txt", "r") as f:
        classes = f.read().split("\n")

    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(logits, dim=1)[0] * 100
    _, indices = torch.sort(logits, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]


if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Processing...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write(f"Prediction:= {i[0]} score {i[1]:.2f}")
