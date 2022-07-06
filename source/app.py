import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from PIL import Image
import torch
from torchvision import transforms
import torch
import streamlit as st

from transformer import VisionTransformer

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Bean Image Classifier")
st.text("Provide URL of Bird Image: ")


# set title of app
st.title("Simple Image Classification Application")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = "jpg")


def predict(image):
    """Return top 5 predictions ranked by highest probability.
    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """

    model = VisionTransformer(img_size=config.IMG_SIZE,
                num_classes=config.NUM_CLASSES,
                hidden_size=config.HIDDEN_SIZE,
                in_channels=config.IN_CHANNELS,
                num_layers=config.NUM_LAYERS,
                num_attention_heads=config.NUM_ATTENTION_HEADS,
                linear_dim=config.LINEAR_DIM,
                dropout_rate=config.DROPOUT_RATE,
                attention_dropout_rate=config.ATTENTION_DROPOUT_RATE,
                eps=config.EPS,
                std_norm=config.STD_NORM)

    model = model.load_state_dict('metadata/model.pt')

    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225])])

    # load the image, pre-process it, and make predictions
    img = Image.open(image)
    x = transform(img)
    x = torch.unsqueeze(img, 0)
    model.eval()
    logits, attn_w = model(x)

    with open('../metadata/classes.txt', 'r') as f:
        classes = f.read().split('\n')

    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(logits, dim = 1)[0] * 100
    _, indices = torch.sort(out, descending = True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]


if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Processing...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])