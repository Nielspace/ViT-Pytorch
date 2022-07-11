from PIL import Image
import cv2
import numpy as np
import torch
import torchvision

import matplotlib.pyplot as plt

from embeddings import Embeddings
from attention_block import Block
from linear import Mlp
from attention import Attention
from transformer import VisionTransformer, Transformer, Encoder
from preprocessing import Dataset

from config import Config

config = Config()

PATH = "metadata/Abbott's_babbler_(Malacocincla_abbotti).jpg"


def attention_viz(model, test_data, img_path=PATH, device="mps"):

    classes = test_data.dataset.classes
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    im = Image.open(PATH)
    x = transform(im)
    x = x.unsqueeze(0)

    logits, att = model(x)
    probs = torch.nn.Softmax(dim=-1)(logits)
    att_mat = torch.stack(att).squeeze(1)
    # # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # # To account for residual connections, we add an identity matrix to the
    # # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat.cpu() + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()

    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    result = (mask * im).astype("uint8")

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title("Original")
    ax2.set_title("Attention Map")
    _ = ax1.imshow(im)
    _ = ax2.imshow(result)
    plt.savefig("metadata/results/attn.png", dpi=300)

    top5 = torch.argsort(probs.cpu(), dim=-1, descending=True)
    print("Prediction Label and Attention Map!\n")
    for idx in top5[0, :5]:
        print(f"{probs[0, idx.item()]:.5f} : {classes[idx.item()]}", end="")


if __name__ == "__main__":
    train_data, val_data, test_data = Dataset(
        config.BATCH_SIZE, config.IMG_SIZE, config.DATASET_SAMPLE
    )
    model = torch.load("metadata/models/model.pth", map_location=torch.device("cpu"))
    attention_viz(model, test_data, PATH)
