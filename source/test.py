import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from embeddings import Embeddings
from attention_block import Block
from linear import Mlp
from attention import Attention
from transformer import VisionTransformer, Transformer, Encoder
from preprocessing import Dataset
from attention_viz import attention_viz

import matplotlib.pyplot as plt
import seaborn as sns


from config import Config
from tqdm import tqdm

config = Config()

train_dir = "../../../data/birds/train/"
val_dir = "../../../data/birds/valid/"
test_dir = "../../../data/birds/test/"

train_data, val_data, test_data = Dataset(
    config.BATCH_SIZE, config.IMG_SIZE, config.DATASET_SAMPLE
)

total_samples = len(test_data.dataset)
correct_samples = 0
total_loss = 0

confusion_matrix = torch.zeros(400, 400)


def test(model, test_data):
    c = 0
    total = 0
    device = "mps"
    n_epochs = 2
    total_step = len(train_data)
    iterations = 12

    logits_ = []
    ground = []

    acc__ = []
    loss__ = []
    confusion_matrix = torch.zeros(400, 400)
    with torch.no_grad():
        model.eval()
        for epoch in range(1, n_epochs):
            with tqdm(test_data, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                for batch_idx, (data, target) in enumerate(tepoch):
                    total_samples = len(train_data.dataset)
                    best_accuracy = 0
                    # device
                    model = model.to(device)
                    x = data.to(device)
                    y = target.to(device)
                    logits, attn_weights = model(x)
                    proba = F.log_softmax(logits, dim=1)
                    loss = F.nll_loss(proba, y, reduction="sum")
                    _, pred = torch.max(logits, dim=1)
                    c += torch.sum(pred == y).item()
                    total += target.size(0)
                    tepoch.set_postfix(loss=loss.item(), accuracy=(100 * c / total))

                    acc__.append((100 * c / total))
                    loss__.append(loss.item())
                    logits_.append(logits)
                    ground.append(y)

                    for t, p in zip(y.view(-1), pred.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

    return acc__, loss__, confusion_matrix


if __name__ == "__main__":
    PATH = "metadata/Abbott's_babbler_(Malacocincla_abbotti).jpg"
    model = torch.load("metadata/models/model.pth", map_location=torch.device("cpu"))
    attention_viz(model, test_data, PATH)

    acc__, loss__, confusion_matrix = test(model, test_data)

    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion_matrix.numpy())
    plt.savefig("metadata/results/confusion_matrix.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.plot(acc__, label="accuracy")
    plt.plot(loss__, label="loss")
    plt.legend()
    plt.savefig("metadata/results/accuracy_loss.png", dpi=300)
    plt.show()
