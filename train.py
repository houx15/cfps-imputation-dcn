from data.loader import load_data, get_data_loader
from model import CFPSModel
import torch
import numpy as np

from utils import LOGGER

# optim 和 loss
import torch.optim as optim
import torch.nn as nn
import time
import os

# 用tensorboardX
from tensorboardX import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

data, embedding = load_data(
    "data/raw/compiled_data_no_neutral.parquet", "data/raw/embeddings.npy"
)

data_loader = get_data_loader(data, embedding, 8, 24)

LOGGER.info(f"Data Loader Length: {len(data_loader)}")

model = CFPSModel(
    dim=50,
    embedding_1_len=data["pid"].nunique(),
    embedding_2_len=data["year"].nunique(),
    embedding_3_dim=4096,
    device=device,
)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.BCELoss()

curr_time = time.time()

# tensorboard记录loss
writer = SummaryWriter()


for e_id in range(10):

    for i, (X, y) in enumerate(data_loader):
        embedding_variable = [embedding[int(x)] for x in X[:, 2].numpy()]
        embedding_variable = torch.stack(embedding_variable)
        # 横着拼
        X = torch.cat([X, embedding_variable], dim=1)
        # to_device
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        # y_pred = y_pred.squeeze()
        # # 判断y_pred是否在0-1之间
        # if (
        #     torch.any(y_pred > 1)
        #     or torch.any(y_pred < 0)
        #     or torch.any(y < 0)
        #     or torch.any(y > 1)
        # ):
        #     print(y_pred)
        #     print(y)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_id = e_id * len(data_loader) + i
        loss_value = loss.item()
        writer.add_scalar("Loss", loss_value, global_id)

        if i % 1000 == 0:
            LOGGER.info(
                f"Iter {i} Loss: {loss.item()}, Cost Time {time.time() - curr_time} seconds"
            )
            curr_time = time.time()

    LOGGER.info(f"Epoch {e_id} Loss: {loss.item()}")
