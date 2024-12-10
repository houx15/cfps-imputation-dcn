from .dcn import CrossNet
import torch
import torch.nn as nn
from utils import LOGGER


class CFPSModel(nn.Module):

    def __init__(
        self, dim, embedding_1_len, embedding_2_len, embedding_3_dim, device="cpu"
    ):
        LOGGER.info(
            f"Initializing CFPSModel with dim={dim}, embedding_1_len={embedding_1_len}, embedding_2_len={embedding_2_len}, embedding_3_dim={embedding_3_dim}, device={device}"
        )
        super(CFPSModel, self).__init__()
        self.embedding_pid = nn.Embedding(embedding_1_len, dim)
        self.embedding_year = nn.Embedding(embedding_2_len, dim)
        self.cross_net = CrossNet(3 * dim, layer_num=3, device=device)
        self.fc_question = nn.Linear(embedding_3_dim, dim)
        self.fc_out_1 = nn.Linear(3 * dim, 150)
        self.fc_out_2 = nn.Linear(150, 150)
        self.fc_out_3 = nn.Linear(150, 150)
        self.fc_out_4 = nn.Linear(150, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pid, year, question = x[:, 0], x[:, 1], x[:, 3:]
        # pid 转成 int
        pid = pid.long()
        year = year.long()
        pid = self.embedding_pid(pid)
        year = self.embedding_year(year)
        question = self.fc_question(question)
        x = torch.cat([pid, year, question], dim=1)
        x = self.cross_net(x)
        x = self.fc_out_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_out_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_out_3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_out_4(x)
        x = self.sigmoid(x)
        return x
