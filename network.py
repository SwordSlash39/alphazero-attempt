import torch
import torch.nn as nn
from Board import chessboard

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(11, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        self.rnn = nn.Sequential(
            nn.Linear(64 * 8 * 8, 4096)
        )
        self.output_fn = nn.LogSoftmax(dim=0)

    def convStrToIndex(self, move: str) -> int:
        return 8**3 * (ord(move[0]) - ord('a')) + 8**2 * (ord(move[1]) - ord('1')) + 8 * (ord(move[2]) - ord('a')) + (ord(move[3]) - ord('1'))

    def forward(self, x, board: chessboard):
        x = x.unsqueeze(0)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.rnn(x)

        legal = torch.tensor([self.convStrToIndex(str(m)) for m in board.get_legal_moves()]).to(device)
        actor_output = x[0][legal]
        return self.output_fn(actor_output)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(11, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        self.rnn = nn.Sequential(
            nn.Linear(64 * 8 * 8, 2048),
            nn.SiLU(),
            nn.Linear(2048, 1)
        )

    def forward(self, x, single_input=True):
        if single_input:
            x = x.unsqueeze(0)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.rnn(x)
        return x