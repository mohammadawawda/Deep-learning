import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 30
DATA_DIR = "./data/"  # the folder of the data

# index of every allele and every amino acid
ALLELES_KEY = {'A0101': 0, 'A0201': 1, 'A0203': 2, 'A0207': 3, 'A0301': 4,
               'A2402': 5}
AA_DICT = {aa: i for i, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}


class SimpleClassifier(nn.Module):
    """
    a simple network with one hidden layer, taken from the Introduction_to_PyTorch2.ipynb tutorial
    """
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(in_features=num_inputs, out_features=num_hidden)
        self.act_fn = nn.ReLU()
        self.linear2 = nn.Linear(in_features=num_hidden, out_features=num_outputs)

    def forward(self, x):
        x = x.float()
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


class PeptideMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # every one of the 20 amino acids gets a learned vector of size 8
        self.aa_emb = nn.Embedding(num_embeddings=20, embedding_dim=8)
        # 9*8=72 inputs, then 128, 64 and 6 outputs
        self.net = nn.Sequential(
            nn.Linear(9*8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 6)   # one logit for every allele
        )

    def forward(self, pep_idx):
        """
        pep_idx: LongTensor of shape (B,9) with values from 0 to 19
        """
        x = self.aa_emb(pep_idx)            # (B,9,8)
        x = x.view(x.size(0), -1)           # (B,72)
        return self.net(x)                  # (B,6)


class LinearPeptideMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # same as PeptideMLP but without ReLU and dropout
        self.aa_emb = nn.Embedding(num_embeddings=20, embedding_dim=8)
        self.net   = nn.Sequential(
            nn.Linear(9*8, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 6)
        )

    def forward(self, pep_idx):
        x = self.aa_emb(pep_idx)            # (B,9,8)
        x = x.view(x.size(0), -1)           # (B,72)
        return self.net(x)                  # (B,6)

def train_model_epoch_eval(
    model,
    optimizer,
    train_loader,
    test_loader,
    loss_module,
    num_epochs,
    device):
    """
    trains the model for num_epochs and returns the train and test loss of every epoch
    the test loss is computed on the first batch of test_loader
    """
    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = []

        for data_inputs, data_labels in train_loader:
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)

            # the labels are converted to float like the model output
            loss = loss_module(preds, data_labels.float())
            epoch_train_loss.append(loss.item())

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        train_losses.append(np.mean(epoch_train_loss))

        with torch.no_grad():
            model.eval()

            (X_test, Y_test) = next(iter(test_loader))
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)

            test_preds = model(X_test).squeeze(dim=1)

            test_loss = loss_module(test_preds, Y_test.float())
            test_losses.append(test_loss.item())

        print(f"Epoch {epoch:02d} — train_loss: {train_losses[-1]:.4f}, "
              f"test_loss:  {test_losses[-1]:.4f}")

    return train_losses, test_losses


def parse_file(fp):
    """ parse peptide file and return peptides and labels """
    base  = os.path.basename(fp).split(".")[0]
    label = [0]*6
    if base != "negs":
        allele = base.split("_")[0]
        label[ALLELES_KEY[allele]] = 1

    peptides = []
    with open(fp) as f:
        for line in f:
            p = line.strip()
            if len(p) == 9:
                peptides.append(p)
    return peptides, [label]*len(peptides)


def plot_loss_curve(train_losses, test_losses, title, x_label, y_label):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

