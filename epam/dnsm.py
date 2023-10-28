"""
Here we define a mutation-selection model that is just about mutation vs no mutation, and is trainable.

We'll use these conventions:

* B is the batch size
* L is the max sequence length

"""

import math
import os

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

from tensorboardX import SummaryWriter

from epam.torch_common import pick_device, PositionalEncoding
import epam.sequences as sequences
from epam.sequences import translate_sequences


class PCPDataset(Dataset):
    def __init__(self, nt_parents, nt_children):
        # skipping storing nt sequences and branch lengths for now; see issue #31
        assert len(nt_parents) == len(
            nt_children
        ), "Lengths of nt_parents and nt_children must be equal."
        aa_parents = translate_sequences(nt_parents)
        aa_children = translate_sequences(nt_children)

        pcp_count = len(nt_parents)
        self.max_aa_seq_len = max(len(seq) for seq in aa_parents)
        self.aa_parents_onehot = torch.zeros((pcp_count, self.max_aa_seq_len, 20))
        self.aa_subs_indicator_tensor = torch.zeros((pcp_count, self.max_aa_seq_len))
        self.padding_mask = torch.ones(
            (pcp_count, self.max_aa_seq_len), dtype=torch.bool
        )

        for i, (parent, child) in enumerate(zip(aa_parents, aa_children)):
            aa_indices_parent = sequences.aa_idx_array_of_str(parent)
            seq_len = len(parent)
            self.aa_parents_onehot[i, :seq_len, aa_indices_parent] = 1
            self.aa_subs_indicator_tensor[i, :seq_len] = torch.tensor(
                [p != c for p, c in zip(parent, child)], dtype=torch.float
            )
            self.padding_mask[i, :seq_len] = False

    def __len__(self):
        return len(self.aa_parents_onehot)

    def __getitem__(self, idx):
        return {
            "aa_onehot": self.aa_parents_onehot[idx],
            "subs_indicator": self.aa_subs_indicator_tensor[idx],
            "padding_mask": self.padding_mask[idx],
        }


class TransformerBinarySelectionModel(nn.Module):
    """A transformer-based model for binary selection.

    This is a model that takes in a batch of one-hot encoded sequences and outputs a binary selection matrix.

    See forward() for details.
    """

    def __init__(
        self,
        nhead: int,
        dim_feedforward: int,
        layer_count: int,
        d_model: int = 20,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.device = pick_device()
        self.ntoken = 20
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(self.d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, layer_count)
        self.linear = nn.Linear(self.d_model, 1)

        self.to(self.device)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, parent_onehots: Tensor, padding_mask: Tensor) -> Tensor:
        """Build a binary selection matrix from a one-hot encoded parent sequence.

        Parameters:
            parent_onehots: A tensor of shape (B, L, 20) representing the one-hot encoding of parent sequences.
            padding_mask: A tensor of shape (B, L) representing the padding mask for the sequence.

        Returns:
            A tensor of shape (B, L, 1) representing the level of selection for each amino acid site.
        """

        parent_onehots = parent_onehots * math.sqrt(self.d_model)
        parent_onehots = self.pos_encoder(parent_onehots)

        # NOTE: not masking due to MPS bug
        out = self.encoder(parent_onehots)  # , src_key_padding_mask=padding_mask)
        out = self.linear(out)
        return torch.sigmoid(out).squeeze(-1)

    def p_substitution_of_aa_str(self, aa_str: str):
        """Do the forward method without gradients from an amino acid string and convert to numpy.

        Parameters:
            aa_str: A string of amino acids.

        Returns:
            A numpy array of the same length as the input string representing
            the level of selection for each amino acid site.
        """
        aa_onehot = torch.zeros((len(aa_str), 20))
        aa_indices_parent = sequences.aa_idx_array_of_str(aa_str)
        aa_onehot[:, aa_indices_parent] = 1

        # Create a padding mask with False values (i.e., no padding)
        padding_mask = torch.zeros(len(aa_str), dtype=torch.bool).to(self.device)

        with torch.no_grad():
            aa_onehot = aa_onehot.to(self.device)
            model_out = self(aa_onehot.unsqueeze(0), padding_mask.unsqueeze(0)).squeeze(
                0
            )

        return model_out.cpu().numpy()[: len(aa_str)]


def train_model(
    pcp_df,
    nhead,
    dim_feedforward,
    layer_count,
    batch_size=32,
    num_epochs=10,
    learning_rate=0.001,
    checkpoint_dir="./_checkpoints",
    log_dir="./_logs",
):
    print("preparing data...")
    nt_parents = pcp_df["parent"]
    nt_children = pcp_df["child"]

    train_len = int(0.8 * len(nt_parents))
    train_parents, val_parents = nt_parents[:train_len], nt_parents[train_len:]
    train_children, val_children = nt_children[:train_len], nt_children[train_len:]

    # It's important to make separate PCPDatasets for training and validation
    # because the maximum sequence length can differ between those two.
    train_set = PCPDataset(train_parents, train_children)
    val_set = PCPDataset(val_parents, val_children)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = TransformerBinarySelectionModel(
        nhead=nhead, dim_feedforward=dim_feedforward, layer_count=layer_count
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = model.device
    writer = SummaryWriter(log_dir=log_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print("training model...")

    for epoch in range(num_epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            aa_onehot = batch["aa_onehot"].to(device)
            aa_subs_indicator = batch["subs_indicator"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(aa_onehot, padding_mask)
            loss = criterion(outputs, aa_subs_indicator)
            loss.backward()
            optimizer.step()
            writer.add_scalar(
                "Training Loss", loss.item(), epoch * len(train_loader) + i
            )

        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                aa_onehot = batch["aa_onehot"].to(device)
                aa_subs_indicator = batch["subs_indicator"].to(device)
                padding_mask = batch["padding_mask"].to(device)

                outputs = model(aa_onehot, padding_mask)
                val_loss += criterion(outputs, aa_subs_indicator).item()

            avg_val_loss = val_loss / len(val_loader)
            writer.add_scalar("Validation Loss", avg_val_loss, epoch)

            # Save model checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_val_loss,
                },
                f"{checkpoint_dir}/model_epoch_{epoch}.pth",
            )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item()}, Validation Loss: {avg_val_loss}"
        )

    writer.close()

    return model
