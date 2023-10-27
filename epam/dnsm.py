"""
Here we define a mutation-selection model that is just about mutation vs no mutation, and is trainable.

We'll use these conventions:

* B is the batch size
* L is the max sequence length

"""

import numpy as np
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

from tensorboardX import SummaryWriter

from epam.models import pick_device
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

    def __init__(self, nhead, dim_feedforward, layer_count):
        super().__init__()
        self.device = pick_device()
        # batch_first means that we have data laid out in terms of (batch, sequence_length, features)
        self.encoder_layer = nn.TransformerEncoderLayer(
            20, nhead, dim_feedforward, batch_first=True
        )
        # This just makes a stack of layer_count of the encoder_layer.
        self.encoder = nn.TransformerEncoder(self.encoder_layer, layer_count)
        self.linear = nn.Linear(dim_feedforward, 1)
        self.to(self.device)

    def forward(self, parent_onehots: Tensor, padding_mask: Tensor):
        """Build a binary selection matrix from a one-hot encoded parent sequence.

        Parameters:
            parent_onehots: A tensor of shape (B, L, 20) representing the one-hot encoding of parent sequences.
            padding_mask: A tensor of shape (B, L) representing the padding mask for the sequence.

        Returns:
            A tensor of shape (B, L, 1) representing the level of selection for each amino acid site.
        """
        out = self.encoder(parent_onehots, src_key_padding_mask=padding_mask)
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
        aa_onehot = sequences.aa_one_hot_tensor(aa_str)
        
        # Create a padding mask with False values (i.e., no padding)
        padding_mask = torch.zeros(len(aa_str), dtype=torch.bool).to(self.device)

        with torch.no_grad():
            aa_onehot = aa_onehot.to(self.device)
            model_out = self.model(aa_onehot.unsqueeze(0), padding_mask.unsqueeze(0)).squeeze(0)

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
    dataset = PCPDataset(nt_parents, nt_children)

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    model = TransformerBinarySelectionModel(nhead, dim_feedforward, layer_count)

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
            for aa_onehot, aa_subs_indicator, padding_mask in val_loader:
                aa_onehot, aa_subs_indicator, padding_mask = (
                    aa_onehot.to(device),
                    aa_subs_indicator.to(device),
                    padding_mask.to(device),
                )
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
