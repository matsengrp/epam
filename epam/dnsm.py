"""
Here we define a mutation-selection model that is just about mutation vs no mutation, and is trainable.

We'll use these conventions:

* B is the batch size
* L is the max sequence length

"""

import logging
import math
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from epam.torch_common import pick_device, PositionalEncoding
import epam.molevol as molevol
import epam.sequences as sequences
from epam.sequences import translate_sequences

from shmple import AttentionModel


class PCPDataset(Dataset):
    def __init__(self, nt_parents, nt_children, shmple_model):
        assert len(nt_parents) == len(
            nt_children
        ), "Lengths of nt_parents and nt_children must be equal."
        pcp_count = len(nt_parents)

        for aa_parent, aa_child in zip(nt_parents, nt_children):
            if aa_parent == aa_child:
                raise ValueError(
                    f"Found an identical parent and child sequence: {aa_parent}"
                )

        aa_parents = translate_sequences(nt_parents)
        aa_children = translate_sequences(nt_children)
        self.max_aa_seq_len = max(len(seq) for seq in aa_parents)
        self.aa_parents_onehot = torch.zeros((pcp_count, self.max_aa_seq_len, 20))
        self.aa_subs_indicator_tensor = torch.zeros((pcp_count, self.max_aa_seq_len))

        mutation_freqs = [
            sequences.mutation_frequency(parent, child)
            for parent, child in zip(nt_parents, nt_children)
        ]

        print("predicting mutabilities and substitutions...")
        all_rates, all_subs_probs = shmple_model.predict_mutabilities_and_substitutions(
            nt_parents, mutation_freqs
        )

        print("consolidating this into substitution probabilities...")

        neutral_aa_mut_prob_l = []

        for nt_parent, rates, subs_probs in zip(nt_parents, all_rates, all_subs_probs):
            parent_idxs = sequences.nt_idx_tensor_of_str(nt_parent)

            # Making sure the rates tensor is of float type for numerical stability.
            mut_probs = 1.0 - torch.exp(-torch.tensor(rates).squeeze().float())
            normed_subs_probs = molevol.normalize_sub_probs(
                parent_idxs, torch.tensor(subs_probs).float()
            )

            neutral_aa_mut_prob = molevol.neutral_aa_mut_prob_v(
                parent_idxs.reshape(-1, 3),
                mut_probs.reshape(-1, 3),
                normed_subs_probs.reshape(-1, 3, 4),
            )

            # Ensure that all values are positive before taking the log later
            assert torch.all(neutral_aa_mut_prob > 0)

            pad_len = self.max_aa_seq_len - neutral_aa_mut_prob.shape[0]
            if pad_len > 0:
                neutral_aa_mut_prob = F.pad(neutral_aa_mut_prob, (0, pad_len), value=1)

            neutral_aa_mut_prob_l.append(neutral_aa_mut_prob)

        # Stacking along a new first dimension (dimension 0)
        self.log_neutral_aa_mut_probs = torch.log(torch.stack(neutral_aa_mut_prob_l))

        # padding_mask is True for padding positions.
        self.padding_mask = torch.ones(
            (pcp_count, self.max_aa_seq_len), dtype=torch.bool
        )

        for i, (aa_parent, aa_child) in enumerate(zip(aa_parents, aa_children)):
            aa_indices_parent = sequences.aa_idx_array_of_str(aa_parent)
            aa_seq_len = len(aa_parent)
            self.aa_parents_onehot[i, :aa_seq_len, aa_indices_parent] = 1
            self.aa_subs_indicator_tensor[i, :aa_seq_len] = torch.tensor(
                [p != c for p, c in zip(aa_parent, aa_child)], dtype=torch.float
            )
            self.padding_mask[i, :aa_seq_len] = False

    def __len__(self):
        return len(self.aa_parents_onehot)

    def __getitem__(self, idx):
        return {
            "aa_onehot": self.aa_parents_onehot[idx],
            "subs_indicator": self.aa_subs_indicator_tensor[idx],
            "padding_mask": self.padding_mask[idx],
            "log_neutral_aa_mut_probs": self.log_neutral_aa_mut_probs[idx],
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
        """Build a binary log selection matrix from a one-hot encoded parent sequence.

        Because we're predicting log of the selection factor, we don't use an
        activation function after the transformer.

        Parameters:
            parent_onehots: A tensor of shape (B, L, 20) representing the one-hot encoding of parent sequences.
            padding_mask: A tensor of shape (B, L) representing the padding mask for the sequence.

        Returns:
            A tensor of shape (B, L, 1) representing the log level of selection
            for each amino acid site.
        """

        parent_onehots = parent_onehots * math.sqrt(self.d_model)
        parent_onehots = self.pos_encoder(parent_onehots)

        # NOTE: not masking due to MPS bug
        out = self.encoder(parent_onehots)  # , src_key_padding_mask=padding_mask)
        out = self.linear(out)
        return out.squeeze(-1)

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
    shmple_weights_directory,
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

    shmple_model = AttentionModel(
        weights_dir=shmple_weights_directory, log_level=logging.WARNING
    )

    # It's important to make separate PCPDatasets for training and validation
    # because the maximum sequence length can differ between those two.
    train_set = PCPDataset(train_parents, train_children, shmple_model)
    val_set = PCPDataset(val_parents, val_children, shmple_model)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = TransformerBinarySelectionModel(
        nhead=nhead, dim_feedforward=dim_feedforward, layer_count=layer_count
    )

    bce_loss = nn.BCELoss()

    def complete_loss_fn(
        log_neutral_aa_mut_probs, log_selection_factors, aa_subs_indicator, padding_mask
    ):
        # Take the product of the neutral mutation probabilities and the selection factors.
        predictions = torch.exp(log_neutral_aa_mut_probs + log_selection_factors)

        predictions = predictions.masked_select(~padding_mask)
        aa_subs_indicator = aa_subs_indicator.masked_select(~padding_mask)

        # In the early stages of training, we can get probabilities > 1.0 because
        # of bad parameter initialization. We clamp the predictions to be between
        # 0 and 0.999 to avoid this: out of range predictions can make NaNs
        # downstream.
        out_of_range_prediction_count = torch.sum(predictions > 1.0)
        if out_of_range_prediction_count > 0:
            print(f"{out_of_range_prediction_count}\tpredictions out of range.")
        predictions = torch.clamp(predictions, min=0.0, max=0.999)

        return bce_loss(predictions, aa_subs_indicator)

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
            log_neutral_aa_mut_probs = batch["log_neutral_aa_mut_probs"].to(device)

            optimizer.zero_grad()
            log_selection_factors = model(aa_onehot, padding_mask)
            loss = complete_loss_fn(
                log_neutral_aa_mut_probs,
                log_selection_factors,
                aa_subs_indicator,
                padding_mask,
            )
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
                log_neutral_aa_mut_probs = batch["log_neutral_aa_mut_probs"].to(device)

                log_selection_factors = model(aa_onehot, padding_mask)
                val_loss += complete_loss_fn(
                    log_neutral_aa_mut_probs,
                    log_selection_factors,
                    aa_subs_indicator,
                    padding_mask,
                ).item()

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
