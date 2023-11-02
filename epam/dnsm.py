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
from epam.models import WrappedBinaryMutSel

from shmple import AttentionModel


class PCPDataset(Dataset):
    def __init__(self, nt_parents, nt_children, shmple_model):
        self.nt_parents = nt_parents
        # TODO I don't think we need the nt_children, but I'm leaving it in for now.
        self.nt_children = nt_children
        self.shmple_model = shmple_model

        assert len(self.nt_parents) == len(self.nt_children)
        pcp_count = len(self.nt_parents)

        for parent, child in zip(self.nt_parents, self.nt_children):
            if parent == child:
                raise ValueError(
                    f"Found an identical parent and child sequence: {parent}"
                )

        aa_parents = translate_sequences(self.nt_parents)
        aa_children = translate_sequences(self.nt_children)
        self.max_aa_seq_len = max(len(seq) for seq in aa_parents)
        self.aa_parents_onehot = torch.zeros((pcp_count, self.max_aa_seq_len, 20))
        self.aa_subs_indicator_tensor = torch.zeros((pcp_count, self.max_aa_seq_len))

        # padding_mask is True for padding positions.
        self.padding_mask = torch.ones(
            (pcp_count, self.max_aa_seq_len), dtype=torch.bool
        )

        for i, (aa_parent, aa_child) in enumerate(zip(aa_parents, aa_children)):
            aa_indices_parent = sequences.aa_idx_array_of_str(aa_parent)
            aa_seq_len = len(aa_parent)
            self.aa_parents_onehot[i, torch.arange(aa_seq_len), aa_indices_parent] = 1
            self.aa_subs_indicator_tensor[i, :aa_seq_len] = torch.tensor(
                [p != c for p, c in zip(aa_parent, aa_child)], dtype=torch.float
            )
            self.padding_mask[i, :aa_seq_len] = False

        # Make initial branch lengths (will get optimized later).
        self._branch_lengths = [
            sequences.mutation_frequency(parent, child)
            for parent, child in zip(self.nt_parents, self.nt_children)
        ]
        self.update_neutral_aa_mut_probs()

    @property
    def branch_lengths(self):
        return self._branch_lengths

    @branch_lengths.setter
    def branch_lengths(self, new_branch_lengths):
        self._branch_lengths = new_branch_lengths
        self.update_neutral_aa_mut_probs()

    def update_neutral_aa_mut_probs(self):
        print("predicting mutabilities and substitutions...")
        (
            all_rates,
            all_subs_probs,
        ) = self.shmple_model.predict_mutabilities_and_substitutions(
            self.nt_parents, self._branch_lengths
        )

        print("consolidating this into substitution probabilities...")

        neutral_aa_mut_prob_l = []

        for nt_parent, rates, subs_probs in zip(
            self.nt_parents, all_rates, all_subs_probs
        ):
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
            # TODO log these?
            neutral_aa_mut_prob = torch.clamp(neutral_aa_mut_prob, min=1e-6)

            pad_len = self.max_aa_seq_len - neutral_aa_mut_prob.shape[0]
            if pad_len > 0:
                # TODO: does this value matter? If we are masking correctly it shoudln't.
                neutral_aa_mut_prob = F.pad(
                    neutral_aa_mut_prob, (0, pad_len), value=1e-6
                )

            neutral_aa_mut_prob_l.append(neutral_aa_mut_prob)

        self.log_neutral_aa_mut_probs = torch.log(torch.stack(neutral_aa_mut_prob_l))

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
        # Have to do the permutation because the positional encoding expects the
        # sequence length to be the first dimension.
        parent_onehots = self.pos_encoder(parent_onehots.permute(1, 0, 2)).permute(
            1, 0, 2
        )

        # NOTE: not masking due to MPS bug
        out = self.encoder(parent_onehots)  # , src_key_padding_mask=padding_mask)
        out = self.linear(out)
        return out.squeeze(-1)

    def selection_factors_of_aa_str(self, aa_str: str):
        """Do the forward method without gradients from an amino acid string and convert to numpy.

        Parameters:
            aa_str: A string of amino acids.

        Returns:
            A numpy array of the same length as the input string representing
            the level of selection for each amino acid site.
        """
        aa_onehot = sequences.aa_onehot_tensor_of_str(aa_str)

        # Create a padding mask with False values (i.e., no padding)
        padding_mask = torch.zeros(len(aa_str), dtype=torch.bool).to(self.device)

        with torch.no_grad():
            aa_onehot = aa_onehot.to(self.device)
            model_out = self(aa_onehot.unsqueeze(0), padding_mask.unsqueeze(0)).squeeze(
                0
            )
            final_out = torch.exp(model_out)
            # TODO think about if we can interpret model outputs greater than 1; in any case this makes problems if we have mutation*selection > 1
            final_out = torch.clamp(final_out, min=1e-6, max=0.999)

        return final_out.cpu().numpy()[: len(aa_str)]


class DNSMBurrito:
    def __init__(
        self,
        pcp_df,
        shmple_weights_directory,
        dnsm,
        batch_size=32,
        learning_rate=0.001,
        checkpoint_dir="./_checkpoints",
        log_dir="./_logs",
    ):
        self.dnsm = dnsm
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir

        print("preparing data...")
        nt_parents = pcp_df["parent"]
        nt_children = pcp_df["child"]

        train_len = int(0.8 * len(nt_parents))
        train_parents, val_parents = nt_parents[:train_len], nt_parents[train_len:]
        train_children, val_children = nt_children[:train_len], nt_children[train_len:]

        shmple_model = AttentionModel(
            weights_dir=shmple_weights_directory, log_level=logging.WARNING
        )
        self.wrapped_dnsm = WrappedBinaryMutSel(self.dnsm, shmple_weights_directory)

        # It's important to make separate PCPDatasets for training and validation
        # because the maximum sequence length can differ between those two.
        self.train_set = PCPDataset(train_parents, train_children, shmple_model)
        self.val_set = PCPDataset(val_parents, val_children, shmple_model)

        self.optimizer = optim.Adam(self.dnsm.parameters(), lr=learning_rate)
        self.device = self.dnsm.device
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.bce_loss = nn.BCELoss()

    def complete_loss_fn(
        self,
        log_neutral_aa_mut_probs,
        log_selection_factors,
        aa_subs_indicator,
        padding_mask,
    ):
        # Take the product of the neutral mutation probabilities and the selection factors.
        predictions = torch.exp(log_neutral_aa_mut_probs + log_selection_factors)

        predictions = predictions.masked_select(~padding_mask)
        aa_subs_indicator = aa_subs_indicator.masked_select(~padding_mask)

        # In the early stages of training, we can get probabilities > 1.0 because
        # of bad parameter initialization. We clamp the predictions to be between
        # 0 and 0.999 to avoid this: out of range predictions can make NaNs
        # downstream.
        # out_of_range_prediction_count = torch.sum(predictions > 1.0)
        # if out_of_range_prediction_count > 0:
        #     print(f"{out_of_range_prediction_count}\tpredictions out of range.")
        predictions = torch.clamp(predictions, min=1e-6, max=0.999)

        return self.bce_loss(predictions, aa_subs_indicator)

    def loss_of_batch(self, batch):
        aa_onehot = batch["aa_onehot"].to(self.device)
        aa_subs_indicator = batch["subs_indicator"].to(self.device)
        padding_mask = batch["padding_mask"].to(self.device)
        log_neutral_aa_mut_probs = batch["log_neutral_aa_mut_probs"].to(self.device)
        log_selection_factors = self.dnsm(aa_onehot, padding_mask)
        return self.complete_loss_fn(
            log_neutral_aa_mut_probs,
            log_selection_factors,
            aa_subs_indicator,
            padding_mask,
        )

    def compute_avg_loss(self, data_loader):
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                total_loss += self.loss_of_batch(batch).item()
        return total_loss / len(data_loader)

    def train(self, num_epochs=10):
        train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
        # Record epoch 0
        self.dnsm.eval()
        avg_train_loss_epoch_zero = self.compute_avg_loss(train_loader)
        avg_val_loss_epoch_zero = self.compute_avg_loss(val_loader)
        self.writer.add_scalar("Training Loss", avg_train_loss_epoch_zero, 0)
        self.writer.add_scalar("Validation Loss", avg_val_loss_epoch_zero, 0)
        print(
            f"Epoch [0/{num_epochs}], Training Loss: {avg_train_loss_epoch_zero}, Validation Loss: {avg_val_loss_epoch_zero}"
        )

        print("training model...")

        for epoch in range(num_epochs):
            self.dnsm.train()
            for i, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                loss = self.loss_of_batch(batch)
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar(
                    "Training Loss", loss.item(), epoch * len(train_loader) + i
                )

            # Validation Loop
            self.dnsm.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    val_loss += self.loss_of_batch(batch).item()

                avg_val_loss = val_loss / len(val_loader)
                self.writer.add_scalar("Validation Loss", avg_val_loss, epoch)

                # Save model checkpoint
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.dnsm.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": avg_val_loss,
                    },
                    f"{self.checkpoint_dir}/model_epoch_{epoch}.pth",
                )

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item()}, Validation Loss: {avg_val_loss}"
            )

        self.writer.close()

    def optimize_branch_lengths(self):
        for dataset in [self.train_set, self.val_set]:
            dataset.branch_lengths = self.wrapped_dnsm.find_optimal_branch_lengths(
                dataset.nt_parents, dataset.nt_children, dataset.branch_lengths
            )
