"""
Here we define a mutation-selection model that is just about mutation vs no mutation, and is trainable.

We'll use these conventions:

* B is the batch size
* L is the max sequence length

"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch import Tensor

from epam.models import TorchModel, MutSel
import epam.sequences as sequences
from epam.sequences import translate_sequences


class PCPDataset(Dataset):
    def __init__(self, nt_parents, nt_children):
        # skipping storing nt sequences and branch lengths for now; see issue #31
        assert len(nt_parents) == len(nt_children), "Lengths of nt_parents and nt_children must be equal."
        aa_parents = translate_sequences(self.nt_parents)
        aa_children = translate_sequences(self.nt_children)

        pcp_count = len(nt_parents)
        self.max_aa_seq_len = max(len(seq) for seq in self.aa_parents)
        self.aa_parents_onehot = torch.zeros((pcp_count, self.max_aa_seq_len, 20))
        aa_subs_indicator_tensor = torch.zeros((pcp_count, self.max_aa_seq_len))
        padding_mask = torch.ones((pcp_count, self.max_aa_seq_len), dtype=torch.bool)

        for i, (parent, child) in enumerate(zip(aa_parents, aa_children)):
            aa_indices_parent = sequences.aa_idx_array_of_str(parent)
            seq_len = len(parent)
            self.aa_parents_onehot[i, :seq_len, aa_indices_parent] = 1
            aa_subs_indicator_tensor[i, :seq_len] = torch.tensor([p != c for p, c in zip(parent, child)], dtype=torch.float)
            padding_mask[i, :seq_len] = False

    def __len__(self):
        return len(self.aa_parents_onehot)

    def __getitem__(self, idx):
        return {
            'aa_onehot': self.aa_parents_onehot[idx],
            'subs_indicator': self.aa_subs_indicator_tensor[idx],
            'padding_mask': self.padding_mask[idx]
        }


class TransformerBinarySelectionModel(nn.Module, TorchModel):
    """A transformer-based model for binary selection.

    This is a model that takes in a batch of one-hot encoded sequences and outputs a binary selection matrix.

    See forward() for details.
    """

    def __init__(self, nhead, dim_feedforward, layer_count):
        super().__init__()
        # batch_first means that we have data laid out in terms of (batch, sequence_length, features)
        self.encoder_layer = nn.TransformerEncoderLayer(
            20, nhead, dim_feedforward, batch_first=True
        )
        # This just makes a stack of layer_count of the encoder_layer.
        self.encoder = nn.TransformerEncoder(self.encoder_layer, layer_count)
        self.linear = nn.Linear(dim_feedforward, 1)
        self.to(self.device)

    def forward(self, parent_onehots: Tensor):
        """Build a binary selection matrix from a one-hot encoded parent sequence.

        Parameters:
            parent_onehots: A tensor of shape (B, L, 20) representing the one-hot encoding of parent sequences.

        Returns:
            A tensor of shape (B, L, 1) representing the level of selection for each amino acid site.
        """
        out = self.encoder(parent_onehots)
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

        with torch.no_grad():
            aa_onehot = aa_onehot.to(self.device)
            model_out = self.model(aa_onehot.unsqueeze(0)).squeeze(0)

        return model_out.cpu().numpy()[: len(aa_str)]


class WrappedBinaryMutSel(MutSel):
    """A mutation selection model that is built from a model that has a `p_substitution_of_aa_str` method."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def build_selection_matrix_from_parent(
        self: TransformerBinarySelectionModel, parent: str
    ):
        # We need to take our binary selection matrix and turn it into a selection matrix which gives the same weight to each off-diagonal element.
        p_substitution = self.model.p_substitution_of_aa_str(parent)
        parent_idxs = sequences.aa_idx_array_of_str(parent)

        # make a np array with the same number of rows as the length of p_substitution
        # and the same number of columns as the number of amino acids
        selection_matrix = np.zeros((len(p_substitution), 20))

        # Set each row to p_substitution/19, which is the probability of the
        # corresponding site mutating to a given alternative amino acid.
        selection_matrix[:, :] = p_substitution[:, np.newaxis] / 19.0

        # Set "diagonal" elements to 1 - p_substitution for each corresponding amino
        # acid in the parent, where "diagonal means keeping the same amino acid.
        selection_matrix[np.arange(len(parent_idxs)), parent_idxs] = (
            1.0 - p_substitution
        )

        return selection_matrix
