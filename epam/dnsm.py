"""
Here we define a mutation-selection model that is just about mutation vs no mutation, and is trainable.

We'll use these conventions:

* B is the batch size
* L is the max sequence length

"""

import torch

from epam import molevol
from epam.models import TorchModel, MutSel



class TransformerBinarySelectionModel(nn.Module, TorchModel):
    """A transformer-based model for binary selection.
    
    This is a model that takes in a batch of one-hot encoded sequences and outputs a binary selection matrix.

    See forward() for details.
    """
    def __init__(self, model_dim, head_count, layer_count):
        super().__init__()
        # batch_first means that we have data laid out in terms of (batch, sequence_length, features)
        self.encoder_layer = nn.TransformerEncoderLayer(model_dim, head_count, batch_first=True)
        # This just makes a stack of layer_count of the encoder_layer.
        self.encoder = nn.TransformerEncoder(self.encoder_layer, layer_count)
        self.linear = nn.Linear(model_dim, 1)

    def forward(self, src):
        """Build a binary selection matrix from a one-hot encoded parent sequence.

        Parameters:
            parent_onehots: A tensor of shape (B, L, 20) representing the one-hot encoding of parent sequences.

        Returns:
            A tensor of shape (B, L, 1) representing the level of selection for each amino acid site.
        """
        out = self.encoder(src)
        out = self.linear(out)
        return torch.sigmoid(out).squeeze(-1)

    def nograd_forward_of_aa_string(aa_str):
        """Do the forward method without gradients from an amino acid string.
        """
        parent_aa_onehot = sequences.aa_one_hot(parent)

        with torch.no_grad():
            selection_matrix = self.model(parent_aa_onehot)
        

class TorchBinaryMutSel(MutSel):
    """A mutation selection model that is just about mutation vs no mutation, and is trainable."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model


    def build_selection_matrix_from_parent(self, parent):
        # We need to take our binary selection matrix and turn it into a selection matrix which gives the same weight to each off-diagonal element.
        selection_matrix = self.model.nograd_forward_of_aa_string(parent)

    def bselection_matrices_from_parent_aa_onehots(self, parent_onehots: Tensor) -> Tensor:
        """Build a binary selection matrix from a one-hot encoded parent sequence.

        Parameters:
            parent_onehots: A tensor of shape (B, L, 20) representing the one-hot encoding of parent sequences.

        Returns:
            A tensor of shape (B, L, 1) representing the level of selection for each amino acid site.
        """
        pass
