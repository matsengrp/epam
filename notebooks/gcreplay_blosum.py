import numpy as np
import pandas as pd
import h5py
from esm import pretrained
from netam.sequences import (
    AA_STR_SORTED,
)
from epam.models import BLOSUM62

replay_igh = "EVQLQESGPSLVKPSQTLSLTCSVTGDSITSGYWNWIRKFPGNKLEYMGYISYSGSTYYNPSLKSRISITRDTSKNQYYLQLNSVTTEDTATYYCARDFDVWGAGTTVTVSS"
replay_igk = "DIVMTQSQKFMSTSVGDRVSVTCKASQNVGTNVAWYQQKPGQSPKALIYSASYRYSGVPDRFTGSGSGTDFTLTISNVQSEDLAEYFCQQYNSYPLTFGSGTKLEIKR"

sequences_aa = [replay_igh, replay_igk]
blosum_sel_model = BLOSUM62()

with h5py.File(output_hdf5, "w") as outfile:
    # attributes related to PCP data file
    outfile.attrs["data_set"] = "replay_igh_igk"
    outfile.attrs["model_name"] = f"BLOSUM62"

    for seq in range(len(sequences_aa)):

        parent = sequences_aa[seq]
        matrix = blosum_sel_model.aaprobs_of_parent_child_pair(sequences_aa[seq])

        outfile.create_dataset(
            f"{parent}", data=matrix, compression="gzip", compression_opts=4
        )