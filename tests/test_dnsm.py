
import pandas as pd
from epam.dnsm import train_model


def test_dnsm():
    """Just make sure that the model trains."""
    pcp_df = pd.read_csv("~/data/wyatt-10x-1p5m_pcp_2023-10-07.first100.csv")

    # filter out rows of pcp_df where the parent and child sequences are identical
    pcp_df = pcp_df[pcp_df["parent"] != pcp_df["child"]]
    print(f"After filtering out identical PCPs, we have {len(pcp_df)} PCPs.")

    nhead = 4
    dim_feedforward = 2048
    layer_count = 3
    model = train_model(pcp_df, nhead=nhead, dim_feedforward=dim_feedforward, layer_count=layer_count, batch_size=32, num_epochs=10, learning_rate=0.001, checkpoint_dir="./_checkpoints", log_dir="./_logs")