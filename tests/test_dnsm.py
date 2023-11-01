import pandas as pd
from epam.dnsm import TransformerBinarySelectionModel, DNSMBurrito


def test_dnsm():
    """Just make sure that the model trains."""
    pcp_df = pd.read_csv("~/data/wyatt-10x-1p5m_pcp_2023-10-07.first100.csv")
    shmple_weights_directory = "data/shmple_weights/my_shmoof"

    # filter out rows of pcp_df where the parent and child sequences are identical
    pcp_df = pcp_df[pcp_df["parent"] != pcp_df["child"]]
    print(f"After filtering out identical PCPs, we have {len(pcp_df)} PCPs.")

    dnsm = TransformerBinarySelectionModel(nhead=2, dim_feedforward=256, layer_count=2)

    burrito = DNSMBurrito(
        pcp_df,
        shmple_weights_directory,
        dnsm,
        batch_size=32,
        learning_rate=0.001,
        checkpoint_dir="./_checkpoints",
        log_dir="./_logs",
    )

    burrito.train(2)
    burrito.optimize_branch_lengths()
