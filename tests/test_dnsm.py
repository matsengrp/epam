
# %%
import pandas as pd
import torch
from epam.dnsm import train_model, TransformerBinarySelectionModel
from epam.sequences import translate_sequences


# %%
pcp_df = pd.read_csv("~/data/wyatt-10x-1p5m_pcp_2023-10-07.first100.csv")

# %%
nhead = 4
dim_feedforward = 2048
layer_count = 3
model = train_model(pcp_df, nhead=nhead, dim_feedforward=dim_feedforward, layer_count=layer_count, batch_size=32, num_epochs=10, learning_rate=0.001, checkpoint_dir="./_checkpoints", log_dir="./_logs")

# # %%
# model = TransformerBinarySelectionModel(
#     nhead=nhead, dim_feedforward=dim_feedforward, layer_count=layer_count
# )
# 
# model.load_state_dict(torch.load("./_checkpoints/model_epoch_9.pth")["model_state_dict"])
# 
# # %%
# [aa_str] = translate_sequences([pcp_df.loc[0, "parent"]])
# model.p_substitution_of_aa_str(aa_str)
# 
# 
