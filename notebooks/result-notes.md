## hydrophobic neighbor mutator simulation

```
nhead = 4
dim_feedforward = 2048
layer_count = 3

Without	the positional encoder:
Epoch [0/2], Training Loss: 5.478170156478882, Validation Loss: 5.341393775448075
Epoch [1/2], Training Loss: 0.22140918672084808, Validation Loss: 0.1811294092190983
Epoch [2/2], Training Loss: 0.22316762804985046, Validation Loss: 0.18113862033559805

With positional encoder:
Epoch [0/2], Training Loss: 6.603643817915294, Validation Loss: 6.601628321289675
Epoch [1/2], Training Loss: 0.17955662310123444, Validation Loss: 0.18135988180736415
Epoch [2/2], Training Loss: 0.12157808989286423, Validation Loss: 0.181174641160193


nhead = 2
dim_feedforward = 256
layer_count = 2

Epoch [0/2], Training Loss: 6.370150334865838, Validation Loss: 6.379060747971849
Epoch [1/2], Training Loss: 0.23558320105075836, Validation Loss: 0.18132765149233335
Epoch [2/2], Training Loss: 0.21081417798995972, Validation Loss: 0.18122180218255965

nhead = 1
dim_feedforward = 16
layer_count = 1

Without	the positional encoder:
Epoch [0/2], Training Loss: 1.877058620859937, Validation Loss: 1.8647038502816142
Epoch [1/2], Training Loss: 0.16272467374801636, Validation Loss: 0.18125704404788578
Epoch [2/2], Training Loss: 0.21040047705173492, Validation Loss: 0.18125464423508902

With positional encoder:
Epoch [0/2], Training Loss: 1.678296947838415, Validation Loss: 1.6787940547254503
Epoch [1/2], Training Loss: 0.1890346258878708, Validation Loss: 0.1817096910068504
Epoch [2/2], Training Loss: 0.21665123105049133, Validation Loss: 0.1816061288331846
```