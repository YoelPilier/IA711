# %%

import torch


imagenes = torch.rand((2, 3, 128, 128))

imagenes.shape

# %%
videos = torch.rand((3, 60, 3, 128, 128))
videos.shape


# %%
arreglo = [1, 2, 3, 4, 5, 6, 7, 8, 9]
i = 2
j = 2

arreglo[i * 3 + j * 1]


# %%
# reshape
videos.shape
v1_reshape = videos.reshape(3, 60, 3 * 128 * 128)
v1_reshape.shape

# %%
v2_view = videos.view(3, 60, -1)

# %%

imagenes[0:2, 2, :, :].view(2, 512 * 512).shape
# %%
