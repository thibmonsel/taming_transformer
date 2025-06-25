"""
Taken from https://github.com/thibmonsel/vqvae/blob/master/vqvae/quantizer.py
"""

import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.num_embeddings, 1.0 / self.num_embeddings
        )

    def forward(self, inputs):
        # inputs: (B, C, H, W), where C is embedding_dim
        # Reshape inputs from (B, C, H, W) to (B*H*W, C)
        inputs_permuted = inputs.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        inputs_flat = inputs_permuted.view(-1, self.embedding_dim)  # (B*H*W, C)

        # Compute distances between inputs and embeddings with L2 norm
        distances = torch.cdist(inputs_flat, self.embedding.weight, p=2) ** 2

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embedding(encoding_indices).squeeze(1)
        codebook_loss = F.mse_loss(quantized, inputs_flat.detach())
        commitment_loss = self.commitment_cost * F.mse_loss(
            inputs_flat, quantized.detach()
        )
        vq_loss = codebook_loss + commitment_loss

        # Straight-Through Estimator
        quantized_st = inputs_flat + (quantized - inputs_flat).detach()

        # Reshape quantized_st from (B*H*W, C) back to
        # the original input shape (B, C, H, W)
        quantized_out = quantized_st.view(inputs.shape)

        # Reshape encoding_indices for returning: (B, H, W)
        # inputs.shape[0] is B, inputs.shape[2] is H, inputs.shape[3] is W
        indices_reshaped = encoding_indices.reshape(
            inputs.shape[0], inputs.shape[2], inputs.shape[3]
        )

        # Computing perplexity
        encodings_one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = torch.mean(encodings_one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized_out, vq_loss, indices_reshaped, perplexity
