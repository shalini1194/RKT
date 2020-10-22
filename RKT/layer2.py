import torch.nn as nn
from SubLayers import MultiHeadAttention, PositionwiseFeedForward


class MatrixFactorizationLayer(nn.Linear):
    ''' Modified linear layer without bias and reused weights '''

    def __init__(self, _weight):
        super().__init__(_weight.size(0), _weight.size(1), bias=False)
        self.weight = _weight


class SelfAttentionBlock(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.2):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input_query, enc_input_key, enc_input_value,
                timestamp_K,corr,l1,non_pad_mask=None, slf_attn_mask=None):
        residual = enc_input_query

        enc_input_query = self.layer_norm(enc_input_query)
        enc_input_key = self.layer_norm(enc_input_key)
        enc_input_value = self.layer_norm(enc_input_value)
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input_query, enc_input_key, enc_input_value,
            timestamp_K, corr,l1,mask=slf_attn_mask)
        enc_output = residual + self.dropout(enc_output)

        enc_output *= non_pad_mask

        residual = enc_output
        enc_output = self.layer_norm(enc_output)

        enc_output = self.pos_ffn(enc_output)
        enc_output = residual + self.dropout(enc_output)

        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn