import torch.nn as nn
from torch import Tensor, add
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, time_size, d_model):
        super().__init__()
        self.pe = nn.Embedding(time_size, d_model)

    def forward(self, time_ids):
        position = self.pe(time_ids)
        return position


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        feature_size: int,
        time_size: int = 24 * 7 * 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        nhead: int = 8,
    ):
        super(Seq2SeqTransformer, self).__init__()

        self.positional_encoding = PositionalEncoding(time_size=time_size, d_model=feature_size)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model=feature_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(
        self,
        src: Tensor,
        src_time: Tensor,
        tgt: Tensor,
        tgt_time: Tensor,
        mask_tgt: Tensor,
        mask_src: Tensor = None,
        padding_mask_src: Tensor = None,
        padding_mask_tgt: Tensor = None,
        memory_key_padding_mask: Tensor = None,
    ):
        src_time = self.positional_encoding(src_time)
        src = self.dropout(add(src, src_time))
        tgt_time = self.positional_encoding(tgt_time)
        tgt = self.dropout(add(tgt, tgt_time))
        memory = self.transformer_encoder(src, mask_src, padding_mask_src)
        outs = self.transformer_decoder(tgt, memory, mask_tgt, None, padding_mask_tgt, memory_key_padding_mask)
        return outs
