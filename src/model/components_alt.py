from typing import Optional, Tuple
import torch
from torch import nn, Tensor
from torchvision.ops.misc import Conv2dNormActivation
import torch.nn.functional as F
from positional_encodings import torch_encodings as pos
from src.model.vim import VisionMamba
__all__ = [
    "ImgCnnBackbone",
    "ImgLinearBackbone",
    "ImgConvStemBackbone",
    "PositionEmbedding",
    "Encoder",
    "Decoder",
    "TokenEmbedding",
    "MultiHeadAttention",
]


class ImgCnnBackbone(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        output_channels: int,
        d_model: int,
        drop_layer: Tuple = None,
    ) -> None:
        super().__init__()

        # drop layers for classification & maxpooling for higher feature resolution
        layers = list(backbone.children())
        nlayer = len(layers)
        keep_layer = set([i for i in range(nlayer)]) - set(drop_layer)
        backbone = [layers[i] for i in keep_layer]
        self.backbone = nn.Sequential(*backbone)
        self.proj = nn.Linear(output_channels, d_model)
        self.channels = output_channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = x.flatten(start_dim=-2).transpose(1, 2)
        assert x.shape[-1] == self.channels, "Image channels size mismatch."
        x = self.proj(x)
        return x


class ImgLinearBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_size: int,
        in_chan: int = 3,
    ) -> None:
        super().__init__()

        self.conv_proj = nn.Conv2d(
            in_chan, out_channels=d_model, kernel_size=patch_size, stride=patch_size
        )
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_proj(x)
        x = x.flatten(start_dim=-2).transpose(1, 2)
        return x


class ImgConvStemBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        downsample_factor: int,
        output_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()

        assert downsample_factor % 2 == 0
        assert output_channels % (downsample_factor // 2) == 0
        input_channels = output_channels // (downsample_factor // 2)

        layers = [
            Conv2dNormActivation(
                3, input_channels, kernel_size=kernel_size, stride=2, padding=1
            )
        ]

        while input_channels != output_channels:
            layers.append(
                Conv2dNormActivation(
                    input_channels,
                    input_channels * 2,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                )
            )
            input_channels = input_channels * 2

        layers.append(nn.Conv2d(output_channels, d_model, kernel_size=1))

        self.conv_stem = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_stem(x)
        x = x.flatten(start_dim=-2).transpose(1, 2)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, batch_first=True):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim 必须是 num_heads 的整数倍"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first 
        
        # Query, Key, Value 线性变换
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, attn_mask=None, key_padding_mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 1, float('-1e8'))
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_length)
            scores = scores.masked_fill(key_padding_mask, float('-1e8'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        
        return attn_output, attn_weights
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        if not self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        
        batch_size, seq_length, _ = query.size()
        
        # 线性变换 + 分头
        Q = self.q_proj(query).contiguous().view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).contiguous().view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).contiguous().view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, attn_mask, key_padding_mask)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        
        # 线性变换
        output = self.out_proj(attn_output)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        return output, attn_weights
       
class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_first: bool,
    ) -> None:
        super().__init__()

        self.norm_first = norm_first

        # 自注意力层
        self.self_attn = MultiHeadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=True
        )

        # 跨注意力层
        self.cross_attn = MultiHeadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=True
        )

        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            getattr(nn, activation.upper())(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # LayerNorm 和 Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: Tensor, 
        memory: Tensor, 
        tgt_mask: Tensor = None, 
        tgt_key_padding_mask: Tensor = None, 
        memory_mask: Tensor = None, 
        memory_key_padding_mask: Tensor = None
    ) -> Tensor:
        # 自注意力层
        if self.norm_first:
            x = self.norm1(x)
            attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
            x = x + self.dropout(attn_output)  # 残差连接
        else:
            attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
            x = x + self.dropout(attn_output)  # 残差连接
            x = self.norm1(x)
        
        # 跨注意力层
        if self.norm_first:
            x = self.norm2(x)
            cross_attn_output, _ = self.cross_attn(x, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
            x = x + self.dropout(cross_attn_output)  # 残差连接
        else:
            cross_attn_output, _ = self.cross_attn(x, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
            x = x + self.dropout(cross_attn_output)  # 残差连接
            x = self.norm2(x)

        # 前馈层
        if self.norm_first:
            x = self.norm3(x)
            ff_output = self.feed_forward(x)
            x = x + self.dropout(ff_output)  # 残差连接
        else:
            ff_output = self.feed_forward(x)
            x = x + self.dropout(ff_output)  # 残差连接
            x = self.norm3(x)

        return x

class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        nlayer: int,
        ff_ratio: int = 4,
    ) -> None:
        super().__init__()

        # 原来代码
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead=nhead,
            dim_feedforward=ff_ratio * d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)


    #     self.encoder = VisionMamba(
    #         embed_dim=d_model,
    #         depth=nlayer,
    #         rms_norm=True,
    #         residual_in_fp32=True,
    #         fused_add_norm=True,
    #         if_abs_pos_embed=True,
    #         if_rope=False,
    #         if_rope_residual=False,
    #         bimamba_type="V2",
    #         if_cls_token=True,
    #         if_divide_out=True,
    #         use_double_cls_token=True
    #     )

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        return x



class VTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        ff_ratio: int = 4,
    ) -> None:
        super().__init__()

        # 创建每一层的 self-attention 和 feedforward 网络
        # self.layers = nn.ModuleList(
        #     [
        #         DecoderLayer(
        #             d_model=d_model,
        #             nhead=nhead,
        #             dim_feedforward=ff_ratio * d_model,
        #             dropout=dropout,
        #             activation=activation,
        #             norm_first=norm_first,
        #         )
        #         for _ in range(nlayer)
        #     ]
        # )

        # 创建一层一层的 self-attention 和 feedforward 网络
        self.Decoder_Layer = DecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_ratio * d_model,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

        # 创建一层一层的 vim 网络
        self.vim = VisionMamba(
            embed_dim=d_model,
            depth=1,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            if_abs_pos_embed=True,
            if_rope=False,
            if_rope_residual=False,
            bimamba_type="V2",
            if_cls_token=True,
            if_divide_out=True,
            use_double_cls_token=True
        )


    def forward(
        self, x: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor
    ) -> Tensor:
        
        x = self.vim(x)
        x = self.Decoder_Layer(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        nlayer: int,
        ff_ratio: int = 4,
    ) -> None:
        super().__init__()

        # 创建每一层的 self-attention 和 feedforward 网络
        self.layers = nn.ModuleList(
            [
                VTBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                )
                for _ in range(nlayer)
            ]
        )

    def forward(
        self, x: Tensor, memory: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor
    ) -> Tensor:
        for layer in self.layers:  # 逐层执行
            x = layer(x, memory, tgt_mask, tgt_padding_mask)
        return x
    
class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # assume x is batch first
        out = self.embedding(torch.arange(x.shape[1], device=x.device))
        return self.dropout(out + x)

class TokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int,
    ) -> None:
        super().__init__()
        assert vocab_size > 0
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)


class PrintLayer(nn.Module):
    """Only for debugging when loss is nan."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(
            "torch.isfinite(x).all(): {}, min. {:.5f}, max. {:.5f}".format(
                torch.isfinite(x).all(), x.min(), x.max()
            )
        )
        return x


# if __name__ == "__main__":
#     from torchvision import models

#     x = torch.rand(1, 3, 392, 392)
#     model = ImgConvStemBackbone(
#         d_model=512, downsample_factor=16, output_channels=64, kernel_size=5
#     )
#     y = model(x)
#     print(model)
#     print(y.shape)

#     model = ImgCnnBackbone(
#         backbone=models.resnet34(),
#         output_channels=512,
#         d_model=512,
#         drop_layer=(3, 8, 9),
#     )

#     # print(model)
#     y = model(x)
#     print(y.shape)
