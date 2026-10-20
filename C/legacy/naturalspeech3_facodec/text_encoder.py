import torch
import torch.nn as nn
from ns3_codec.transformer import TransformerEncoder

class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size=21128,         # BERT中文词表大小
        d_model=256,              # 模型维度
        nhead=4,                  # 注意力头数
        num_layers=4,             # Transformer层数
        dim_feedforward=1024,     # 前馈网络维度
        dropout=0.1,
        max_seq_len=5000
    ):
        super().__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Transformer编码器
        self.transformer = TransformerEncoder(
            encoder_layer=num_layers,
            encoder_hidden=d_model,
            encoder_head=nhead,
            conv_filter_size=dim_feedforward,
            conv_kernel_size=5,
            encoder_dropout=dropout,
            use_cln=False
        )
        
        # 输出投影层
        self.output_proj = nn.Linear(d_model, 256)  # 投影到FACodec的输入维度
        
    def forward(self, text_ids, text_mask=None):
        # text_ids: [batch_size, seq_len] 或 [1, batch_size, seq_len]
        
        # 打印输入形状和值范围
        print(f"输入形状: {text_ids.shape}")
        print(f"输入值范围: [{text_ids.min().item()}, {text_ids.max().item()}]")
        
        # 确保输入索引在有效范围内
        if text_ids.max() >= self.embedding.num_embeddings:
            print(f"警告：输入索引超出词表范围！词表大小: {self.embedding.num_embeddings}")
            text_ids = torch.clamp(text_ids, 0, self.embedding.num_embeddings - 1)
        
        # 处理输入维度
        if len(text_ids.shape) == 3:
            # 如果是 [1, batch_size, seq_len]，去掉第一个维度
            text_ids = text_ids.squeeze(0)
        
        # 词嵌入
        x = self.embedding(text_ids)  # [batch_size, seq_len, d_model]
        print(f"词嵌入后形状: {x.shape}")
        
        # 添加位置编码
        seq_len = x.size(1)
        pos_enc = self.pos_encoder[:, :seq_len, :]
        print(f"位置编码形状: {pos_enc.shape}")
        x = x + pos_enc
        
        # Transformer编码
        x = self.transformer(x, text_mask)
        print(f"Transformer后形状: {x.shape}")
        
        # 投影到FACodec输入维度
        x = self.output_proj(x)  # [batch_size, seq_len, 256]
        print(f"最终输出形状: {x.shape}")
        
        return x
