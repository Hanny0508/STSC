import torch
import torch.nn as nn
from text_encoder import TextEncoder
from ns3_codec.facodec import FACodecEncoderV2, FACodecDecoderV2

class TTSModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 文本编码器
        self.text_encoder = TextEncoder(
            vocab_size=21128,  # BERT中文词表大小
            d_model=256,
            nhead=4,
            num_layers=4,
            dim_feedforward=1024,
            dropout=0.1
        )
        
        # FACodec编码器和解码器
        self.fa_encoder = FACodecEncoderV2(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256
        )
        
        self.fa_decoder = FACodecDecoderV2(
            in_channels=256,
            upsample_initial_channel=1024,
            ngf=32,
            up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2,
            vq_num_q_p=1,
            vq_num_q_r=3,
            vq_dim=256,
            codebook_dim=8,
            codebook_size_prosody=10,
            codebook_size_content=10,
            codebook_size_residual=10,
            use_gr_x_timbre=True,
            use_gr_residual_f0=True,
            use_gr_residual_phone=True,
        )
        
        # 添加一个投影层来调整文本特征的时间维度
        self.text_proj = nn.Linear(256, 256)
        
    def forward(self, text_ids, reference_audio, text_mask=None):
        # 1. 文本编码
        text_features = self.text_encoder(text_ids, text_mask)  # [batch_size, seq_len, 256]
        
        # 2. 获取参考音频的特征
        with torch.no_grad():
            # 获取参考音频的编码和说话人嵌入
            enc_out_ref = self.fa_encoder(reference_audio)  # [batch_size, 256, T]
            prosody_ref = self.fa_encoder.get_prosody_feature(reference_audio)
            vq_post_emb_ref, vq_id_ref, _, quantized_ref, spk_embs_ref = self.fa_decoder(
                enc_out_ref, prosody_ref, eval_vq=False, vq=True
            )
            
            # 获取参考音频的时间长度
            ref_len = enc_out_ref.size(-1)
            
            # 调整文本特征的时间维度以匹配参考音频
            text_features = text_features.transpose(1, 2)  # [batch_size, 256, seq_len]
            text_features = nn.functional.interpolate(
                text_features,
                size=ref_len,
                mode='linear',
                align_corners=False
            )
            
            # 使用文本特征作为内容，参考音频的说话人嵌入作为音色
            vq_post_emb, vq_id, _, quantized, spk_embs = self.fa_decoder(
                text_features,  # 使用调整后的文本特征作为内容
                prosody_feature=prosody_ref,  # 使用参考音频的韵律特征
                eval_vq=False,
                vq=True,
                speaker_embedding=spk_embs_ref  # 使用参考音频的说话人嵌入
            )
            
            # 使用内容编码和参考音频的说话人嵌入生成音频
            vq_post_emb = self.fa_decoder.vq2emb(vq_id, use_residual=False)
            generated_audio = self.fa_decoder.inference(vq_post_emb, spk_embs_ref)
            
            return {
                'generated_audio': generated_audio,
                'text_features': text_features,
                'prosody_features': prosody_ref,
                'vq_post_emb': vq_post_emb,
                'vq_id': vq_id,
                'quantized': quantized,
                'spk_embs': spk_embs_ref
            }