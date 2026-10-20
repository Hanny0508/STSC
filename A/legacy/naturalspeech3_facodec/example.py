import torch
import soundfile as sf
import librosa
from tts_model import TTSModel
from huggingface_hub import hf_hub_download
import numpy as np
from transformers import BertTokenizer
import os

# 设置CUDA启动阻塞
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_audio(wav_path):
    """加载音频文件并转换为模型所需的格式"""
    wav = librosa.load(wav_path, sr=16000)[0]
    wav = torch.from_numpy(wav).float()
    wav = wav.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    return wav

def load_model():
    # 创建模型
    model = TTSModel()
    
    # 加载预训练的FACodec权重
    encoder_v2_ckpt = hf_hub_download(
        repo_id="amphion/naturalspeech3_facodec", 
        filename="ns3_facodec_encoder_v2.bin"
    )
    decoder_v2_ckpt = hf_hub_download(
        repo_id="amphion/naturalspeech3_facodec", 
        filename="ns3_facodec_decoder_v2.bin"
    )
    
    model.fa_encoder.load_state_dict(torch.load(encoder_v2_ckpt))
    model.fa_decoder.load_state_dict(torch.load(decoder_v2_ckpt))
    
    return model

def text_to_speech(text, reference_audio_path, model, tokenizer, output_path="output.wav", device='cuda'):
    """
    将文本转换为语音，使用参考音频的音色
    
    Args:
        text: 输入文本
        reference_audio_path: 参考音频文件路径
        model: TTS模型
        tokenizer: 文本分词器
        output_path: 输出音频路径
        device: 运行设备
    """
    # 将模型移到指定设备
    model = model.to(device)
    model.eval()
    
    # 加载参考音频
    reference_audio = load_audio(reference_audio_path).to(device)
    
    # 文本预处理
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100,  # 限制文本长度，避免生成过长的音频
        add_special_tokens=True
    )
    text_ids = inputs['input_ids'].to(device)
    
    # 生成语音
    with torch.no_grad():
        try:
            # 生成语音
            output = model(text_ids, reference_audio)
            audio = output['generated_audio']
            
            # 确保音频维度正确
            if len(audio.shape) == 2:
                audio = audio.unsqueeze(0)
            
            # 将音频转换为numpy数组并保存
            audio_np = audio[0][0].cpu().numpy()
            
            # 归一化音频
            audio_np = audio_np / np.max(np.abs(audio_np))
            
            # 保存音频
            sf.write(output_path, audio_np, 16000)  # 采样率为16kHz
            print(f"音频已保存到: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"生成过程中出现错误: {str(e)}")
            raise e

def main():
    try:
        # 检查CUDA是否可用
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"当前设备: {torch.cuda.get_device_name(0)}")
        
        # 加载模型和分词器
        print("正在加载模型...")
        model = load_model()
        print("正在加载分词器...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # 示例文本和参考音频
        text = "this is a test code for the natural speech 3 model"
        reference_audio_path = "./audio/2.wav"  # 请确保这个路径存在
        
        # 生成语音
        print("正在生成语音...")
        output_path = text_to_speech(text, reference_audio_path, model, tokenizer)
        print(f"生成的音频已保存到: {output_path}")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
