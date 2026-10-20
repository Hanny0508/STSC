import librosa
import numpy as np
import matplotlib.pyplot as plt

# 加载音频文件
audio_path = './audio/1.wav'
y, sr = librosa.load(audio_path, sr=16000)  # 建议16kHz采样率

# 使用YIN算法提取F0
f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                             fmin=librosa.note_to_hz('C2'),  # 最低频率（如男性语音约65Hz）
                                             fmax=librosa.note_to_hz('C7'))  # 最高频率（如女性语音约1000Hz）

# 可视化F0轨迹
times = librosa.times_like(f0, sr=sr)
plt.figure(figsize=(12, 4))
plt.plot(times, f0, label='F0', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('F0 Contour')
plt.ylim(50, 500)  # 根据实际调整范围
plt.show()

# 分析统计特征
f0_voiced = f0[voiced_flag]  # 仅保留有效基频点
mean_f0 = np.mean(f0_voiced)
std_f0 = np.std(f0_voiced)
dynamic_range = np.max(f0_voiced) - np.min(f0_voiced)

print(f"Mean F0: {mean_f0:.2f} Hz")
print(f"F0 Std: {std_f0:.2f} Hz")
print(f"Dynamic Range: {dynamic_range:.2f} Hz")