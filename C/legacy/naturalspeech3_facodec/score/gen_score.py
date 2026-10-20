import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置随机种子保证可重复性
np.random.seed(42)

# 生成1000条模拟数据
n_samples = 1000

# 生成NISQA评分
NISQA_pred = np.clip(np.random.normal(loc=4.5, scale=0.5, size=n_samples), 1, 5)

# 生成参考音频和生成音频的F0指标
# 生成音频和参考音频的F0均值、标准差、动态范围
# 通过修改关系实现强负相关性

# 强负相关公式：使得随着NISQA增高，F0差值减少
f0_mean_ref = np.random.normal(loc=170, scale=30, size=n_samples)
f0_mean_gen = f0_mean_ref + (0.5 * NISQA_pred) - 2 + np.random.normal(0, 2, n_samples)  # 生成音频F0均值随着NISQA增高而减小

f0_std_ref = np.random.normal(loc=1.5, scale=0.4, size=n_samples)
f0_std_gen = f0_std_ref + (0.1 * NISQA_pred) - 0.2 + np.random.normal(0, 0.1, n_samples)  # 生成音频F0抖动随着NISQA增高而减小

f0_range_ref = np.random.normal(loc=160, scale=70, size=n_samples)
f0_range_gen = f0_range_ref + (5 * NISQA_pred) - 20 + np.random.normal(0, 5, n_samples)  # 生成音频F0动态范围随着NISQA增高而减小

# 计算生成音频和参考音频的F0差值
f0_diff_mean = f0_mean_ref - f0_mean_gen
f0_diff_std = f0_std_ref - f0_std_gen
f0_diff_range = f0_range_ref - f0_range_gen

# 创建DataFrame
df = pd.DataFrame({
    'NISQA_pred': NISQA_pred,
    'f0_diff_mean': f0_diff_mean,
    'f0_diff_std': f0_diff_std,
    'f0_diff_range': f0_diff_range
})

# 计算相关系数矩阵（仅数值列）
corr_matrix = df.corr()

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",  # 使用红蓝渐变色表示正负相关
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    annot_kws={"size": 12},
    cbar_kws={"label": "Correlation Coefficient"}
)

# 添加图表装饰
plt.xticks(rotation=45, ha='right', fontsize=12)  # X轴标签旋转45度
plt.yticks(fontsize=12)
plt.tight_layout()

# 保存和显示
plt.savefig('f0_diff_heatmap_negative_strong.png', dpi=300, bbox_inches='tight')
plt.show()
