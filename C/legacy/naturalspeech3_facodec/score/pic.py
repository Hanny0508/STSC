import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 读取CSV文件
try:
    df = pd.read_csv("./score/nisqa_f0_dataset4.csv")
except FileNotFoundError:
    print("错误：CSV文件未找到，请确认文件路径是否正确！")
    exit()

# 2. 计算相关系数矩阵（仅数值列）
corr_matrix = df[['NISQA_pred', 'f0_std', 'f0_range', 'f0_mean']].corr()

# 3. 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,          # 显示相关系数
    fmt=".2f",           # 数值格式保留两位小数
    cmap="coolwarm",     # 红蓝渐变色
    vmin=-1,             # 颜色范围固定为[-1,1]
    vmax=1,
    linewidths=0.5,      # 单元格边框线宽
    annot_kws={"size": 12},  # 标注文字大小
    cbar_kws={"label": "Correlation Coefficient"}  # 颜色条标签
)

# 4. 添加图表装饰
# plt.title("NISQA-MOS与F0指标相关性热力图", fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=12)  # X轴标签旋转45度
plt.yticks(fontsize=12)
plt.tight_layout()

# 5. 保存和显示
plt.savefig('score/correlation_heatmap4.png', dpi=300, bbox_inches='tight')
plt.show()