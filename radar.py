import numpy as np
import matplotlib.pyplot as plt
from math import pi

# 定义模型性能数据
models = ['Naïve Bayes', 'kNN(k = 7)', 'LOGREG', 'MLP', 'SVM', 'GBM', 'LGBM', 'XGBOOST', 'CATBOOST', 'AGABOOST', 'Random Forest', 'VOTING(hard)', 'VOTING(soft)', 'STACKING']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']

values = [
    [0.7975, 0.8678, 0.7971, 0.8308, 0.8783],
    [0.7989, 0.8271, 0.8574, 0.8418, 0.8650],
    [0.8187, 0.8399, 0.8771, 0.8579, 0.9071],
    [0.8086, 0.8349, 0.8646, 0.8492, 0.8802],
    [0.8176, 0.8361, 0.8807, 0.8577, 0.7968],
    [0.8156, 0.8427, 0.8668, 0.8544, 0.9017],
    [0.8232, 0.8479, 0.8739, 0.8605, 0.9046],
    [0.8215, 0.8471, 0.8717, 0.8590, 0.9042],
    [0.8192, 0.8441, 0.8717, 0.8576, 0.9049],
    [0.8176, 0.8381, 0.8775, 0.8572, 0.9021],
    [0.8165, 0.8418, 0.8704, 0.8556, 0.9001],
    [0.8209, 0.8418, 0.8784, 0.8596, 0.8020],
    [0.8232, 0.8416, 0.8833, 0.8618, 0.9086],
    [0.8229, 0.8412, 0.8833, 0.8616, 0.9091]
]

# 计算角度
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist() + [0]  # 确保角度闭合

# 添加第一个数据点到每个模型的值末尾以确保图形闭合
values = np.array(values)
values = np.concatenate([values, values[:, [0]]], axis=1)

# 设置图形
fig, ax = plt.subplots(figsize=(10, 8), dpi=100, subplot_kw=dict(polar=True))

# 绘制每个模型的表现
for i, model in enumerate(models):
    ax.plot(angles, values[i], linewidth=2, linestyle='solid', label=model)

# 设置坐标轴
ax.set_yticks([0.75, 0.8, 0.85, 0.9, 1.0])  # 从0.75开始
ax.set_yticklabels([f'{i:.2f}' for i in np.linspace(0.79, 0.91, 5)])  # 设置标签为0.75到1之间的数值

# 设置y轴的范围，使得雷达图的圆心是0.75
ax.set_ylim(0.79, 0.91)

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=12)

# 设置标题
ax.set_title("Model Performance Comparison", size=16, color='black', fontweight='bold')

# 添加图例
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
#
# # 显示图形
# plt.tight_layout()
# plt.show()

# 设置保存路径，例如保存为PNG文件
save_path = 'cv_results/model_performance_comparison.png'

# 保存图像
plt.savefig(save_path)
