# 导入数据处理和绘图所需的库
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 设置输入和输出路径
# 输入文件与程序放在同一文件夹；主要结果统一保存到water_output文件夹
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "water_output")
os.makedirs(output_dir, exist_ok=True)

# 设置中文字体
plt.rcParams.update({
    "font.sans-serif": ["SimHei"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 10
})

# 1.读取数据
df = pd.read_excel("Water_Parameters.xlsx")

# 2.计算描述统计量
stats = df.agg(["mean", "std", "max", "min", "median"])
stats.index = ["平均值", "标准差", "最大值", "最小值", "中位值"]
print("描述统计结果：\n", stats.round(3))
# 将描述统计结果保存为Excel文件
stats.round(3).to_excel("描述统计结果.xlsx")

# 3.计算Pearson相关系数及p值
corr = df.corr()
n = len(df.columns)
p = pd.DataFrame(np.ones((n, n)), index=df.columns, columns=df.columns)
for i in range(n):
    for j in range(i):
        p.iloc[i, j] = p.iloc[j, i] = \
            pearsonr(df.iloc[:, i], df.iloc[:, j])[1]
print("\nPearson相关系数：\n", corr.round(3))
print("\np值矩阵：\n", p.round(3))
# 将Pearson相关系数矩阵和p值矩阵保存到water_output文件夹
corr.round(3).to_excel(
    os.path.join(output_dir, "Pearson相关系数矩阵.xlsx")
)
p.round(3).to_excel(
    os.path.join(output_dir, "p值矩阵.xlsx")
)

# 4.绘制热图
fig, ax = plt.subplots(figsize=(8, 7))
image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(n), df.columns, rotation=45, ha="right")
ax.set_yticks(range(n), df.columns)
for i in range(n):
    for j in range(n):
        star = "**" if p.iloc[i, j] < 0.01 \
            else "*" if p.iloc[i, j] < 0.05 else ""
        label = f"{corr.iloc[i, j]:.2f}" + \
            (f"$^{{{star}}}$" if star else "")
        color = "white" if abs(corr.iloc[i, j]) >= 0.5 else "black"
        ax.text(j, i, label, ha="center", va="center", color=color)
plt.colorbar(image, ax=ax, label="Pearson相关系数")
ax.set(title="水质参数Pearson相关性热图", xlabel="水质参数", ylabel="水质参数")
fig.text(0.5, 0.01, "注：*表示 p < 0.05，**表示 p < 0.01。", ha="center")
plt.tight_layout(rect=[0, 0.04, 1, 1])

# 将热图以PNG、SVG和PDF格式保存到water_output文件夹
figure_base = os.path.join(output_dir, "水质参数相关性热图")
fig.savefig(
    figure_base + ".png",
    dpi=300,
    bbox_inches="tight"
)
fig.savefig(figure_base + ".svg", bbox_inches="tight")
fig.savefig(figure_base + ".pdf", bbox_inches="tight")
plt.show()
print("\n主要结果和图已保存至：", output_dir)
