# 1. 导入数据
# 加载需要使用的库
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# 统一控制台输出编码，避免Windows命令行无法输出R²等字符
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 若KNN运行时出现距离计算或线程库错误，取消下面两行的注释
# from sklearn import set_config
# set_config(enable_cython_pairwise_dist=False)

# 设置输入文件和输出文件夹路径
# 程序和两个输入文件放在同一文件夹，运行时不需要指定工作路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 在程序所在文件夹内建立output文件夹，用于保存结果表和图形
# exist_ok=True表示文件夹已存在时继续使用，不删除其中已有文件
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)
print("输出文件夹：", output_dir)

# 设置中文字体和矢量图文字格式
plt.rcParams.update({
    "font.sans-serif": [
        "Microsoft YaHei", "SimHei", "DejaVu Sans"
    ],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42
})

# 读取两个输入文件
k_data = pd.read_excel(
    os.path.join(script_dir, "k_model_data.xlsx")
)
q_data = pd.read_excel(
    os.path.join(script_dir, "methane_comparison.xlsx")
)

# 核对工作表、列名、记录数和缺失值
for file_name, data in [
    ("k_model_data.xlsx", k_data),
    ("methane_comparison.xlsx", q_data)
]:
    print(f"\n{file_name}")
    print("数据维度：", data.shape)
    print("列名：", data.columns.tolist())
    print("记录数：", len(data))
    print("缺失值数量：")
    print(data.isna().sum())


# 2. 划分数据并训练模型
# 按80%∶20%划分训练集和测试集
# train_df不剔除异常值，train_clean仅剔除训练集异常值
train_df, test_df = train_test_split(
    k_data,
    test_size=0.20,
    random_state=29
)

# 仅使用训练集计算异常值界限
cols = ["precip_cm", "kinverse"]
q20 = train_df[cols].quantile(0.20)
q80 = train_df[cols].quantile(0.80)
iqr_20_80 = q80 - q20
lower = q20 - 1.5 * iqr_20_80
upper = q80 + 1.5 * iqr_20_80

# 判断训练集异常值
is_outlier = (
    (train_df[cols] < lower) |
    (train_df[cols] > upper)
).any(axis=1)

# 得到剔除异常值后的训练集
train_clean = train_df.loc[~is_outlier].copy()
print("原始训练集记录数：", len(train_df))
print("剔除的训练集异常记录数：", is_outlier.sum())
print("处理后训练集记录数：", len(train_clean))
print("测试集记录数：", len(test_df))


# 3. 模型调参与模型训练
# 设置三个模型及其参数搜索范围
# GBM：n_estimators为回归树数量，max_depth为树深，learning_rate为学习率
model_settings = {
    "GBM": (
        GradientBoostingRegressor(random_state=29),
        {
            "n_estimators": [10, 30, 50, 100, 150, 200, 300],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2]
        }
    ),
    # StandardScaler完成数据标准化，SVR使用标准化数据训练
    # C为惩罚系数，epsilon为不敏感区间宽度，gamma控制单个样本的影响范围
    "SVR": (
        Pipeline([
            ("scale", StandardScaler()),
            ("model", SVR(kernel="rbf"))
        ]),
        {
            "model__C": [0.1, 1, 5, 10, 20, 50, 100],
            "model__epsilon": [
                0.005, 0.01, 0.05, 0.1, 0.2, 0.5
            ],
            "model__gamma": [
                "scale", "auto", 0.1, 0.01, 0.001
            ]
        }
    ),
    # KNN：n_neighbors为邻近样本数量K，weights为权重，algorithm为搜索算法，p为距离度量参数
    "KNN": (
        KNeighborsRegressor(),
        {
            "n_neighbors": [
                3, 5, 7, 8, 9, 10, 11, 12, 13, 15
            ],
            "weights": ["uniform", "distance"],
            "algorithm": [
                "auto", "ball_tree", "kd_tree", "brute"
            ],
            "p": [1, 2]
        }
    )
}

# 定义训练和评价函数
def train_models(train_data, treatment):
    # 提取输入特征：年平均降水量
    X_train = train_data[["precip_cm"]]
    # 提取预测目标：反演得到的k值
    y_train = train_data["kinverse"]
    # 提取独立测试集的输入特征和预测目标
    X_test = test_df[["precip_cm"]]
    y_test = test_df["kinverse"]
    # 用于保存各模型的评价结果
    rows = []
    # 依次读取模型名称、模型对象及参数搜索范围
    for model_name, (model, parameters) in model_settings.items():
        # cv=5表示采用5折交叉验证，scoring="r2"表示使用交叉验证平均R²选择最优参数
        search = GridSearchCV(
            model,
            parameters,
            cv=5,
            scoring="r2"
        )

        # 在训练集上进行参数搜索和模型训练
        search.fit(X_train, y_train)

        # 取得交叉验证确定的最优模型
        best_model = search.best_estimator_

        # 使用最优模型分别预测训练集和测试集
        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)

        # SVR参数名称带有model__前缀，删除该前缀，使输出的参数名称更简洁
        best_params = {
            key.replace("model__", ""): value
            for key, value in search.best_params_.items()
        }

        # 保存异常值处理方式、最优参数和评价指标
        rows.append({
            "异常值处理": treatment,
            "模型": model_name,
            "最优参数": str(best_params),
            "CV R²": search.best_score_,
            "训练集 R²": r2_score(y_train, train_pred),
            "测试集 R²": r2_score(y_test, test_pred),
            "测试集 MSE": mean_squared_error(y_test, test_pred)
        })

    # 返回GBM、SVR和KNN三个模型的结果
    return rows

# 分别训练“不剔除异常值”和“仅剔除训练集异常值”两组模型
# 建立空列表，用于汇总两种异常值处理方案下的模型结果
model_rows = []

# 使用未剔除异常值的训练集进行模型调参与评价
# 返回GBM、SVR和KNN三个模型的结果，并添加到model_rows
model_rows += train_models(
    train_df,
    "不剔除"
)

# 使用仅剔除训练集异常值的数据进行模型调参与评价
# 测试集保持不变，结果继续添加到model_rows
model_rows += train_models(
    train_clean,
    "仅剔除训练集异常值"
)


# 4. 结果评价
# 4.1 k值预测模型结果
model_results = pd.DataFrame(model_rows)

# 调整行顺序，使同一模型的两种异常值处理方案相邻，便于填写表1并进行图形比较
model_order = {"GBM": 0, "SVR": 1, "KNN": 2}
treatment_order = {
    "不剔除": 0,
    "仅剔除训练集异常值": 1
}
model_results["_model_order"] = model_results["模型"].map(model_order)
model_results["_treatment_order"] = model_results["异常值处理"].map(
    treatment_order
)
model_results = (
    model_results
    .sort_values(["_model_order", "_treatment_order"])
    .drop(columns=["_model_order", "_treatment_order"])
    .reset_index(drop=True)
)

print("\nk值预测模型结果")
print(model_results.to_string(index=False))

# 产生表1：保存6组模型的最优参数和评价指标
table1_path = os.path.join(
    output_dir,
    "table1_k_model_results.xlsx"
)
model_results.to_excel(table1_path, index=False)
print("表1已保存：", table1_path)

# 4.2 甲烷产生量计算结果
# 两种MAPE必须使用完全相同的有效记录。
# 有效记录要求三个Q字段均可转换为数值且Qestimate不为0。
q_columns = ["Qestimate", "Qinventory", "Qpredicted"]
q_numeric = q_data[q_columns].apply(pd.to_numeric, errors="coerce")
valid = q_numeric.notna().all(axis=1) & q_numeric["Qestimate"].ne(0)
q_valid = q_data.loc[valid].copy()
q_valid[q_columns] = q_numeric.loc[valid]

# 计算每条有效记录的绝对百分比误差APE
q_valid["Inventory_APE"] = (
    (q_valid["Qestimate"] - q_valid["Qinventory"]).abs()
    / q_valid["Qestimate"].abs()
    * 100
)
q_valid["Predicted_APE"] = (
    (q_valid["Qestimate"] - q_valid["Qpredicted"]).abs()
    / q_valid["Qestimate"].abs()
    * 100
)

# MAPE是全部有效记录APE的算术平均值
inventory_mape = q_valid["Inventory_APE"].mean()
predicted_mape = q_valid["Predicted_APE"].mean()

# 计算Qpredicted相对于Qinventory的MAPE降低率
# 结果大于0表示Qpredicted的总体MAPE较低，
# 结果小于0表示Qpredicted的总体MAPE较高
error_reduction = (
    (inventory_mape - predicted_mape)
    / inventory_mape
    * 100
)
# 输出有关的结果
print("有效记录数：", len(q_valid))
print("Qinventory对应结果MAPE：", inventory_mape)
print("Qpredicted对应结果MAPE：", predicted_mape)
print("相对Inventory的MAPE降低率：", error_reduction)

# Qpredicted由methane_comparison.xlsx直接提供，
# 表示机器学习优化参数对应的既有甲烷产生量结果，
# 不由本程序训练得到的kpredicted重新计算。
if error_reduction > 0:
    predicted_summary = "MAPE低于Qinventory"
elif error_reduction < 0:
    predicted_summary = "MAPE高于Qinventory"
else:
    predicted_summary = "MAPE与Qinventory相同"

# 产生表2：汇总两种甲烷产生量结果的MAPE和MAPE降低率
methane_results = pd.DataFrame({
    "甲烷产生量": ["Qinventory", "Qpredicted"],
    "有效记录数": [len(q_valid), len(q_valid)],
    "MAPE（%）": [inventory_mape, predicted_mape],
    "相对Inventory的MAPE降低率（%）": [np.nan, error_reduction],
    "总体误差表现": ["基准", predicted_summary]
})

# 表2工作簿包含MAPE汇总和逐记录APE两个工作表，
# 便于追溯总体指标及后续绘制误差分布图
table2_path = os.path.join(
    output_dir,
    "table2_methane_evaluation.xlsx"
)
with pd.ExcelWriter(table2_path) as writer:
    methane_results.to_excel(
        writer,
        sheet_name="MAPE汇总",
        index=False
    )
    q_valid.to_excel(
        writer,
        sheet_name="逐记录APE",
        index=False
    )
print("表2已保存：", table2_path)


# 5. 结果可视化
# 5.1 可视化表1：三种R²和测试集MSE
x = np.arange(len(model_results))
x_labels = [
    f"{row['模型']}\n{row['异常值处理']}"
    for _, row in model_results.iterrows()
]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 左图：比较交叉验证、训练集和测试集R²
r2_columns = ["CV R²", "训练集 R²", "测试集 R²"]
r2_colors = ["#9AA0A6", "#4C78A8", "#F2A65A"]
bar_width = 0.24
for i, (column, color) in enumerate(zip(r2_columns, r2_colors)):
    bars = axes[0].bar(
        x + (i - 1) * bar_width,
        model_results[column],
        width=bar_width,
        label=column,
        color=color,
        edgecolor="white",
        linewidth=0.6
    )
    axes[0].bar_label(
        bars,
        labels=[f"{value:.3f}" for value in model_results[column]],
        padding=2,
        fontsize=7,
        rotation=0
    )
axes[0].axhline(0, color="#666666", linewidth=0.8)
axes[0].set_title("（a）k值预测模型的R²比较")
axes[0].set_ylabel("R²")
axes[0].set_xticks(x)
axes[0].set_xticklabels(x_labels, fontsize=8)
axes[0].legend(frameon=False, ncol=3, fontsize=8)
axes[0].grid(axis="y", alpha=0.2)
axes[0].margins(y=0.18)

# 右图：比较测试集MSE，数值越小表示平均平方偏差越小
mse_bars = axes[1].bar(
    x,
    model_results["测试集 MSE"],
    color="#6BA292",
    edgecolor="white",
    linewidth=0.6
)
axes[1].bar_label(
    mse_bars,
    labels=[
        f"{value:.2e}"
        for value in model_results["测试集 MSE"]
    ],
    padding=3,
    fontsize=8,
    rotation=0
)
axes[1].set_title("（b）k值预测模型的测试集MSE")
axes[1].set_ylabel("测试集MSE（yr$^{-2}$）")
axes[1].set_xticks(x)
axes[1].set_xticklabels(x_labels, fontsize=8)
axes[1].grid(axis="y", alpha=0.2)
axes[1].margins(y=0.18)
fig.suptitle("图1  k值预测模型评价结果", fontsize=13)
fig.tight_layout()

# 保存图1：PNG，SVG和PDF三种格式
figure1_base = os.path.join(
    output_dir,
    "figure1_k_model_comparison"
)
fig.savefig(figure1_base + ".png", dpi=600, bbox_inches="tight")
fig.savefig(figure1_base + ".svg", bbox_inches="tight")
fig.savefig(figure1_base + ".pdf", bbox_inches="tight")
plt.show()
plt.close(fig)

# 5.2 可视化表2：总体MAPE和逐记录APE分布
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

# 左图：比较两种甲烷产生量结果的总体MAPE
mape_values = [inventory_mape, predicted_mape]
mape_colors = ["#9AA0A6", "#4C78A8"]
mape_bars = axes[0].bar(
    ["Qinventory", "Qpredicted"],
    mape_values,
    color=mape_colors,
    width=0.58,
    edgecolor="white"
)

# 标注两个柱形对应的MAPE
axes[0].bar_label(
    mape_bars,
    labels=[f"{value:.2f}%" for value in mape_values],
    padding=4,
    fontsize=9
)

# 增加图形顶部留白
axes[0].set_ylim(0, max(mape_values) * 1.18)

# 将MAPE降低率放在右上角的独立注释框中
axes[0].text(
    0.97,
    0.91,
    f"相对Inventory的MAPE降低率：{error_reduction:.2f}%",
    ha="right",
    va="top",
    transform=axes[0].transAxes,
    fontsize=9,
    bbox={
        "boxstyle": "round,pad=0.35",
        "facecolor": "white",
        "edgecolor": "#B8B8B8",
        "linewidth": 0.8,
        "alpha": 0.95
    }
)
axes[0].set_title("（a）总体MAPE比较", pad=10)
axes[0].set_ylabel("MAPE（%）")
axes[0].grid(axis="y", alpha=0.2)

# 右图：使用箱线图和全部记录的散点展示APE分布
ape_data = [
    q_valid["Inventory_APE"].to_numpy(),
    q_valid["Predicted_APE"].to_numpy()
]
box = axes[1].boxplot(
    ape_data,
    tick_labels=["Qinventory", "Qpredicted"],
    patch_artist=True,
    showmeans=True,
    widths=0.48,
    medianprops={"color": "#222222", "linewidth": 1.4},
    meanprops={
        "marker": "D",
        "markerfacecolor": "white",
        "markeredgecolor": "#222222",
        "markersize": 5
    }
)

for patch, color in zip(box["boxes"], mape_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.72)

# 使用确定性的横向偏移展示全部记录，不改变任何数据值
for position, values, color in zip([1, 2], ape_data, mape_colors):
    row_number = np.arange(len(values))
    jitter = position + 0.045 * np.sin(row_number * 2.399)
    axes[1].scatter(
        jitter,
        values,
        s=9,
        alpha=0.20,
        color=color,
        edgecolors="none",
        rasterized=True
    )

axes[1].set_title(f"（b）逐记录APE分布（n = {len(q_valid)}）")
axes[1].set_ylabel("绝对百分比误差APE（%）")
axes[1].set_ylim(bottom=0)
axes[1].grid(axis="y", alpha=0.2)
fig.suptitle("图2  甲烷产生量误差比较", fontsize=13)
fig.tight_layout()

# 保存图2：PNG，SVG和PDF三种格式
figure2_base = os.path.join(
    output_dir,
    "figure2_methane_error_comparison"
)
fig.savefig(figure2_base + ".png", dpi=600, bbox_inches="tight")
fig.savefig(figure2_base + ".svg", bbox_inches="tight")
fig.savefig(figure2_base + ".pdf", bbox_inches="tight")
plt.show()
plt.close(fig)

print("\n结果表和图形已保存至：", output_dir)
