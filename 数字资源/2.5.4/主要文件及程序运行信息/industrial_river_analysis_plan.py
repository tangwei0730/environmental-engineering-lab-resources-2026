"""
某工业受纳河流水质数据：适用条件检查与预设统计分析代码

默认设置 RUN_FINAL_TESTS = False，因此不会执行最终的：
1. Mn Welch ANOVA；
2. Games-Howell 事后比较；
3. 合并数据 Pearson 描述；
4. 分断面 Spearman 秩相关、双侧精确置换检验和 Holm 校正。

将 RUN_FINAL_TESTS 改为 True 后，才会执行上述最终分析。
"""

from __future__ import annotations

from itertools import combinations, permutations
from math import factorial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import rankdata, studentized_range


# -------------------------
# 运行控制
# -------------------------
RUN_CONDITION_CHECKS = True
MAKE_DIAGNOSTIC_PLOTS = True
RUN_FINAL_TESTS = False  # 人工确认后改为 True

ALPHA = 0.05  # 常用预设；执行最终检验前须由人工确认
OUTPUT_DIR = Path("analysis_outputs")

INDICATOR_UNITS = {
    "EC": "μS/cm",
    "COD": "mg/L",
    "Mn": "mg/L",
}

PAIR_LIST = [
    ("EC", "COD"),
    ("EC", "Mn"),
    ("COD", "Mn"),
]


def build_confirmed_clean_data() -> pd.DataFrame:
    """建立第三阶段已确认的清洗数据；不插补，不整行删除。"""
    rows = [
        # 断面, 编号, EC, COD, Mn, 状态, 数据质量标记
        ("A", 1, 210, 12.4, 0.08, "正常", "有效"),
        ("A", 2, 215, 11.8, 0.09, "正常", "有效"),
        ("A", 3, 208, 13.1, 0.07, "正常", "有效"),
        ("A", 4, 220, 12.0, np.nan, "仪器基线调零故障", "Mn缺失：原始-99.0按缺失处理"),
        ("A", 5, 212, 12.6, 0.08, "正常", "有效"),
        ("A", 6, 218, 11.5, 0.09, "正常", "有效"),
        ("A", 7, 205, 12.8, 0.07, "正常", "有效"),
        ("A", 8, 214, 12.2, 0.08, "正常", "有效"),
        ("B", 1, 850, 48.5, 1.25, "正常", "有效"),
        ("B", 2, 880, 52.1, 1.38, "正常", "有效"),
        ("B", 3, 830, 45.0, 1.18, "正常", "有效"),
        ("B", 4, 860, 50.2, np.nan, "样品消解管破损", "Mn缺失：原始NA"),
        ("B", 5, 890, 55.4, 1.42, "正常", "有效"),
        ("B", 6, 840, 47.8, 1.20, "正常", "有效"),
        ("B", 7, 870, 51.0, 1.31, "正常", "有效"),
        ("B", 8, 865, 49.6, 1.28, "正常", "有效"),
        ("C", 1, 430, 24.5, 0.45, "正常", "有效"),
        ("C", 2, 450, 26.8, 0.52, "正常", "有效"),
        ("C", 3, 420, 22.0, 0.39, "正常", "有效"),
        ("C", 4, 440, 25.1, 0.48, "正常", "有效"),
        ("C", 5, 460, 28.0, 0.56, "正常", "有效"),
        ("C", 6, 435, 23.9, 0.42, "正常", "有效"),
        ("C", 7, 455, 27.2, 0.51, "正常", "有效"),
        ("C", 8, 445, 25.5, 0.47, "正常", "有效"),
    ]
    return pd.DataFrame(
        rows,
        columns=["断面", "采样编号", "EC", "COD", "Mn", "采样与仪器状态", "数据质量标记"],
    )


def validate_confirmed_data(df: pd.DataFrame) -> None:
    """防止代码运行时意外改变已确认的数据结构。"""
    assert len(df) == 24, "总记录数应为24。"
    assert df.groupby("断面").size().to_dict() == {"A": 8, "B": 8, "C": 8}
    assert df[["EC", "COD"]].notna().all().all(), "EC和COD不应存在缺失。"

    missing_mn = set(
        map(tuple, df.loc[df["Mn"].isna(), ["断面", "采样编号"]].to_numpy())
    )
    assert missing_mn == {("A", 4), ("B", 4)}, "Mn缺失位置与确认结果不一致。"
    assert not df.duplicated().any(), "确认后的数据不应含完全重复记录。"


def effective_sample_sizes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分别生成单指标和成对相关分析的有效样本量。"""
    indicator_n = (
        df.groupby("断面")[["EC", "COD", "Mn"]]
        .count()
        .reset_index()
        .melt(id_vars="断面", var_name="指标", value_name="有效样本量n")
    )
    indicator_n["单位"] = indicator_n["指标"].map(INDICATOR_UNITS)

    pair_rows = []
    for section, section_df in df.groupby("断面", sort=True):
        for x_name, y_name in PAIR_LIST:
            n_pair = int(section_df[[x_name, y_name]].dropna().shape[0])
            pair_rows.append(
                {
                    "断面": section,
                    "指标组合": f"{x_name}–{y_name}",
                    "有效配对样本量n": n_pair,
                    "单位": "相关系数无单位；变量单位见坐标轴",
                }
            )
    pair_n = pd.DataFrame(pair_rows)
    return indicator_n, pair_n


def run_condition_checks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """执行已规划的适用条件检查，不执行最终组间或相关检验。"""
    # Mn各断面Shapiro-Wilk检验
    shapiro_rows = []
    for section, section_df in df.groupby("断面", sort=True):
        values = section_df["Mn"].dropna().to_numpy(dtype=float)
        result = stats.shapiro(values)
        shapiro_rows.append(
            {
                "断面": section,
                "指标": "Mn",
                "单位": "mg/L",
                "有效样本量n": len(values),
                "Shapiro-Wilk W": result.statistic,
                "p值": result.pvalue,
            }
        )
    shapiro_mn = pd.DataFrame(shapiro_rows)

    # Brown-Forsythe形式：Levene检验以中位数为中心
    mn_groups = [
        group_df["Mn"].dropna().to_numpy(dtype=float)
        for _, group_df in df.groupby("断面", sort=True)
    ]
    levene_result = stats.levene(*mn_groups, center="median")
    levene_mn = pd.DataFrame(
        [
            {
                "指标": "Mn",
                "单位": "mg/L",
                "检验": "Levene（以中位数为中心/Brown-Forsythe）",
                "统计量": levene_result.statistic,
                "p值": levene_result.pvalue,
                "组数": 3,
                "总有效样本量n": int(df["Mn"].notna().sum()),
            }
        ]
    )

    # 合并数据的单变量Shapiro-Wilk检验仅用于描述分组混合特征，
    # 不作为是否可以计算Pearson相关系数的唯一判据。
    pooled_rows = []
    for indicator in ["EC", "COD", "Mn"]:
        values = df[indicator].dropna().to_numpy(dtype=float)
        result = stats.shapiro(values)
        pooled_rows.append(
            {
                "指标": indicator,
                "单位": INDICATOR_UNITS[indicator],
                "有效样本量n": len(values),
                "Shapiro-Wilk W": result.statistic,
                "p值": result.pvalue,
                "解释限制": "合并分布受A、B、C断面分层影响",
            }
        )
    pooled_shapiro = pd.DataFrame(pooled_rows)

    return {
        "Mn分断面Shapiro": shapiro_mn,
        "Mn方差齐性": levene_mn,
        "合并数据Shapiro": pooled_shapiro,
    }


def make_diagnostic_plots(df: pd.DataFrame) -> None:
    """绘制Mn分断面Q-Q图和分断面相关散点图，仅用于适用条件检查。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, (section, section_df) in zip(axes, df.groupby("断面", sort=True)):
        values = section_df["Mn"].dropna().to_numpy(dtype=float)
        stats.probplot(values, dist="norm", plot=axis)
        axis.set_title(f"{section}断面 Mn Q-Q图 (n={len(values)})")
        axis.set_xlabel("正态理论分位数")
        axis.set_ylabel("Mn样本分位数 (mg/L)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mn_qq_by_section.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for row_index, section in enumerate(["A", "B", "C"]):
        section_df = df[df["断面"] == section]
        for col_index, (x_name, y_name) in enumerate(PAIR_LIST):
            axis = axes[row_index, col_index]
            pair_df = section_df[[x_name, y_name]].dropna()
            sns.scatterplot(data=pair_df, x=x_name, y=y_name, ax=axis, s=55)
            axis.set_title(
                f"{section}断面：{x_name}–{y_name} (n={len(pair_df)})"
            )
            axis.set_xlabel(f"{x_name} ({INDICATOR_UNITS[x_name]})")
            axis.set_ylabel(f"{y_name} ({INDICATOR_UNITS[y_name]})")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "within_section_scatter_checks.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def welch_anova_mn(df: pd.DataFrame) -> pd.DataFrame:
    """Mn三独立断面的Welch单因素ANOVA。"""
    grouped_values = [
        group_df["Mn"].dropna().to_numpy(dtype=float)
        for _, group_df in df.groupby("断面", sort=True)
    ]
    group_count = len(grouped_values)
    sample_sizes = np.asarray([len(values) for values in grouped_values], dtype=float)
    means = np.asarray([np.mean(values) for values in grouped_values], dtype=float)
    variances = np.asarray(
        [np.var(values, ddof=1) for values in grouped_values], dtype=float
    )
    if np.any(variances <= 0):
        raise ValueError("Welch ANOVA要求各组样本方差大于0。")

    weights = sample_sizes / variances
    total_weight = weights.sum()
    weighted_mean = np.sum(weights * means) / total_weight
    weighted_between = np.sum(weights * (means - weighted_mean) ** 2)
    correction_sum = np.sum(
        (1.0 - weights / total_weight) ** 2 / (sample_sizes - 1.0)
    )

    df_num = float(group_count - 1)
    df_denom = float((group_count**2 - 1.0) / (3.0 * correction_sum))
    denominator_correction = 1.0 + (
        2.0 * (group_count - 2.0) / (group_count**2 - 1.0)
    ) * correction_sum
    f_statistic = (weighted_between / df_num) / denominator_correction
    p_value = stats.f.sf(f_statistic, df_num, df_denom)
    total_n = int(sample_sizes.sum())

    return pd.DataFrame(
        [
            {
                "指标": "Mn",
                "单位": "mg/L",
                "检验": "Welch单因素ANOVA",
                "统计量": f_statistic,
                "分子自由度": df_num,
                "分母自由度": df_denom,
                "p值": p_value,
                "总有效样本量n": total_n,
            }
        ]
    )


def games_howell_mn(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Mn的Games-Howell两两比较；适用于方差不齐和样本量不完全相等。"""
    grouped = {
        name: group_df["Mn"].dropna().to_numpy(dtype=float)
        for name, group_df in df.groupby("断面", sort=True)
    }
    k_groups = len(grouped)
    rows = []

    for group_1, group_2 in combinations(grouped.keys(), 2):
        values_1 = grouped[group_1]
        values_2 = grouped[group_2]
        n_1, n_2 = len(values_1), len(values_2)
        mean_1, mean_2 = np.mean(values_1), np.mean(values_2)
        var_1, var_2 = np.var(values_1, ddof=1), np.var(values_2, ddof=1)

        mean_diff = mean_1 - mean_2
        se_diff = np.sqrt(var_1 / n_1 + var_2 / n_2)
        welch_df = (var_1 / n_1 + var_2 / n_2) ** 2 / (
            (var_1 / n_1) ** 2 / (n_1 - 1)
            + (var_2 / n_2) ** 2 / (n_2 - 1)
        )
        q_stat = np.sqrt(2.0) * abs(mean_diff) / se_diff
        p_value = studentized_range.sf(q_stat, k_groups, welch_df)
        q_critical = studentized_range.ppf(1.0 - alpha, k_groups, welch_df)
        margin = q_critical * se_diff / np.sqrt(2.0)

        rows.append(
            {
                "比较": f"{group_1}–{group_2}",
                "指标": "Mn",
                "单位": "mg/L",
                "n1": n_1,
                "n2": n_2,
                "均值差": mean_diff,
                "Welch自由度": welch_df,
                "Games-Howell q": q_stat,
                "p值": p_value,
                "均值差95%CI下限": mean_diff - margin,
                "均值差95%CI上限": mean_diff + margin,
            }
        )
    return pd.DataFrame(rows)


def pooled_pearson_description(df: pd.DataFrame) -> pd.DataFrame:
    """合并数据Pearson r仅作总体线性描述，不计算或解释显著性p值。"""
    rows = []
    for x_name, y_name in PAIR_LIST:
        pair_df = df[[x_name, y_name]].dropna()
        r_value = np.corrcoef(pair_df[x_name], pair_df[y_name])[0, 1]
        rows.append(
            {
                "数据范围": "A、B、C合并",
                "指标组合": f"{x_name}–{y_name}",
                "有效配对样本量n": len(pair_df),
                "Pearson r": r_value,
                "单位": "无单位",
                "解释": "未经断面调整的总体线性描述，不代表断面内关联",
            }
        )
    return pd.DataFrame(rows)


def spearman_exact_permutation(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, int]:
    """
    Spearman rho及双侧精确置换p值。

    - 并列值使用平均秩；
    - 固定x，穷举y的全部n!个标签排列；
    - 双侧p = P(|rho_perm| >= |rho_obs|)；
    - 因为是穷举精确检验，不使用Monte Carlo的“+1”修正。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x与y必须为同长度的成对观测。")
    if len(x) < 3:
        raise ValueError("至少需要3个有效配对观测。")

    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = np.sqrt(
        np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)
    )
    if denominator == 0:
        raise ValueError("至少一个变量在该断面内为常数，无法计算Spearman相关。")

    observed_rho = float(np.dot(x_centered, y_centered) / denominator)
    extreme_count = 0
    total_permutations = factorial(len(y))
    tolerance = 1e-12

    # 按索引排列可正确保留并列观测的排列重数。
    for permutation_index in permutations(range(len(y))):
        permuted_y = y_centered[list(permutation_index)]
        permuted_rho = float(np.dot(x_centered, permuted_y) / denominator)
        if abs(permuted_rho) >= abs(observed_rho) - tolerance:
            extreme_count += 1

    exact_two_sided_p = extreme_count / total_permutations
    return observed_rho, exact_two_sided_p, total_permutations


def within_section_spearman_with_holm(df: pd.DataFrame) -> pd.DataFrame:
    """计算9项分断面Spearman精确检验，并把9项定义为一个Holm比较族。"""
    rows = []
    for section, section_df in df.groupby("断面", sort=True):
        for x_name, y_name in PAIR_LIST:
            pair_df = section_df[[x_name, y_name]].dropna()
            rho, exact_p, permutation_count = spearman_exact_permutation(
                pair_df[x_name].to_numpy(dtype=float),
                pair_df[y_name].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "断面": section,
                    "指标组合": f"{x_name}–{y_name}",
                    "有效配对样本量n": len(pair_df),
                    "Spearman rho": rho,
                    "双侧精确置换p值": exact_p,
                    "穷举排列数": permutation_count,
                    "单位": "无单位",
                }
            )

    result_df = pd.DataFrame(rows)
    reject, adjusted_p = holm_adjust(
        result_df["双侧精确置换p值"].to_numpy(), alpha=ALPHA
    )
    result_df["Holm校正p值（9项）"] = adjusted_p
    result_df["Holm校正后p<0.05"] = reject
    return result_df


def holm_adjust(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Holm逐步校正；返回拒绝标记和按原顺序排列的校正p值。"""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or len(p_values) == 0:
        raise ValueError("p_values必须是一维非空数组。")
    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p值必须位于0至1之间。")

    test_count = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    scale = test_count - np.arange(test_count)
    adjusted_sorted = np.maximum.accumulate(scale * sorted_p)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(test_count, dtype=float)
    adjusted[order] = adjusted_sorted
    reject = adjusted <= alpha
    return reject, adjusted


def run_final_analyses(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """仅在人工确认并将RUN_FINAL_TESTS设为True后调用。"""
    return {
        "Mn_Welch_ANOVA": welch_anova_mn(df),
        "Mn_Games_Howell": games_howell_mn(df, alpha=ALPHA),
        "总体_Pearson_描述": pooled_pearson_description(df),
        "分断面_Spearman_精确置换_Holm": within_section_spearman_with_holm(df),
    }


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = build_confirmed_clean_data()
    validate_confirmed_data(df)

    indicator_n, pair_n = effective_sample_sizes(df)
    print("\n单指标有效样本量：")
    print(indicator_n.to_string(index=False))
    print("\n分断面相关分析的有效配对样本量：")
    print(pair_n.to_string(index=False))

    if RUN_CONDITION_CHECKS:
        condition_tables = run_condition_checks(df)
        for table_name, table in condition_tables.items():
            print(f"\n{table_name}：")
            print(table.to_string(index=False))

    if MAKE_DIAGNOSTIC_PLOTS:
        make_diagnostic_plots(df)
        print(f"\n诊断图已保存至：{OUTPUT_DIR.resolve()}")

    if not RUN_FINAL_TESTS:
        print(
            "\n最终检验未执行。人工确认后，将RUN_FINAL_TESTS改为True再运行。"
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_tables = run_final_analyses(df)
    for table_name, table in final_tables.items():
        print(f"\n{table_name}：")
        print(table.to_string(index=False))
        table.to_csv(OUTPUT_DIR / f"{table_name}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
