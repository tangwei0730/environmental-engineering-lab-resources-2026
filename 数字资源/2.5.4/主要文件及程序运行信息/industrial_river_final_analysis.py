"""某工业受纳河流水质数据最终统计分析（不绘图）。"""

from __future__ import annotations

import platform
from itertools import combinations, permutations
from math import factorial

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import rankdata, studentized_range


ALPHA = 0.05
PAIRS = [("EC", "COD"), ("EC", "Mn"), ("COD", "Mn")]
UNITS = {"EC": "μS/cm", "COD": "mg/L", "Mn": "mg/L"}


def confirmed_clean_data() -> pd.DataFrame:
    """第三轮确认的清洗数据：Mn的-99.0和NA为缺失，不插补、不删整行。"""
    rows = [
        ("A", 1, 210, 12.4, 0.08, "正常", "未发现数据质量问题"),
        ("A", 2, 215, 11.8, 0.09, "正常", "未发现数据质量问题"),
        ("A", 3, 208, 13.1, 0.07, "正常", "未发现数据质量问题"),
        ("A", 4, 220, 12.0, np.nan, "仪器基线调零故障", "Mn缺失（原值-99.0；仪器故障）"),
        ("A", 5, 212, 12.6, 0.08, "正常", "未发现数据质量问题"),
        ("A", 6, 218, 11.5, 0.09, "正常", "未发现数据质量问题"),
        ("A", 7, 205, 12.8, 0.07, "正常", "未发现数据质量问题"),
        ("A", 8, 214, 12.2, 0.08, "正常", "未发现数据质量问题"),
        ("B", 1, 850, 48.5, 1.25, "正常", "未发现数据质量问题"),
        ("B", 2, 880, 52.1, 1.38, "正常", "未发现数据质量问题"),
        ("B", 3, 830, 45.0, 1.18, "正常", "未发现数据质量问题"),
        ("B", 4, 860, 50.2, np.nan, "样品消解管破损", "Mn缺失（原值NA；消解管破损）"),
        ("B", 5, 890, 55.4, 1.42, "正常", "未发现数据质量问题"),
        ("B", 6, 840, 47.8, 1.20, "正常", "未发现数据质量问题"),
        ("B", 7, 870, 51.0, 1.31, "正常", "未发现数据质量问题"),
        ("B", 8, 865, 49.6, 1.28, "正常", "未发现数据质量问题"),
        ("C", 1, 430, 24.5, 0.45, "正常", "未发现数据质量问题"),
        ("C", 2, 450, 26.8, 0.52, "正常", "未发现数据质量问题"),
        ("C", 3, 420, 22.0, 0.39, "正常", "未发现数据质量问题"),
        ("C", 4, 440, 25.1, 0.48, "正常", "未发现数据质量问题"),
        ("C", 5, 460, 28.0, 0.56, "正常", "未发现数据质量问题"),
        ("C", 6, 435, 23.9, 0.42, "正常", "未发现数据质量问题"),
        ("C", 7, 455, 27.2, 0.51, "正常", "未发现数据质量问题"),
        ("C", 8, 445, 25.5, 0.47, "正常", "未发现数据质量问题"),
    ]
    return pd.DataFrame(
        rows,
        columns=["断面", "采样编号", "EC", "COD", "Mn", "采样与仪器状态", "数据质量标记"],
    )


def validate_data(df: pd.DataFrame) -> None:
    assert len(df) == 24
    assert df.groupby("断面").size().to_dict() == {"A": 8, "B": 8, "C": 8}
    assert df[["EC", "COD"]].notna().all().all()
    missing_mn = set(
        map(tuple, df.loc[df["Mn"].isna(), ["断面", "采样编号"]].to_numpy())
    )
    assert missing_mn == {("A", 4), ("B", 4)}
    assert not df.duplicated().any()


def runtime_info() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "项目": ["Python", "NumPy", "pandas", "SciPy"],
            "版本": [platform.python_version(), np.__version__, pd.__version__, scipy.__version__],
        }
    )


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for section in ["A", "B", "C"]:
        section_df = df[df["断面"] == section]
        for indicator in ["EC", "COD", "Mn"]:
            values = section_df[indicator].dropna().to_numpy(dtype=float)
            n = len(values)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            t_critical = float(stats.t.ppf(1 - ALPHA / 2, df=n - 1))
            margin = t_critical * sd / np.sqrt(n)
            rows.append(
                {
                    "断面": section,
                    "指标": indicator,
                    "单位": UNITS[indicator],
                    "有效样本量n": n,
                    "均值": mean,
                    "样本标准差": sd,
                    "中位数": float(np.median(values)),
                    "最小值": float(np.min(values)),
                    "最大值": float(np.max(values)),
                    "均值95%CI下限": mean - margin,
                    "均值95%CI上限": mean + margin,
                }
            )
    return pd.DataFrame(rows)


def descriptive_eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """描述性eta平方：组间平方和/总平方和。"""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    grand_mean = values.mean()
    ss_total = np.sum((values - grand_mean) ** 2)
    ss_between = 0.0
    for group in np.unique(groups):
        group_values = values[groups == group]
        ss_between += len(group_values) * (group_values.mean() - grand_mean) ** 2
    return float(ss_between / ss_total)


def welch_anova_mn(df: pd.DataFrame) -> pd.DataFrame:
    analysis_df = df[["断面", "Mn"]].dropna()
    grouped = [
        group_df["Mn"].to_numpy(dtype=float)
        for _, group_df in analysis_df.groupby("断面", sort=True)
    ]
    k = len(grouped)
    n = np.asarray([len(x) for x in grouped], dtype=float)
    means = np.asarray([np.mean(x) for x in grouped], dtype=float)
    variances = np.asarray([np.var(x, ddof=1) for x in grouped], dtype=float)
    weights = n / variances
    weight_sum = weights.sum()
    weighted_mean = np.sum(weights * means) / weight_sum
    correction_sum = np.sum((1 - weights / weight_sum) ** 2 / (n - 1))

    df1 = float(k - 1)
    df2 = float((k**2 - 1) / (3 * correction_sum))
    numerator = np.sum(weights * (means - weighted_mean) ** 2) / df1
    denominator = 1 + (2 * (k - 2) / (k**2 - 1)) * correction_sum
    f_value = float(numerator / denominator)
    p_value = float(stats.f.sf(f_value, df1, df2))
    eta_squared = descriptive_eta_squared(
        analysis_df["Mn"].to_numpy(dtype=float),
        analysis_df["断面"].to_numpy(),
    )
    return pd.DataFrame(
        [{
            "指标": "Mn",
            "单位": "mg/L",
            "检验": "Welch单因素ANOVA",
            "总有效样本量n": len(analysis_df),
            "F": f_value,
            "df1": df1,
            "df2": df2,
            "p值": p_value,
            "效应量": "描述性η²",
            "效应量值": eta_squared,
        }]
    )


def games_howell_mn(df: pd.DataFrame) -> pd.DataFrame:
    grouped = {
        section: group_df["Mn"].dropna().to_numpy(dtype=float)
        for section, group_df in df.groupby("断面", sort=True)
    }
    k = len(grouped)
    rows = []
    for first, second in combinations(grouped, 2):
        x, y = grouped[first], grouped[second]
        n1, n2 = len(x), len(y)
        mean_difference = float(np.mean(x) - np.mean(y))
        var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
        se = float(np.sqrt(var1 / n1 + var2 / n2))
        df_welch = float(
            (var1 / n1 + var2 / n2) ** 2
            / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))
        )
        q_value = float(np.sqrt(2) * abs(mean_difference) / se)
        adjusted_p = float(studentized_range.sf(q_value, k, df_welch))
        q_critical = float(studentized_range.ppf(1 - ALPHA, k, df_welch))
        margin = q_critical * se / np.sqrt(2)
        rows.append(
            {
                "比较（前者-后者）": f"{first}-{second}",
                "单位": "mg/L",
                "n1": n1,
                "n2": n2,
                "均值差": mean_difference,
                "95%CI下限": mean_difference - margin,
                "95%CI上限": mean_difference + margin,
                "Games-Howell q": q_value,
                "自由度": df_welch,
                "校正p值": adjusted_p,
            }
        )
    return pd.DataFrame(rows)


def pooled_pearson_description(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for x_name, y_name in PAIRS:
        pair_df = df[[x_name, y_name]].dropna()
        r_value = float(np.corrcoef(pair_df[x_name], pair_df[y_name])[0, 1])
        rows.append(
            {
                "范围": "A、B、C合并",
                "指标组合": f"{x_name}-{y_name}",
                "有效配对样本量n": len(pair_df),
                "Pearson r": r_value,
                "p值": "不计算（预设为描述性分析）",
                "说明": "未经断面调整，不代表断面内关联",
            }
        )
    return pd.DataFrame(rows)


def spearman_exact(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """平均秩Spearman rho及穷举排列得到的双侧精确p值。"""
    x_rank = rankdata(np.asarray(x, dtype=float), method="average")
    y_rank = rankdata(np.asarray(y, dtype=float), method="average")
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = float(
        np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    if denominator == 0:
        raise ValueError("变量为常数，无法计算Spearman相关。")
    rho_observed = float(np.dot(x_centered, y_centered) / denominator)
    extreme = 0
    for permuted_index in permutations(range(len(y_centered))):
        rho_permuted = float(
            np.dot(x_centered, y_centered[list(permuted_index)]) / denominator
        )
        if abs(rho_permuted) >= abs(rho_observed) - 1e-12:
            extreme += 1
    permutation_count = factorial(len(y_centered))
    return rho_observed, extreme / permutation_count, permutation_count


def holm_adjust(p_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.minimum(
        np.maximum.accumulate((m - np.arange(m)) * sorted_p), 1.0
    )
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted <= ALPHA, adjusted


def within_section_spearman(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for section in ["A", "B", "C"]:
        section_df = df[df["断面"] == section]
        for x_name, y_name in PAIRS:
            pair_df = section_df[[x_name, y_name]].dropna()
            rho, exact_p, permutation_count = spearman_exact(
                pair_df[x_name].to_numpy(), pair_df[y_name].to_numpy()
            )
            rows.append(
                {
                    "断面": section,
                    "指标组合": f"{x_name}-{y_name}",
                    "有效配对样本量n": len(pair_df),
                    "Spearman ρ": rho,
                    "双侧精确置换p值": exact_p,
                    "穷举排列数": permutation_count,
                }
            )
    results = pd.DataFrame(rows)
    reject, adjusted = holm_adjust(results["双侧精确置换p值"].to_numpy())
    results["Holm校正p值（9项）"] = adjusted
    results["Holm校正后p<0.05"] = reject
    return results


def print_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n### {title}")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 240,
        "display.float_format", lambda value: f"{value:.12g}",
    ):
        print(table.to_string(index=False))


def main() -> None:
    df = confirmed_clean_data()
    validate_data(df)
    print_table("运行信息", runtime_info())
    print_table("描述统计", descriptive_statistics(df))
    print_table("Mn Welch ANOVA及效应量", welch_anova_mn(df))
    print_table("Mn Games-Howell事后比较", games_howell_mn(df))
    print_table("总体Pearson线性描述", pooled_pearson_description(df))
    print_table("分断面Spearman精确置换检验及Holm校正", within_section_spearman(df))


if __name__ == "__main__":
    main()
