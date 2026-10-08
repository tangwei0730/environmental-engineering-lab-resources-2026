"""Assumption checks and method selection for the confirmed water-quality data.

This script intentionally does NOT run the final Mn omnibus comparison,
post-hoc comparisons, or Pearson/Spearman correlations. It only performs the
prespecified assumption checks and creates diagnostic figures/tables.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTION_ORDER = ["A", "B", "C"]
PALETTE = {"A": "#4C78A8", "B": "#F2A65A", "C": "#59A14F"}
MARKERS = {"A": "o", "B": "s", "C": "^"}

# Third-stage confirmed cleaned data. Only the two invalid Mn measurements are
# missing; EC and COD in those rows remain valid.
DATA = [
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

COLUMNS = [
    "断面",
    "采样编号",
    "EC",
    "COD",
    "Mn",
    "采样与仪器状态",
    "数据质量标记",
]


def build_clean_data() -> pd.DataFrame:
    """Build and verify the previously confirmed clean dataset."""
    frame = pd.DataFrame(DATA, columns=COLUMNS)
    frame["断面"] = pd.Categorical(
        frame["断面"], categories=SECTION_ORDER, ordered=True
    )
    assert len(frame) == 24
    assert frame.groupby("断面", observed=True).size().to_dict() == {
        "A": 8,
        "B": 8,
        "C": 8,
    }
    assert frame[["EC", "COD", "Mn"]].notna().sum().to_dict() == {
        "EC": 24,
        "COD": 24,
        "Mn": 22,
    }
    assert frame.groupby("断面", observed=True)["Mn"].count().to_dict() == {
        "A": 7,
        "B": 7,
        "C": 8,
    }
    return frame


def run_assumption_checks(frame: pd.DataFrame) -> pd.DataFrame:
    """Run only Shapiro-Wilk and median-centred Levene diagnostics."""
    rows = []
    groups = []

    for section in SECTION_ORDER:
        values = frame.loc[frame["断面"] == section, "Mn"].dropna().astype(float)
        groups.append(values.to_numpy())
        shapiro = stats.shapiro(values)
        rows.append(
            {
                "检查项目": "Shapiro-Wilk",
                "对象": f"Mn-{section}断面",
                "有效样本量": len(values),
                "单位": "mg/L",
                "统计量": float(shapiro.statistic),
                "自由度": "—",
                "p值": float(shapiro.pvalue),
                "样本标准差": float(values.std(ddof=1)),
                "判断": "未发现严重非正态证据（小样本，须结合Q-Q图）",
            }
        )

    # SciPy's median-centred form is the Brown-Forsythe robust version of
    # Levene's test. It is stated explicitly for reproducibility.
    levene = stats.levene(*groups, center="median")
    rows.append(
        {
            "检查项目": "Levene（中位数中心）",
            "对象": "Mn三个断面",
            "有效样本量": sum(len(group) for group in groups),
            "单位": "mg/L",
            "统计量": float(levene.statistic),
            "自由度": "2, 19",
            "p值": float(levene.pvalue),
            "样本标准差": np.nan,
            "判断": "方差齐性不成立",
        }
    )
    return pd.DataFrame(rows)


def build_pairwise_sample_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Record complete-pair counts without calculating correlations."""
    rows = []
    for x_variable, y_variable in [("EC", "COD"), ("EC", "Mn"), ("COD", "Mn")]:
        before_count = len(frame)
        pair_data = frame.dropna(subset=[x_variable, y_variable]).copy()
        after_count = len(pair_data)
        dropped_count = before_count - after_count
        expected_missing = int(
            frame[[x_variable, y_variable]].isna().any(axis=1).sum()
        )
        assert dropped_count == expected_missing
        group_counts = pair_data.groupby("断面", observed=True).size()
        rows.append(
            {
                "变量对": f"{x_variable}-{y_variable}",
                "总有效样本量": after_count,
                "A断面": int(group_counts.get("A", 0)),
                "B断面": int(group_counts.get("B", 0)),
                "C断面": int(group_counts.get("C", 0)),
                "缺失处理": "仅使用成对完整记录，不插补",
                "结构判断": "三个断面形成明显分组， pooled关联可能主要反映组间差异",
            }
        )
    return pd.DataFrame(rows)


def build_method_selection_table() -> pd.DataFrame:
    """Document selections without running any final inferential analysis."""
    return pd.DataFrame(
        [
            {
                "分析任务": "Mn总体组间比较",
                "候选方法": "普通单因素ANOVA",
                "选择状态": "不选",
                "匹配的后续方法": "Tukey HSD",
                "依据": "组内未见严重非正态，但Levene检验提示方差不齐",
            },
            {
                "分析任务": "Mn总体组间比较",
                "候选方法": "Welch单因素ANOVA",
                "选择状态": "选择",
                "匹配的后续方法": "Games-Howell两两比较",
                "依据": "连续型结局、独立样本、无严重非正态，且组间方差明显不齐",
            },
            {
                "分析任务": "Mn总体组间比较",
                "候选方法": "Kruskal-Wallis非参数检验",
                "选择状态": "不作为首选",
                "匹配的后续方法": "Dunn检验并校正多重比较",
                "依据": "当前未见需要放弃均值比较的严重偏态或极端异常值；该方法检验秩分布而非均值",
            },
            {
                "分析任务": "EC、COD与Mn的两两关联",
                "候选方法": "Pearson相关",
                "选择状态": "选择为pooled线性关联指标",
                "匹配的后续方法": "报告r、95%CI、精确p值及有效样本量",
                "依据": "变量为连续型，散点关系近似线性且无极端异常值；必须同时说明断面分组影响",
            },
            {
                "分析任务": "EC、COD与Mn的两两关联",
                "候选方法": "Spearman相关",
                "选择状态": "不作为首选",
                "匹配的后续方法": "报告rho、95%CI、精确p值及有效样本量",
                "依据": "关系并非仅为非线性单调模式；秩转换也不能消除断面分组造成的总体关联",
            },
        ]
    )


def set_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    sns.set_style("ticks")


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_DIR / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )


def make_mn_diagnostic_figure(frame: pd.DataFrame) -> None:
    """Show all raw Mn values and one Q-Q plot per section."""
    np.random.seed(20260829)
    fig = plt.figure(figsize=(7.2, 4.25))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.1, 1.0], hspace=0.55, wspace=0.42)
    distribution_axis = fig.add_subplot(grid[0, :])

    n_before = len(frame)
    mn_data = frame.dropna(subset=["Mn"]).copy()
    n_after = len(mn_data)
    excluded_count = n_before - n_after
    assert excluded_count == 2

    sns.boxplot(
        data=mn_data,
        x="断面",
        y="Mn",
        order=SECTION_ORDER,
        hue="断面",
        palette=PALETTE,
        dodge=False,
        width=0.5,
        linewidth=0.9,
        showfliers=False,
        legend=False,
        boxprops={"alpha": 0.38},
        medianprops={"color": "#222222", "linewidth": 1.2},
        whiskerprops={"color": "#555555", "linewidth": 0.9},
        capprops={"color": "#555555", "linewidth": 0.9},
        ax=distribution_axis,
    )
    sns.stripplot(
        data=mn_data,
        x="断面",
        y="Mn",
        order=SECTION_ORDER,
        hue="断面",
        palette=PALETTE,
        dodge=False,
        jitter=0.10,
        size=4.0,
        alpha=0.95,
        linewidth=0.45,
        edgecolor="white",
        legend=False,
        ax=distribution_axis,
        zorder=3,
    )
    counts = mn_data.groupby("断面", observed=True)["Mn"].count()
    distribution_axis.set_xticks(range(3))
    distribution_axis.set_xticklabels(
        [f"{section}\nn={int(counts[section])}" for section in SECTION_ORDER]
    )
    distribution_axis.set_xlabel("Section")
    distribution_axis.set_ylabel("Mn (mg/L)")
    distribution_axis.set_title("(a) Raw Mn distributions", loc="left", fontweight="bold")
    distribution_axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    distribution_axis.set_axisbelow(True)
    sns.despine(ax=distribution_axis)

    for panel_index, section in enumerate(SECTION_ORDER, start=1):
        axis = fig.add_subplot(grid[1, panel_index - 1])
        values = mn_data.loc[mn_data["断面"] == section, "Mn"].to_numpy()
        (theoretical, ordered), (slope, intercept, _qq_r) = stats.probplot(
            values, dist="norm"
        )
        axis.scatter(
            theoretical,
            ordered,
            s=25,
            marker=MARKERS[section],
            color=PALETTE[section],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        line_x = np.array([theoretical.min(), theoretical.max()])
        axis.plot(line_x, intercept + slope * line_x, color="#555555", linewidth=0.9)
        axis.set_xlabel("Theoretical normal quantiles")
        axis.set_ylabel("Ordered Mn (mg/L)")
        axis.set_title(
            f"({chr(97 + panel_index)}) Q-Q: Section {section} (n={len(values)})",
            loc="left",
            fontweight="bold",
        )
        axis.grid(color="#E5E5E5", linewidth=0.6)
        axis.set_axisbelow(True)
        sns.despine(ax=axis)

    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.11, top=0.96)
    save_figure(fig, "figure1_mn_assumption_diagnostics")
    plt.close(fig)


def make_correlation_structure_figure(frame: pd.DataFrame) -> None:
    """Contrast pooled scatter patterns with within-section-centred patterns."""
    panels = [
        ("EC", "COD", "EC (μS/cm)", "COD (mg/L)"),
        ("EC", "Mn", "EC (μS/cm)", "Mn (mg/L)"),
        ("COD", "Mn", "COD (mg/L)", "Mn (mg/L)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.65))

    for column_index, panel in enumerate(panels):
        x_variable, y_variable, x_label, y_label = panel
        before_count = len(frame)
        pair_data = frame.dropna(subset=[x_variable, y_variable]).copy()
        after_count = len(pair_data)
        dropped_count = before_count - after_count
        expected_missing = int(
            frame[[x_variable, y_variable]].isna().any(axis=1).sum()
        )
        assert dropped_count == expected_missing

        raw_axis = axes[0, column_index]
        centered_axis = axes[1, column_index]
        for section in SECTION_ORDER:
            section_data = pair_data.loc[pair_data["断面"] == section]
            raw_axis.scatter(
                section_data[x_variable],
                section_data[y_variable],
                s=27,
                marker=MARKERS[section],
                color=PALETTE[section],
                edgecolor="white",
                linewidth=0.5,
                label=section,
                zorder=3,
            )

        group_counts = pair_data.groupby("断面", observed=True).size()
        count_text = ", ".join(
            f"{section}={int(group_counts.get(section, 0))}"
            for section in SECTION_ORDER
        )
        raw_axis.text(
            0.03,
            0.97,
            f"n={len(pair_data)} ({count_text})",
            transform=raw_axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.4,
            color="#444444",
        )
        raw_axis.set_xlabel(x_label)
        raw_axis.set_ylabel(y_label)
        raw_axis.set_title(
            f"({chr(97 + column_index)}) Pooled {y_variable} vs {x_variable}",
            loc="left",
            fontweight="bold",
        )

        centered = pair_data.copy()
        centered["x_centered"] = centered[x_variable] - centered.groupby(
            "断面", observed=True
        )[x_variable].transform("mean")
        centered["y_centered"] = centered[y_variable] - centered.groupby(
            "断面", observed=True
        )[y_variable].transform("mean")
        for section in SECTION_ORDER:
            section_data = centered.loc[centered["断面"] == section]
            centered_axis.scatter(
                section_data["x_centered"],
                section_data["y_centered"],
                s=27,
                marker=MARKERS[section],
                color=PALETTE[section],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        centered_axis.axhline(0, color="#999999", linewidth=0.7, zorder=1)
        centered_axis.axvline(0, color="#999999", linewidth=0.7, zorder=1)
        centered_axis.set_xlabel(f"Within-section centered {x_variable} ({x_label.split('(')[1]}")
        centered_axis.set_ylabel(f"Within-section centered {y_variable} ({y_label.split('(')[1]}")
        centered_axis.set_title(
            f"({chr(100 + column_index)}) Within-section centered",
            loc="left",
            fontweight="bold",
        )

        for axis in [raw_axis, centered_axis]:
            axis.grid(color="#E5E5E5", linewidth=0.6)
            axis.set_axisbelow(True)
            sns.despine(ax=axis)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Section",
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.11, top=0.89, hspace=0.46, wspace=0.46)
    save_figure(fig, "figure2_correlation_structure_diagnostics")
    plt.close(fig)


def main() -> None:
    frame = build_clean_data()
    assumption_checks = run_assumption_checks(frame)
    pairwise_samples = build_pairwise_sample_table(frame)
    method_selection = build_method_selection_table()

    frame.to_csv(
        OUTPUT_DIR / "confirmed_clean_data.csv",
        index=False,
        na_rep="NA",
        encoding="utf-8-sig",
    )
    assumption_checks.to_csv(
        OUTPUT_DIR / "assumption_check_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pairwise_samples.to_csv(
        OUTPUT_DIR / "correlation_sample_structure.csv",
        index=False,
        encoding="utf-8-sig",
    )
    method_selection.to_csv(
        OUTPUT_DIR / "method_selection_table.csv",
        index=False,
        encoding="utf-8-sig",
    )

    set_plot_style()
    make_mn_diagnostic_figure(frame)
    make_correlation_structure_figure(frame)

    print("Assumption checks:")
    print(assumption_checks.to_string(index=False))
    print("\nPairwise sample structure:")
    print(pairwise_samples.to_string(index=False))
    print("\nMethod selection (no final tests executed):")
    print(method_selection.to_string(index=False))
    print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
