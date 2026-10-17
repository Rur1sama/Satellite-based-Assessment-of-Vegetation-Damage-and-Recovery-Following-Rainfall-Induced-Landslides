from __future__ import annotations

import base64
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\python_base\Slide_dect")
DID_DIR = ROOT / "Output" / "EcoService_DID"
DID_RESULTS = DID_DIR / "04_did_results"
AUDIT_DIR = DID_DIR / "06_snap_lai_qc_audit"
OUT_DIR = DID_DIR / "07_change_tracking_storyboard"
FIG_DIR = OUT_DIR / "figures"
SNAP_SINGLE = ROOT / "Output" / "SNAP_Results" / "Biophysical_SingleBand"

OUTCOME_LABELS = {
    "LAI_SR2": "LAI",
    "FVC": "FVC",
    "FAPAR": "FAPAR",
}
ZONE_LABELS = {
    "core_inner": "滑坡核心内侧",
    "core": "滑坡核心",
    "ring_0_10m": "外圈 0-10 m",
    "ring_10_20m": "外圈 10-20 m",
}
ZONE_ORDER = ["core_inner", "core", "ring_0_10m", "ring_10_20m"]
OUTCOME_ORDER = ["LAI_SR2", "FVC", "FAPAR"]
COLORS = {
    "core_inner": "#8b1e3f",
    "core": "#d24d3e",
    "ring_0_10m": "#e39b32",
    "ring_10_20m": "#2f7f9f",
}


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 220


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def fig_workflow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_axis_off()
    boxes = [
        ("完整保留时间序列", "LAI / FVC / FAPAR\n不先按 LAI>0.8 删除", 0.03),
        ("提取滑坡前后变化", "以前一年为基线\n看突降、低谷、恢复", 0.28),
        ("用 SCL/QC 做解释标签", "SCL=5、QC非零\n提示不确定性，不是先删", 0.53),
        ("输出跟踪信号", "核心区损失\n外圈影响\n恢复年限", 0.78),
    ]
    for title, body, x in boxes:
        ax.text(
            x + 0.09,
            0.5,
            f"{title}\n\n{body}",
            ha="center",
            va="center",
            fontsize=11.5,
            color="#26323f",
            linespacing=1.45,
        )
    for x in [0.22, 0.47, 0.72]:
        ax.plot([x - 0.02, x + 0.04], [0.5, 0.5], color="#5b6b7a", lw=1.8)
    ax.text(0.5, 0.88, "变化跟踪版：不要把异常值先剪掉，先让时间序列说话", ha="center", fontsize=17, weight="bold")
    ax.text(0.5, 0.12, "适合滑坡识别：重点看同一位置从滑坡前到滑坡后的变化形状，而不是单期 LAI 的绝对值。", ha="center", fontsize=12, color="#5b6570")
    savefig(path)


def fig_event_study(path: Path, event: pd.DataFrame) -> None:
    df = event[event["zone_name"].isin(["core", "ring_0_10m", "ring_10_20m"])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for ax, outcome in zip(axes, OUTCOME_ORDER):
        sub = df[df["outcome"] == outcome]
        for zone in ["core", "ring_0_10m", "ring_10_20m"]:
            z = sub[sub["zone_name"] == zone].sort_values("relative_year")
            if z.empty:
                continue
            ax.plot(z["relative_year"], z["att"], marker="o", lw=2.2, color=COLORS[zone], label=ZONE_LABELS[zone])
            ax.fill_between(z["relative_year"], z["ci_low"], z["ci_high"], color=COLORS[zone], alpha=0.13, linewidth=0)
        ax.axhline(0, color="#222", lw=1)
        ax.axvline(0, color="#555", lw=1, ls="--")
        ax.set_title(f"{OUTCOME_LABELS[outcome]}：滑坡后相对变化", fontsize=13, weight="bold")
        ax.set_xlabel("相对滑坡年份")
        ax.grid(True, alpha=0.25)
        if outcome == "LAI_SR2":
            ax.set_ylabel("DID变化量（负值=下降）")
    axes[-1].legend(loc="best", frameon=False, fontsize=9)
    fig.suptitle("图1  滑坡前后变化曲线：核心区突降，外圈仍有可见影响", fontsize=16, weight="bold", y=1.02)
    savefig(path)


def fig_focus_2024(path: Path, focus: pd.DataFrame) -> None:
    df = focus[focus["zone_name"].isin(["core", "ring_0_10m", "ring_10_20m"])].copy()
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, outcome in zip(axes, OUTCOME_ORDER):
        sub = df[df["outcome"] == outcome]
        for zone in ["core", "ring_0_10m", "ring_10_20m"]:
            z = sub[sub["zone_name"] == zone].sort_values("relative_year")
            if z.empty:
                continue
            ax.plot(z["relative_year"], z["treated_change"], marker="o", color=COLORS[zone], lw=2.0, label=f"{ZONE_LABELS[zone]}：滑坡区")
            ax.plot(z["relative_year"], z["control_change"], marker="x", color=COLORS[zone], lw=1.4, ls="--", alpha=0.65, label=f"{ZONE_LABELS[zone]}：对照")
        ax.axhline(0, color="#222", lw=1)
        ax.axvline(0, color="#555", lw=1, ls="--")
        ax.set_ylabel(OUTCOME_LABELS[outcome])
        ax.set_title(f"{OUTCOME_LABELS[outcome]}：2024 年滑坡组相对 2023 的真实变化 vs 对照变化", fontsize=12, weight="bold")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("相对 2024 滑坡年份")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:6], labels[:6], loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("图2  2024 年滑坡组跟踪：不是只看绝对值，而是看滑坡区是否比对照下降更多", fontsize=16, weight="bold", y=0.995)
    plt.subplots_adjust(bottom=0.15)
    savefig(path)


def fig_damage_recovery(path: Path, recovery: pd.DataFrame) -> None:
    df = recovery[recovery["zone_name"].isin(ZONE_ORDER)].copy()
    df["zone_label"] = df["zone_name"].map(ZONE_LABELS)
    df["outcome_label"] = df["outcome"].map(OUTCOME_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    pivot = df.pivot_table(index="zone_label", columns="outcome_label", values="max_damage_att", aggfunc="first")
    pivot = pivot.reindex([ZONE_LABELS[z] for z in ZONE_ORDER if ZONE_LABELS[z] in pivot.index])
    pivot.plot(kind="barh", ax=axes[0], color=["#9c2445", "#e08a2e", "#2f7f9f"])
    axes[0].axvline(0, color="#222", lw=1)
    axes[0].set_title("最大损失强度（越负，下降越明显）", fontsize=12, weight="bold")
    axes[0].set_xlabel("最大DID下降量")
    axes[0].set_ylabel("")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend(frameon=False)

    rec = df.pivot_table(index="zone_label", columns="outcome_label", values="time_to_80pct_recovery", aggfunc="first")
    rec = rec.reindex(pivot.index)
    im = axes[1].imshow(rec.values.astype(float), cmap="YlGnBu", vmin=0, vmax=np.nanmax(rec.values.astype(float)))
    axes[1].set_xticks(range(len(rec.columns)), rec.columns)
    axes[1].set_yticks(range(len(rec.index)), rec.index)
    axes[1].set_title("恢复到 80% 损失回补所需年数", fontsize=12, weight="bold")
    for i in range(rec.shape[0]):
        for j in range(rec.shape[1]):
            val = rec.iloc[i, j]
            txt = "" if pd.isna(val) else f"{val:.0f}年"
            axes[1].text(j, i, txt, ha="center", va="center", color="#17202a", fontsize=11, weight="bold")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle("图3  影响强度与恢复速度：核心区最强，外圈较弱但仍可跟踪", fontsize=16, weight="bold", y=1.03)
    savefig(path)


def fig_scene_backbone(path: Path, variable_stats: pd.DataFrame) -> None:
    df = variable_stats[variable_stats["tile"] == "T49RGH"].copy()
    df["year"] = df["date"].astype(str).str[:4].astype(int)
    df["label"] = df["variable"].replace({"LAI": "LAI", "FAPAR": "FAPAR", "FVC": "FVC"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, var in zip(axes, ["LAI", "FVC", "FAPAR"]):
        sub = df[df["variable"] == var].sort_values("year")
        ax.plot(sub["year"], sub["mean"], marker="o", lw=2.2, color="#2f7f9f")
        ax.fill_between(sub["year"], sub["p25"], sub["p75"], color="#2f7f9f", alpha=0.18, label="P25-P75")
        ax.set_title(f"{var} 年度影像总体均值", fontsize=12, weight="bold")
        ax.set_xlabel("年份")
        ax.grid(True, alpha=0.25)
        if var == "LAI":
            ax.set_ylabel("整景统计值")
    fig.suptitle("图4  数据骨架：8 景 T49RGH 产品按年排好，用来做变化跟踪", fontsize=16, weight="bold", y=1.02)
    savefig(path)


def fig_scl_qc_diagnostic(path: Path, cross: pd.DataFrame) -> None:
    df = cross[cross["scl_code"] == 5].copy()
    df["year"] = df["date"].astype(str).str[:4].astype(int)
    df = df.sort_values("year")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(df["year"], df["qc0_lai_gt_0_8_pct_of_all"] * 100, color="#d24d3e", label="QC=0 且 LAI>0.8")
    ax.bar(
        df["year"],
        (df["qc_nonzero_lai_gt_0_8"] / df["pixels"]) * 100,
        bottom=df["qc0_lai_gt_0_8_pct_of_all"] * 100,
        color="#6b7280",
        label="QC非零 且 LAI>0.8",
    )
    ax.set_ylabel("占 SCL=5 像元比例 (%)")
    ax.set_xlabel("年份")
    ax.set_title("图5  裸地高 LAI 是诊断标签，不是删除规则", fontsize=15, weight="bold")
    ax.text(
        0.02,
        0.93,
        "用法：保留这些像元进入时间序列；\n后面用 SCL/QC 解释变化可信度。",
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        color="#334155",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    savefig(path)


def fig_drop_classes(path: Path, drop: pd.DataFrame) -> None:
    df = drop[(drop["zone_name"] == "core") & (drop["outcome"].isin(OUTCOME_ORDER))].copy()
    valid_classes = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    df = df[df["drop_class"].isin(valid_classes)]
    grouped = df.groupby(["outcome", "drop_class"], as_index=False)["n_landslides"].sum()
    pivot = grouped.pivot_table(index="drop_class", columns="outcome", values="n_landslides", fill_value=0)
    pivot = pivot.reindex(valid_classes)
    pivot = pivot.rename(columns=OUTCOME_LABELS)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    pivot.plot(kind="bar", stacked=False, ax=ax, color=["#9c2445", "#e08a2e", "#2f7f9f"])
    ax.set_title("图6  单个滑坡区的下降幅度分组：用于识别“明显突降”的对象", fontsize=15, weight="bold")
    ax.set_xlabel("事件年相对滑坡前的下降比例")
    ax.set_ylabel("滑坡数量")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    savefig(path)


def image_tag(path: Path, title: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"<section><h2>{title}</h2><img src=\"data:image/png;base64,{data}\" /></section>"


def build_key_numbers(recovery: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOME_ORDER:
        for zone in ["core", "ring_0_10m", "ring_10_20m"]:
            row = recovery[(recovery["outcome"] == outcome) & (recovery["zone_name"] == zone)]
            if row.empty:
                continue
            r = row.iloc[0]
            rows.append(
                {
                    "指标": OUTCOME_LABELS[outcome],
                    "区域": ZONE_LABELS[zone],
                    "最大下降": r["max_damage_att"],
                    "最大下降发生相对年": r["max_damage_relative_year"],
                    "恢复到80%所需年数": r["time_to_80pct_recovery"],
                }
            )
    scl5 = cross[cross["scl_code"] == 5]
    if not scl5.empty:
        rows.append(
            {
                "指标": "裸地高LAI诊断",
                "区域": "SCL=5",
                "最大下降": None,
                "最大下降发生相对年": None,
                "恢复到80%所需年数": None,
                "说明": f"SCL=5 且 LAI>0.8 占 {scl5['lai_gt_0_8'].sum() / scl5['pixels'].sum():.2%}；不要先删，作为解释标签。",
            }
        )
    return pd.DataFrame(rows)


def write_readme(path: Path, key_numbers: pd.DataFrame) -> None:
    lines = [
        "# 滑坡变化跟踪版结果",
        "",
        "这版不是 QC 清洗报告，而是给滑坡识别/跟踪看的图文结果。",
        "",
        "## 怎么看",
        "",
        "1. 先看 `index.html`，它把 6 张图按阅读顺序排好了。",
        "2. 图1 看总体规律：滑坡发生年附近是否出现突降，外圈是否也下降。",
        "3. 图2 看 2024 这一批滑坡：滑坡区相对 2023 的变化是否比对照更差。",
        "4. 图3 看哪个指标最敏感、哪个区域恢复慢。",
        "5. 图5 是提醒：裸地高 LAI 不要直接删除，它只是变化可信度的解释标签。",
        "",
        "## 关键判断",
        "",
        "- 单期 LAI 高不等于不是滑坡；新鲜裸土、阴影、湿土、混合像元都可能让 LAI 反演偏。",
        "- 做滑坡识别时要看同一位置的时间变化：滑坡前高、事件年突降、随后缓慢恢复，这个形状比单期阈值更重要。",
        "- SCL/QC 保留为标签：当变化很强但 QC 非零时，标为“需要人工复核”；当变化强且 QC 稳定时，可信度更高。",
        "",
        "## 文件",
        "",
        "- `index.html`：中文图文总览。",
        "- `figures/`：6 张 PNG 图。",
        "- `tracking_key_numbers.csv`：最少量关键数字。",
        "",
        "## T49RFH",
        "",
        "T49RFH 是多下载的瓦片，已从 `Output\\SNAP_Results` 派生产品中删除；本版只看 T49RGH 的 2018-2025 序列。",
        "",
    ]
    if not key_numbers.empty:
        lines.append("## 关键数字预览")
        lines.append("")
        for _, row in key_numbers.head(12).iterrows():
            if pd.notna(row.get("最大下降")):
                lines.append(
                    f"- {row['指标']} / {row['区域']}：最大下降 {row['最大下降']:.3f}，恢复到80%约 {row['恢复到80%所需年数']} 年。"
                )
            elif row.get("说明"):
                lines.append(f"- {row['说明']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(path: Path, figure_paths: list[tuple[Path, str]], key_numbers: pd.DataFrame) -> None:
    cards = []
    for _, row in key_numbers.head(9).iterrows():
        if pd.notna(row.get("最大下降")):
            cards.append(
                f"<div class='card'><b>{row['指标']} / {row['区域']}</b><span>最大下降 {row['最大下降']:.3f}</span><span>80%恢复：{row['恢复到80%所需年数']} 年</span></div>"
            )
    figures = "\n".join(image_tag(p, title) for p, title in figure_paths)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>滑坡变化跟踪版结果</title>
<style>
body {{ margin: 0; font-family: "Microsoft YaHei", "Segoe UI", sans-serif; color: #17202a; background: #f3f5f7; }}
header {{ padding: 28px 42px; background: #1f2933; color: white; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
p {{ line-height: 1.7; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 26px; }}
.note {{ background: white; border-left: 5px solid #d24d3e; padding: 18px 20px; margin-bottom: 20px; }}
.cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 18px 0 26px; }}
.card {{ background: white; padding: 14px 16px; border: 1px solid #d8dee6; border-radius: 8px; display: flex; flex-direction: column; gap: 7px; }}
section {{ background: white; padding: 20px; margin: 18px 0; border: 1px solid #d8dee6; border-radius: 8px; }}
h2 {{ margin-top: 0; font-size: 20px; }}
img {{ width: 100%; height: auto; display: block; }}
footer {{ padding: 18px 42px 32px; color: #59636e; }}
</style>
</head>
<body>
<header>
<h1>滑坡变化跟踪版结果</h1>
<p>目标：保留 LAI/FVC/FAPAR 时间序列，看突降、影响范围和恢复过程；SCL/QC 只作为解释标签，不先删除。</p>
</header>
<main>
<div class="note">
<b>读图原则：</b> 单期 LAI 偏高不等于无效。对滑坡识别，更重要的是同一区域在滑坡前后是否出现“突降 - 低谷 - 恢复”的形状。
</div>
<div class="cards">
{''.join(cards)}
</div>
{figures}
</main>
<footer>输出目录：{OUT_DIR}</footer>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    setup_plot_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    event = read_csv(DID_RESULTS / "event_study_att.csv")
    focus = read_csv(DID_RESULTS / "focus_2024_event_study_att.csv")
    recovery = read_csv(DID_RESULTS / "recovery_metrics.csv")
    drop = read_csv(DID_RESULTS / "individual_zone_drop_class_summary.csv")
    variable_stats = read_csv(AUDIT_DIR / "biophysical_variable_stats.csv")
    cross = read_csv(AUDIT_DIR / "lai_scl_qc_crosscheck.csv")

    # Keep only the tile that remains in the SNAP products.
    variable_stats = variable_stats[variable_stats["tile"] == "T49RGH"].copy()

    figure_paths = [
        (FIG_DIR / "fig01_event_study_tracking.png", "图1 滑坡前后变化曲线"),
        (FIG_DIR / "fig02_focus_2024_treated_vs_control.png", "图2 2024滑坡组 vs 对照"),
        (FIG_DIR / "fig03_damage_and_recovery.png", "图3 影响强度与恢复速度"),
        (FIG_DIR / "fig04_scene_backbone.png", "图4 年度产品数据骨架"),
        (FIG_DIR / "fig05_scl5_high_lai_diagnostic.png", "图5 裸地高LAI诊断"),
        (FIG_DIR / "fig06_drop_class_counts.png", "图6 明显突降对象分组"),
    ]

    fig_event_study(figure_paths[0][0], event)
    fig_focus_2024(figure_paths[1][0], focus)
    fig_damage_recovery(figure_paths[2][0], recovery)
    fig_scene_backbone(figure_paths[3][0], variable_stats)
    fig_scl_qc_diagnostic(figure_paths[4][0], cross)
    fig_drop_classes(figure_paths[5][0], drop)

    key_numbers = build_key_numbers(recovery, cross)
    key_numbers.to_csv(OUT_DIR / "tracking_key_numbers.csv", index=False, encoding="utf-8-sig")
    write_readme(OUT_DIR / "README_变化跟踪版.md", key_numbers)
    write_html(OUT_DIR / "index.html", figure_paths, key_numbers)

    product_counts = []
    for product_dir in sorted(SNAP_SINGLE.glob("*")):
        if product_dir.is_dir():
            product_counts.append({"product": product_dir.name, "tif_count": len(list(product_dir.glob("*.tif")))})
    pd.DataFrame(product_counts).to_csv(OUT_DIR / "current_snap_product_counts.csv", index=False, encoding="utf-8-sig")

    print(f"Story board written to {OUT_DIR}")


if __name__ == "__main__":
    main()
