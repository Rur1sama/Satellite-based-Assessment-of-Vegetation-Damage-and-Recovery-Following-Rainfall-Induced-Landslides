from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots


ROOT = Path(r"E:\python_base\Slide_dect")
DID_DIR = ROOT / "Output" / "EcoService_DID"
DID_RESULTS = DID_DIR / "04_did_results"
AUDIT_DIR = DID_DIR / "06_snap_lai_qc_audit"
OUT_DIR = DID_DIR / "07_change_tracking_storyboard"
CHART_DIR = OUT_DIR / "charts"
SNAP_SINGLE = ROOT / "Output" / "SNAP_Results" / "Biophysical_SingleBand"

OUTCOME_LABELS = {"LAI_SR2": "LAI", "FVC": "FVC", "FAPAR": "FAPAR"}
OUTCOME_ORDER = ["LAI_SR2", "FVC", "FAPAR"]
ZONE_LABELS = {
    "core_inner": "滑坡核心内侧",
    "core": "滑坡核心",
    "ring_0_10m": "外圈 0-10 m",
    "ring_10_20m": "外圈 10-20 m",
}
ZONE_ORDER = ["core", "ring_0_10m", "ring_10_20m"]
COLORS = {"core": "#d24d3e", "ring_0_10m": "#e39b32", "ring_10_20m": "#2f7f9f", "core_inner": "#8b1e3f"}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def apply_layout(fig: go.Figure, title: str, height: int = 520) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        font=dict(family="Microsoft YaHei, SimHei, Arial", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=55, r=30, t=80, b=85),
    )
    return fig


def event_study_fig(event: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=3, subplot_titles=[OUTCOME_LABELS[o] for o in OUTCOME_ORDER], shared_yaxes=False)
    df = event[event["zone_name"].isin(ZONE_ORDER)].copy()
    for col, outcome in enumerate(OUTCOME_ORDER, start=1):
        sub = df[df["outcome"] == outcome]
        for zone in ZONE_ORDER:
            z = sub[sub["zone_name"] == zone].sort_values("relative_year")
            if z.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=z["relative_year"],
                    y=z["att"],
                    mode="lines+markers",
                    name=ZONE_LABELS[zone],
                    legendgroup=zone,
                    showlegend=(col == 1),
                    line=dict(color=COLORS[zone], width=3),
                    marker=dict(size=7),
                    hovertemplate="相对年=%{x}<br>DID变化=%{y:.3f}<extra>" + ZONE_LABELS[zone] + "</extra>",
                ),
                row=1,
                col=col,
            )
        fig.add_hline(y=0, line_color="#334155", line_width=1, row=1, col=col)
        fig.add_vline(x=0, line_color="#64748b", line_width=1, line_dash="dash", row=1, col=col)
        fig.update_xaxes(title_text="相对滑坡年份", row=1, col=col)
    fig.update_yaxes(title_text="DID变化量（负值=下降）", row=1, col=1)
    return apply_layout(fig, "图1  滑坡前后变化曲线：核心区突降，外圈仍有影响", 500)


def focus_2024_fig(focus: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, subplot_titles=[OUTCOME_LABELS[o] for o in OUTCOME_ORDER], shared_xaxes=True)
    df = focus[focus["zone_name"].isin(ZONE_ORDER)].copy()
    for row, outcome in enumerate(OUTCOME_ORDER, start=1):
        sub = df[df["outcome"] == outcome]
        for zone in ZONE_ORDER:
            z = sub[sub["zone_name"] == zone].sort_values("relative_year")
            if z.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=z["relative_year"],
                    y=z["treated_change"],
                    mode="lines+markers",
                    name=f"{ZONE_LABELS[zone]}：滑坡区",
                    legendgroup=f"{zone}_treated",
                    showlegend=(row == 1),
                    line=dict(color=COLORS[zone], width=2.7),
                    hovertemplate="相对年=%{x}<br>滑坡区变化=%{y:.3f}<extra></extra>",
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=z["relative_year"],
                    y=z["control_change"],
                    mode="lines+markers",
                    name=f"{ZONE_LABELS[zone]}：对照",
                    legendgroup=f"{zone}_control",
                    showlegend=(row == 1),
                    line=dict(color=COLORS[zone], width=1.8, dash="dash"),
                    marker=dict(symbol="x"),
                    hovertemplate="相对年=%{x}<br>对照变化=%{y:.3f}<extra></extra>",
                ),
                row=row,
                col=1,
            )
        fig.add_hline(y=0, line_color="#334155", line_width=1, row=row, col=1)
        fig.add_vline(x=0, line_color="#64748b", line_width=1, line_dash="dash", row=row, col=1)
        fig.update_yaxes(title_text=OUTCOME_LABELS[outcome], row=row, col=1)
    fig.update_xaxes(title_text="相对 2024 滑坡年份", row=3, col=1)
    return apply_layout(fig, "图2  2024 年滑坡组：看滑坡区是否比对照下降更多", 820)


def recovery_fig(recovery: pd.DataFrame) -> go.Figure:
    df = recovery[recovery["zone_name"].isin(["core_inner", "core", "ring_0_10m", "ring_10_20m"])].copy()
    df["zone_label"] = df["zone_name"].map(ZONE_LABELS)
    df["outcome_label"] = df["outcome"].map(OUTCOME_LABELS)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["最大下降量", "恢复到80%所需年数"])
    for outcome in OUTCOME_ORDER:
        sub = df[df["outcome"] == outcome]
        fig.add_trace(
            go.Bar(
                y=sub["zone_label"],
                x=sub["max_damage_att"],
                orientation="h",
                name=OUTCOME_LABELS[outcome],
                hovertemplate="%{y}<br>最大下降=%{x:.3f}<extra>" + OUTCOME_LABELS[outcome] + "</extra>",
            ),
            row=1,
            col=1,
        )
    heat = df.pivot_table(index="zone_label", columns="outcome_label", values="time_to_80pct_recovery", aggfunc="first")
    heat = heat.reindex([ZONE_LABELS[z] for z in ["core_inner", "core", "ring_0_10m", "ring_10_20m"]])
    fig.add_trace(
        go.Heatmap(
            z=heat.values,
            x=heat.columns,
            y=heat.index,
            colorscale="YlGnBu",
            text=heat.values,
            texttemplate="%{text}年",
            hovertemplate="%{y}<br>%{x}<br>80%恢复=%{z}年<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="DID变化量（负值=下降）", row=1, col=1)
    return apply_layout(fig, "图3  影响强度与恢复速度：核心区最强，外圈较弱但可跟踪", 560)


def scene_backbone_fig(variable_stats: pd.DataFrame) -> go.Figure:
    df = variable_stats[variable_stats["tile"] == "T49RGH"].copy()
    df["year"] = df["date"].astype(str).str[:4].astype(int)
    fig = make_subplots(rows=1, cols=3, subplot_titles=["LAI", "FVC", "FAPAR"])
    for col, var in enumerate(["LAI", "FVC", "FAPAR"], start=1):
        sub = df[df["variable"] == var].sort_values("year")
        fig.add_trace(
            go.Scatter(
                x=sub["year"],
                y=sub["mean"],
                mode="lines+markers",
                name=var,
                line=dict(width=3),
                showlegend=False,
                hovertemplate="年份=%{x}<br>整景均值=%{y:.3f}<extra>" + var + "</extra>",
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="年份", row=1, col=col)
    fig.update_yaxes(title_text="整景均值", row=1, col=1)
    return apply_layout(fig, "图4  数据骨架：T49RGH 的 2018-2025 年序列已经排齐", 470)


def scl_qc_fig(cross: pd.DataFrame) -> go.Figure:
    df = cross[cross["scl_code"] == 5].copy()
    df["year"] = df["date"].astype(str).str[:4].astype(int)
    df = df.sort_values("year")
    fig = go.Figure()
    fig.add_bar(
        x=df["year"],
        y=df["qc0_lai_gt_0_8_pct_of_all"] * 100,
        name="QC=0 且 LAI>0.8",
        marker_color="#d24d3e",
        hovertemplate="年份=%{x}<br>比例=%{y:.2f}%<extra></extra>",
    )
    fig.add_bar(
        x=df["year"],
        y=df["qc_nonzero_lai_gt_0_8"] / df["pixels"] * 100,
        name="QC非零 且 LAI>0.8",
        marker_color="#64748b",
        hovertemplate="年份=%{x}<br>比例=%{y:.2f}%<extra></extra>",
    )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="占 SCL=5 像元比例 (%)")
    fig.update_xaxes(title_text="年份")
    return apply_layout(fig, "图5  裸地高 LAI 是诊断标签，不是删除规则", 500)


def drop_class_fig(drop: pd.DataFrame) -> go.Figure:
    valid = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    df = drop[(drop["zone_name"] == "core") & (drop["outcome"].isin(OUTCOME_ORDER)) & (drop["drop_class"].isin(valid))]
    grouped = df.groupby(["outcome", "drop_class"], as_index=False)["n_landslides"].sum()
    fig = go.Figure()
    for outcome in OUTCOME_ORDER:
        sub = grouped[grouped["outcome"] == outcome].set_index("drop_class").reindex(valid).reset_index()
        fig.add_bar(x=sub["drop_class"], y=sub["n_landslides"], name=OUTCOME_LABELS[outcome])
    fig.update_xaxes(title_text="事件年相对滑坡前的下降比例")
    fig.update_yaxes(title_text="滑坡数量")
    return apply_layout(fig, "图6  单个滑坡区下降幅度分组：用于识别明显突降对象", 500)


def write_chart(fig: go.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def build_key_numbers(recovery: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOME_ORDER:
        for zone in ["core", "ring_0_10m", "ring_10_20m"]:
            sub = recovery[(recovery["outcome"] == outcome) & (recovery["zone_name"] == zone)]
            if sub.empty:
                continue
            r = sub.iloc[0]
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
                "说明": f"SCL=5 且 LAI>0.8 占 {scl5['lai_gt_0_8'].sum() / scl5['pixels'].sum():.2%}；不要先删，作为解释标签。",
            }
        )
    return pd.DataFrame(rows)


def write_readme(key: pd.DataFrame) -> None:
    lines = [
        "# 滑坡变化跟踪版结果",
        "",
        "这版不是 QC 清洗报告，而是给滑坡识别/跟踪看的图文结果。",
        "",
        "## 怎么看",
        "",
        "1. 打开 `index.html`，按图1到图6读。",
        "2. 图1看总体：滑坡发生年是否突降，外圈是否也下降。",
        "3. 图2看2024：滑坡区是否比对照下降更多。",
        "4. 图3看哪个指标最敏感、哪个区域恢复慢。",
        "5. 图5提醒：裸地高LAI不要直接删，它是解释标签。",
        "",
        "## 关键原则",
        "",
        "- 不用 `LAI>0.8` 作为删除条件。",
        "- 不先用 QC 波段剔除像元。",
        "- 保留 LAI/FVC/FAPAR 原始时间序列，用 SCL/QC 给变化信号分级解释。",
        "- 识别滑坡时看“滑坡前高、事件年突降、之后恢复”的形状。",
        "",
        "## T49RFH",
        "",
        "T49RFH 是多下载瓦片，已从 `Output\\SNAP_Results` 派生产品中删除；本版只看 T49RGH 的 8 景序列。",
        "",
        "## 文件",
        "",
        "- `index.html`：一页图文总览。",
        "- `charts/*.html`：每张图单独打开版。",
        "- `tracking_key_numbers.csv`：少量关键数字。",
        "- `current_snap_product_counts.csv`：当前 SNAP 产品数量。",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "README_变化跟踪版.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_index(chart_divs: list[tuple[str, str]], key: pd.DataFrame) -> None:
    cards = []
    for _, row in key.head(9).iterrows():
        if pd.notna(row.get("最大下降")):
            cards.append(
                f"<div class='card'><b>{row['指标']} / {row['区域']}</b>"
                f"<span>最大下降 {row['最大下降']:.3f}</span>"
                f"<span>80%恢复：{row['恢复到80%所需年数']} 年</span></div>"
            )
    sections = "\n".join(f"<section><h2>{title}</h2>{div}</section>" for title, div in chart_divs)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>滑坡变化跟踪版结果</title>
<script>{get_plotlyjs()}</script>
<style>
body {{ margin:0; font-family:"Microsoft YaHei","Segoe UI",sans-serif; color:#17202a; background:#f3f5f7; }}
header {{ padding:28px 42px; background:#1f2933; color:white; }}
h1 {{ margin:0 0 8px; font-size:28px; }}
main {{ max-width:1200px; margin:0 auto; padding:26px; }}
.note {{ background:white; border-left:5px solid #d24d3e; padding:18px 20px; margin-bottom:20px; line-height:1.8; }}
.cards {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:18px 0 26px; }}
.card {{ background:white; padding:14px 16px; border:1px solid #d8dee6; border-radius:8px; display:flex; flex-direction:column; gap:7px; }}
section {{ background:white; padding:20px; margin:18px 0; border:1px solid #d8dee6; border-radius:8px; }}
h2 {{ margin-top:0; font-size:20px; }}
</style>
</head>
<body>
<header>
<h1>滑坡变化跟踪版结果</h1>
<p>保留 LAI/FVC/FAPAR 时间序列，看突降、影响范围和恢复；SCL/QC 只做解释标签。</p>
</header>
<main>
<div class="note"><b>读图原则：</b> 单期 LAI 偏高不等于无效。滑坡识别更看重同一区域在滑坡前后是否出现“突降 - 低谷 - 恢复”的形状。</div>
<div class="cards">{''.join(cards)}</div>
{sections}
</main>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    event = read_csv(DID_RESULTS / "event_study_att.csv")
    focus = read_csv(DID_RESULTS / "focus_2024_event_study_att.csv")
    recovery = read_csv(DID_RESULTS / "recovery_metrics.csv")
    drop = read_csv(DID_RESULTS / "individual_zone_drop_class_summary.csv")
    variable_stats = read_csv(AUDIT_DIR / "biophysical_variable_stats.csv")
    cross = read_csv(AUDIT_DIR / "lai_scl_qc_crosscheck.csv")

    figs = [
        ("图1 滑坡前后变化曲线", event_study_fig(event), CHART_DIR / "fig01_event_study_tracking.html"),
        ("图2 2024滑坡组 vs 对照", focus_2024_fig(focus), CHART_DIR / "fig02_focus_2024_treated_vs_control.html"),
        ("图3 影响强度与恢复速度", recovery_fig(recovery), CHART_DIR / "fig03_damage_and_recovery.html"),
        ("图4 年度产品数据骨架", scene_backbone_fig(variable_stats), CHART_DIR / "fig04_scene_backbone.html"),
        ("图5 裸地高LAI诊断", scl_qc_fig(cross), CHART_DIR / "fig05_scl5_high_lai_diagnostic.html"),
        ("图6 明显突降对象分组", drop_class_fig(drop), CHART_DIR / "fig06_drop_class_counts.html"),
    ]
    chart_divs = [(title, write_chart(fig, path)) for title, fig, path in figs]

    key = build_key_numbers(recovery, cross)
    key.to_csv(OUT_DIR / "tracking_key_numbers.csv", index=False, encoding="utf-8-sig")
    product_counts = []
    for product_dir in sorted(SNAP_SINGLE.glob("*")):
        if product_dir.is_dir():
            product_counts.append({"product": product_dir.name, "tif_count": len(list(product_dir.glob("*.tif")))})
    pd.DataFrame(product_counts).to_csv(OUT_DIR / "current_snap_product_counts.csv", index=False, encoding="utf-8-sig")

    write_readme(key)
    write_index(chart_divs, key)
    print(f"Plotly storyboard written to {OUT_DIR}")


if __name__ == "__main__":
    main()
