"""
plotting.py

这个文件集中负责生态服务 DID 分析结果的所有 Plotly 可视化输出。

【整体设计思路】
1. 输入：
   - event_study、group_time 等 pandas.DataFrame；
   - 这些表一般由 run_pipeline.py 前序步骤计算生成。
2. 输出：
   - 多个 HTML 图件，文件名以 Dashboard_01 / Dashboard_02 等顺序编号开头；
   - 每个图件都通过 _write_html() 统一写出，因此全局的 HTML 后处理也放在这里。
3. 图件编号逻辑：
   - 01：事件研究百分比 ATT 曲线；
   - 02：辐射剖面 heatmap；
   - 03：观测值分布诊断图；
   - 04/05：t-1 到 t0 的观测突降等级诊断图；
   - 06：滑坡面积与 DID 恢复速度之间的拓展分析。
4. 当前绘图规范：
   - 坐标轴范围保持真实最小/最大刻度，不通过左右扩展 x 轴来“避开裁切”；
   - 位于边界刻度上的点，通过 HTML/SVG 后处理取消 Plotly clip-path 来完整显示；
   - t-1 作为参考期，只显示普通虚线，不显示 marker 点；
   - 子图左上角小字母背景为完全透明，避免遮挡曲线、热力图或网格线。
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .io_utils import ensure_dir
# ---------------------------------------------------------------------
# 全局配色与排序设置
# ---------------------------------------------------------------------
# 每个生态指标固定一个主色，保证同一指标在全部主图中颜色一致。
# 当前主结果只展示 FVC、LAI_SR2、GPP；FAPAR 仅作为 GPP 的中间输入。
OUTCOME_COLORS = {
    "LAI_SR2": "#b2182b",
    "FVC": "#1b7837",
    "FAPAR": "#e08214",
    "GPP": "#2166ac",
    "VOD": "#6a3d9a",
}

# FAPAR 只作为 GPP 的中间输入，不再作为主结果图展示。
APPENDIX_OUTCOMES: set[str] = {"FAPAR"}

# core_inner 是把滑坡体边缘向内侵蚀后的保守核心区，用来减少 10 m 像元的边缘混合影响。
DIRECT_ZONE_NAMES = {"core", "core_weighted", "core_inner"}

# 当输入表里出现 OUTCOME_COLORS 没有预先定义的新指标时，使用备用色。
# 备用色的作用是保证代码不因为新增 outcome 而中断，同时仍然给出可区分的线条颜色。
FALLBACK_COLORS = [
    "#4d4d4d",
    "#762a83",
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#66a61e",
]


def _safe_name(value: str) -> str:
    """Helper."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _ordered_zones(zones: list[str]) -> list[str]:
    """Helper."""

    def key(zone: str) -> tuple[int, float]:
        if zone == "core":
            return (0, 0.0)
        if zone == "core_weighted":
            return (0, 0.5)
        if zone == "core_inner":
            return (0, 1.0)
        if zone.startswith("ring_"):
            parts = zone.replace("m", "").split("_")
            return (1, float(parts[1]))
        return (2, 999999.0)

    return sorted(zones, key=key)


def _is_direct_zone_name(value: object) -> bool:
    """Helper."""
    return str(value) in DIRECT_ZONE_NAMES


def _preferred_direct_zone(df: pd.DataFrame) -> str:
    """Helper."""
    zones = set(df.get("zone_name", pd.Series(dtype=str)).dropna().astype(str).tolist())
    if "core" in zones:
        return "core"
    if "core_weighted" in zones:
        return "core_weighted"
    if "core_inner" in zones:
        return "core_inner"
    return "core"


def _zone_label(zone_name: str) -> str:
    """Readable zone labels used in captions and titles."""
    labels = {
        "core": "core",
        "core_weighted": "weighted core",
        "core_inner": "inner core",
        "ring_0_10m": "0-10 m ring",
        "ring_10_20m": "10-20 m ring",
    }
    return labels.get(str(zone_name), str(zone_name))


def _ordered_outcomes(outcomes: list[str]) -> list[str]:
    """Helper."""
    preferred = ["FVC", "LAI_SR2", "GPP"]
    rank = {name: i for i, name in enumerate(preferred)}
    return sorted(outcomes, key=lambda x: (rank.get(str(x), 999), str(x)))


def _main_outcomes(outcomes: list[str]) -> list[str]:
    """Helper."""
    return [outcome for outcome in _ordered_outcomes(outcomes) if outcome not in APPENDIX_OUTCOMES]


def _outcome_color(outcome: str) -> str:
    """Helper."""
    if outcome in OUTCOME_COLORS:
        return OUTCOME_COLORS[outcome]
    idx = abs(hash(outcome)) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Helper."""
    color = hex_color.lstrip("#")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _mix_hex(color_a: str, color_b: str, t: float) -> str:
    """
    在两个不透明 hex 颜色之间线性混合。

    t=0 返回 color_a；t=1 返回 color_b。
    这里专门用于 heatmap，避免 rgba 透明度在浏览器端叠白造成色阶感知不稳定。
    """
    t = max(0.0, min(1.0, float(t)))
    a = color_a.lstrip("#")
    b = color_b.lstrip("#")
    ar, ag, ab = int(a[0:2], 16), int(a[2:4], 16), int(a[4:6], 16)
    br, bg, bb = int(b[0:2], 16), int(b[2:4], 16), int(b[4:6], 16)
    rr = round(ar * (1 - t) + br * t)
    gg = round(ag * (1 - t) + bg * t)
    bb2 = round(ab * (1 - t) + bb * t)
    return f"#{rr:02x}{gg:02x}{bb2:02x}"



def _article_layout(fig: go.Figure, title: str, height: int) -> None:
    """Helper."""
    fig.update_layout(
        title=dict(
            text=title,
            x=0.0,
            xanchor="left",
            y=0.985,
            yanchor="top",
            font=dict(size=20, family="Arial, Microsoft YaHei, sans-serif", color="#111"),
        ),
        height=height,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=82, r=160, t=78, b=155),
        font=dict(family="Arial, Microsoft YaHei, sans-serif", size=13, color="#222"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e9e9e9",
        zeroline=False,
        ticks="outside",
        showline=True,
        mirror=True,
        linecolor="#111",
        layer="below traces",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e9e9e9",
        zeroline=False,
        ticks="outside",
        showline=True,
        mirror=True,
        linecolor="#111",
        layer="below traces",
    )


def _write_html(fig: go.Figure, path: Path) -> None:
    """Write Plotly html and force scatter points/lines above subplot frames without expanding axes."""
    # 不能通过扩大 x 轴 range 解决端点裁切；坐标轴范围仍然由各绘图函数设置为 min/max tick。
    # 这里做两件事：
    # 1) 坐标轴/边框图层放在 traces 下面；
    # 2) HTML 渲染后移除 scatter 所在 SVG plot group 的 clip-path，让位于边界刻度上的点完整显示。
    fig.update_xaxes(layer="below traces")
    fig.update_yaxes(layer="below traces")
    fig.update_traces(cliponaxis=False, selector=dict(type="scatter"))

    html = fig.to_html(include_plotlyjs=True, full_html=True)
    layer_fix = """
<script>
(function () {
  function removeClip(el) {
    if (!el) return;
    el.removeAttribute('clip-path');
    el.style.clipPath = 'none';
  }

  function fixPlotlyScatterLayering() {
    document.querySelectorAll('.js-plotly-plot').forEach(function(gd) {
      gd.querySelectorAll('.scatterlayer').forEach(function(layer) {
        removeClip(layer);

        // Plotly 的裁切通常不在 marker 自己身上，而是在父级 .plot/.xy/.subplot group 上。
        // 逐级移除这些父级 clip-path，才能让 x=min/x=max 的边界点完整露出。
        var node = layer;
        for (var i = 0; i < 8 && node && node.tagName && node.tagName.toLowerCase() !== 'svg'; i++) {
          removeClip(node);
          node = node.parentNode;
        }

        layer.querySelectorAll('*').forEach(removeClip);

        // 把 scatterlayer 放到同一 subplot group 的最后，保证点线在坐标轴框之上。
        if (layer.parentNode && layer.parentNode.lastElementChild !== layer) {
          layer.parentNode.appendChild(layer);
        }
      });

      gd.querySelectorAll('.annotationlayer').forEach(function(layer) {
        if (layer.parentNode && layer.parentNode.lastElementChild !== layer) {
          layer.parentNode.appendChild(layer);
        }
      });
    });
  }

  var fixingScheduled = false;
  function scheduleFixes() {
    if (fixingScheduled) return;
    fixingScheduled = true;
    setTimeout(function() {
      fixingScheduled = false;
      fixPlotlyScatterLayering();
      setTimeout(fixPlotlyScatterLayering, 100);
      setTimeout(fixPlotlyScatterLayering, 400);
      setTimeout(fixPlotlyScatterLayering, 1000);
    }, 0);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', scheduleFixes);
  } else {
    scheduleFixes();
  }
  window.addEventListener('load', scheduleFixes);

  // 本地 HTML 初次打开、缩放、悬停重绘后 Plotly 可能重建 SVG，继续补一次。
  var observer = new MutationObserver(function() { scheduleFixes(); });
  observer.observe(document.documentElement, {childList: true, subtree: true});

  document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.js-plotly-plot').forEach(function(gd) {
      if (gd.on) {
        gd.on('plotly_afterplot', scheduleFixes);
        gd.on('plotly_relayout', scheduleFixes);
        gd.on('plotly_redraw', scheduleFixes);
      }
    });
  });
})();
</script>
"""
    if "</body>" in html:
        html = html.replace("</body>", layer_fix + "</body>")
    else:
        html += layer_fix
    path.write_text(html, encoding="utf-8")


def _panel_letter(index: int) -> str:
    """Helper."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    if index <= len(letters):
        return letters[index - 1]
    return f"{letters[(index - 1) % len(letters)]}{(index - 1) // len(letters) + 1}"


def _axis_key(axis: str, row: int, col: int, cols: int) -> str:
    """Helper."""
    subplot_index = (row - 1) * cols + col
    return f"{axis}axis" if subplot_index == 1 else f"{axis}axis{subplot_index}"


def _subplot_domain(fig: go.Figure, row: int, col: int, cols: int) -> tuple[list[float], list[float]]:
    """Helper."""
    xaxis = fig.layout[_axis_key("x", row, col, cols)]
    yaxis = fig.layout[_axis_key("y", row, col, cols)]
    return list(xaxis.domain), list(yaxis.domain)


def _add_panel_letters(fig: go.Figure, rows: int, cols: int, labels: list[str] | None = None) -> None:
    """Helper."""
    total = rows * cols
    labels = labels or [_panel_letter(i) for i in range(1, total + 1)]
    for idx, label in enumerate(labels[:total], start=1):
        row = (idx - 1) // cols + 1
        col = (idx - 1) % cols + 1
        x_domain, y_domain = _subplot_domain(fig, row, col, cols)
        fig.add_annotation(
            x=x_domain[0] + 0.012,
            y=y_domain[1] - 0.018,
            xref="paper",
            yref="paper",
            text=f"<b>({label})</b>",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(size=14, color="#111"),
            # 子图角标背景设置为完全透明：
            # - 不再使用半透明白底，避免挡住曲线、散点、热力图色块或网格线；
            # - 角标文字仍然加粗，通常足够清晰；
            # - 如果后续某张图背景过深，只需要单独调整 font 颜色，而不是恢复白底。
            bgcolor="rgba(255,255,255,0)", # RGB alpha，透明度设置为0 
        )


def _caption_text(lead: str, descriptions: list[str]) -> str:
    """把总图说明和 (a)(b)(c) 子图说明拼成论文式图注。"""
    pieces = [lead.rstrip(".") + "."]
    for idx, desc in enumerate(descriptions, start=1):
        pieces.append(f"({_panel_letter(idx)}) {desc}")
    return " ".join(pieces)


def _add_caption(fig: go.Figure, text: str, y: float = -0.16) -> None:
    """在 HTML 图底部添加统一图注，便于后续截图或迁移到 PPT。"""
    fig.add_annotation(
        x=0,
        y=y,
        xref="paper",
        yref="paper",
        text=text,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font=dict(size=12, color="#222"),
    )


def _add_global_axis_titles(fig: go.Figure, x_title: str = "", y_title: str = "") -> None:
    """添加整张大图共享的横纵轴标题，减少每个子图重复标注。"""
    if y_title:
        fig.add_annotation(
            x=-0.075,
            y=0.5,
            xref="paper",
            yref="paper",
            text=y_title,
            showarrow=False,
            textangle=-90,
            xanchor="center",
            yanchor="middle",
            font=dict(size=15, color="#111"),
        )
    if x_title:
        fig.add_annotation(
            x=0.5,
            y=-0.085,
            xref="paper",
            yref="paper",
            text=x_title,
            showarrow=False,
            xanchor="center",
            yanchor="top",
            font=dict(size=15, color="#111"),
        )


def _apply_subplot_axes(
    fig: go.Figure,
    rows: int,
    cols: int,
    x_title: str = "",
    y_title: str = "",
    show_y_title_col: int = 1,
) -> None:
    """Helper."""
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            fig.update_xaxes(
                title_text="",
                showticklabels=(row == rows),
                showline=True,
                mirror=True,
                ticks="outside",
                linecolor="#111",
                layer="below traces",
                row=row,
                col=col,
            )
            fig.update_yaxes(
                title_text="",
                showticklabels=(col == 1),
                showline=True,
                mirror=True,
                ticks="outside",
                linecolor="#111",
                layer="below traces",
                row=row,
                col=col,
            )
    _add_global_axis_titles(fig, x_title=x_title, y_title=y_title)


def _trace_line(color: str, width: float = 2.4, dash: str = "solid") -> dict[str, object]:
    """Helper."""
    return dict(color=color, width=width, dash=dash, shape="linear")


def _trace_marker(color: str, size: float = 6, symbol: str = "circle") -> dict[str, object]:
    """Helper."""
    return dict(size=size, color=color, symbol=symbol)


def _pchip_slopes(x_values: list[float], y_values: list[float]) -> list[float]:
    """Helper."""
    n = len(x_values)
    if n <= 1:
        return [0.0] * n
    h = [x_values[i + 1] - x_values[i] for i in range(n - 1)]
    delta = [(y_values[i + 1] - y_values[i]) / h[i] if h[i] != 0 else 0.0 for i in range(n - 1)]
    if n == 2:
        return [delta[0], delta[0]]

    slopes = [0.0] * n

    def endpoint_slope(h0: float, h1: float, d0: float, d1: float) -> float:
        if h0 == 0:
            return 0.0
        slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1) if (h0 + h1) != 0 else d0
        if slope * d0 <= 0:
            return 0.0
        if d0 * d1 < 0 and abs(slope) > abs(3 * d0):
            return 3 * d0
        return slope

    slopes[0] = endpoint_slope(h[0], h[1], delta[0], delta[1])
    slopes[-1] = endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            slopes[i] = 0.0
            continue
        w1 = 2 * h[i] + h[i - 1]
        w2 = h[i] + 2 * h[i - 1]
        denom = (w1 / delta[i - 1]) + (w2 / delta[i])
        slopes[i] = (w1 + w2) / denom if denom != 0 else 0.0
    return slopes


def _smooth_xy(
    x_values: pd.Series,
    y_values: pd.Series,
    points_per_segment: int = 18,
) -> tuple[list[float], list[float]]:
    """Helper."""
    curve = (
        pd.DataFrame({"x": pd.to_numeric(x_values, errors="coerce"), "y": pd.to_numeric(y_values, errors="coerce")})
        .dropna()
        .groupby("x", as_index=False)["y"]
        .mean()
        .sort_values("x")
    )
    if len(curve) <= 2:
        return curve["x"].astype(float).tolist(), curve["y"].astype(float).tolist()

    xs = curve["x"].astype(float).tolist()
    ys = curve["y"].astype(float).tolist()
    slopes = _pchip_slopes(xs, ys)
    smooth_x: list[float] = []
    smooth_y: list[float] = []
    steps = max(int(points_per_segment), 2)
    for i in range(len(xs) - 1):
        h = xs[i + 1] - xs[i]
        if h == 0:
            continue
        for step in range(steps + 1):
            if i > 0 and step == 0:
                continue
            t = step / steps
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            y = h00 * ys[i] + h10 * h * slopes[i] + h01 * ys[i + 1] + h11 * h * slopes[i + 1]
            lo = min(ys[i], ys[i + 1])
            hi = max(ys[i], ys[i + 1])
            smooth_x.append(xs[i] + t * h)
            smooth_y.append(min(max(y, lo), hi))
    return smooth_x, smooth_y



def _subset_marker_payload(payload: object, keep_mask: pd.Series, original_index: pd.Index) -> object:
    """Subset marker/customdata payloads when removing the t-1 marker from relative-year plots."""
    if payload is None:
        return None
    if not hasattr(payload, "__len__") or isinstance(payload, (str, bytes)):
        return payload
    try:
        payload_len = len(payload)  # type: ignore[arg-type]
    except TypeError:
        return payload
    if payload_len != len(keep_mask):
        return payload
    if isinstance(payload, pd.Series):
        if payload.index.equals(original_index):
            return payload.loc[keep_mask].tolist()
        return payload.iloc[keep_mask.to_numpy()].tolist()
    if isinstance(payload, pd.DataFrame):
        if payload.index.equals(original_index):
            return payload.loc[keep_mask].to_numpy()
        return payload.iloc[keep_mask.to_numpy()].to_numpy()
    return [item for item, keep in zip(list(payload), keep_mask.tolist()) if keep]


def _payload_item(payload: object, idx: int, default: object) -> object:
    """从 marker_symbol / marker_size 这类 payload 中取第 idx 个值；如果是标量，则直接返回标量。"""
    if payload is None:
        return default
    if isinstance(payload, pd.Series):
        return payload.iloc[idx] if len(payload) > idx else default
    if isinstance(payload, pd.DataFrame):
        return payload.iloc[idx].tolist() if len(payload) > idx else default
    if isinstance(payload, (list, tuple)):
        return payload[idx] if len(payload) > idx else default
    if hasattr(payload, "__len__") and not isinstance(payload, (str, bytes)):
        try:
            values = list(payload)
            return values[idx] if len(values) > idx else default
        except TypeError:
            return payload
    return payload


def _marker_symbol_to_text(symbol: object) -> str:
    """
    把 Plotly marker symbol 转换成 annotation 可以显示的文本符号。

    原因：
    Plotly 的 scatter marker 位于 scatterlayer；即使设置 cliponaxis=False，
    边界点仍可能被单独的 axislayer 压住。annotation layer 位于更上层，
    所以这里用 annotation 叠加边界视觉点，确保点真正盖住坐标轴框。
    """
    symbol_text = str(symbol).lower()
    if "square" in symbol_text:
        return "□" if "open" in symbol_text else "■"
    if "diamond" in symbol_text:
        return "◇" if "open" in symbol_text else "◆"
    if "triangle" in symbol_text:
        return "△" if "open" in symbol_text else "▲"
    if symbol_text == "x" or symbol_text.endswith("-x"):
        return "×"
    if "cross" in symbol_text:
        return "+"
    return "○" if "open" in symbol_text else "●"


def _marker_size_to_font_size(size_value: object) -> float:
    """把 Plotly marker 的像素大小近似转换为 annotation 字体大小。"""
    try:
        size = float(size_value)
    except (TypeError, ValueError):
        size = 6.0
    return max(size * 1.85, 9.0)


def _add_top_boundary_marker_annotations(
    fig: go.Figure,
    marker_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xref: str,
    yref: str,
    color: str,
    marker_symbol_payload: object,
    marker_size_payload: object,
    opacity: float = 1.0,
) -> None:
    """
    在 annotation layer 上叠加边界点，让边界点盖住坐标轴。

    这不会改变数据和 x 轴范围；原 scatter marker 仍保留用于 hover。
    annotation 只负责视觉层级，主要解决 01 系列端点被坐标轴框压住的问题。
    """
    if marker_df.empty:
        return
    x_numeric = pd.to_numeric(marker_df[x_col], errors="coerce")
    if not x_numeric.notna().any():
        return
    valid_x = x_numeric.dropna()
    edge_values = {float(valid_x.min()), float(valid_x.max())}

    for idx, x_value in enumerate(x_numeric.tolist()):
        if pd.isna(x_value) or float(x_value) not in edge_values:
            continue
        row = marker_df.iloc[idx]
        symbol = _payload_item(marker_symbol_payload, idx, "circle")
        size_value = _payload_item(marker_size_payload, idx, 6)
        fig.add_annotation(
            x=row[x_col],
            y=row[y_col],
            xref=xref,
            yref=yref,
            text=_marker_symbol_to_text(symbol),
            showarrow=False,
            xanchor="center",
            yanchor="middle",
            align="center",
            font=dict(
                size=_marker_size_to_font_size(size_value),
                color=color,
                family="Arial, Microsoft YaHei, sans-serif",
            ),
            opacity=opacity,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            captureevents=False,
        )

def _add_smooth_point_line(
    fig: go.Figure,
    zdf: pd.DataFrame,
    x_col: str,
    y_col: str,
    row: int,
    col: int,
    color: str,
    name: str,
    showlegend: bool,
    dash: str = "solid",
    width: float = 2.4,
    marker_symbol: str | list[str] = "circle",
    marker_size: float | list[float] = 6,
    marker_line_width: float = 0.0,
    legendgroup: str | None = None,
    hovertemplate: str | None = None,
    customdata: object | None = None,
    opacity: float = 1.0,
) -> None:
    """Helper."""
    # 只保留 x/y 都存在的点，避免 Plotly 在缺失值位置断线或产生异常 hover。
    plot_df = zdf.dropna(subset=[x_col, y_col]).sort_values(x_col).copy()
    if plot_df.empty:
        return

    # 平滑曲线使用完整 plot_df：
    # 对 relative_year 图而言，t-1 参考点会参与曲线连接，使曲线从基准期 0 开始；
    # 但 t-1 的 marker 会在下面单独过滤掉，所以视觉上只留下虚线，不留下点。
    smooth_x, smooth_y = _smooth_xy(plot_df[x_col], plot_df[y_col])
    fig.add_trace(
        go.Scatter(
            x=smooth_x,
            y=smooth_y,
            mode="lines",
            line=_trace_line(color, width=width, dash=dash),
            opacity=opacity,
            name=name,
            legendgroup=legendgroup or name,
            showlegend=showlegend,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )
    marker_df = plot_df.copy()
    keep_marker = pd.Series(True, index=plot_df.index)
    if x_col == "relative_year":
        keep_marker = pd.to_numeric(plot_df[x_col], errors="coerce").ne(-1)
        marker_df = plot_df.loc[keep_marker].copy()

    if not marker_df.empty:
        marker_symbol_payload = _subset_marker_payload(marker_symbol, keep_marker, plot_df.index)
        marker_size_payload = _subset_marker_payload(marker_size, keep_marker, plot_df.index)
        customdata_payload = _subset_marker_payload(customdata, keep_marker, plot_df.index)
        fig.add_trace(
            go.Scatter(
                x=marker_df[x_col],
                y=marker_df[y_col],
                mode="markers",
                marker=dict(
                    size=marker_size_payload,
                    symbol=marker_symbol_payload,
                    color=color,
                    line=dict(color=color, width=marker_line_width),
                ),
                opacity=opacity,
                name=name,
                legendgroup=legendgroup or name,
                showlegend=False,
                customdata=customdata_payload,
                hovertemplate=hovertemplate,
                cliponaxis=False,
            ),
            row=row,
            col=col,
        )

        # 关键：scatter marker 可能仍被 Plotly 的 axislayer 压住。
        # 因此读取刚加入的 marker trace 的坐标轴引用，在 annotation layer
        # 上叠加端点视觉点，使边界点真正位于坐标轴框上方。
        marker_trace = fig.data[-1]
        _add_top_boundary_marker_annotations(
            fig=fig,
            marker_df=marker_df,
            x_col=x_col,
            y_col=y_col,
            xref=getattr(marker_trace, "xaxis", "x") or "x",
            yref=getattr(marker_trace, "yaxis", "y") or "y",
            color=color,
            marker_symbol_payload=marker_symbol_payload,
            marker_size_payload=marker_size_payload,
            opacity=opacity,
        )


def _outcome_zone_titles(outcomes: list[str], zones: list[str]) -> list[str]:
    """Helper."""
    return [f"{outcome} | {zone}" for outcome in outcomes for zone in zones]


def _symmetric_range(values: pd.Series, pad: float = 0.10) -> list[float]:
    """Helper."""
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return [-1.0, 1.0]
    max_abs = max(abs(float(vals.min())), abs(float(vals.max())))
    if max_abs == 0:
        max_abs = 1.0
    return [-max_abs * (1 + pad), max_abs * (1 + pad)]


def _positive_range(values: pd.Series, pad: float = 0.10) -> list[float]:
    """Helper."""
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return [-1.0, 1.0]
    vmax = max(float(vals.max()), 1.0)
    return [-vmax * 0.06, vmax * (1 + pad)]


def _nice_step(raw_step: float) -> float:
    """Helper."""
    if not math.isfinite(raw_step) or raw_step <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    base = 10**exponent
    fraction = raw_step / base
    for nice in (1.0, 2.0, 2.5, 5.0, 10.0):
        if fraction <= nice:
            return nice * base
    return 10.0 * base


def _format_tick(value: float) -> str:
    """Helper."""
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.6g}"


def _axis_tick_spec(values: pd.Series, mode: str = "symmetric", major_count: int = 5) -> dict[str, object]:
    """Helper."""
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        vals = pd.Series([0.0])
    major_count = max(int(major_count), 3)

    if mode == "positive":
        upper_raw = max(float(vals.max()), 0.0)
        step = _nice_step((upper_raw * 1.08) / (major_count - 1))
        upper = step * (major_count - 1)
        if upper <= 0:
            upper = step * (major_count - 1)
        lower = 0.0
        tickvals = [lower + step * i for i in range(major_count)]
    else:
        extent_raw = max(abs(float(vals.min())), abs(float(vals.max())))
        step = _nice_step((extent_raw * 1.08) / 2.0)
        extent = step * 2.0
        if extent <= 0:
            step = 1.0
            extent = 2.0
        lower = -extent
        upper = extent
        tickvals = [lower + step * i for i in range(5)]

    minor_tickvals = [(tickvals[i] + tickvals[i + 1]) / 2.0 for i in range(len(tickvals) - 1)]
    return {
        "range": [float(lower), float(upper)],
        "tickvals": [float(x) for x in tickvals],
        "ticktext": [_format_tick(float(x)) for x in tickvals],
        "minor_tickvals": [float(x) for x in minor_tickvals],
    }


def _fixed_positive_axis_spec(upper: float = 40.0, step: float = 10.0) -> dict[str, object]:
    """Helper."""
    count = int(round(float(upper) / float(step)))
    tickvals = [float(step) * i for i in range(count + 1)]
    minor_tickvals = [(tickvals[i] + tickvals[i + 1]) / 2.0 for i in range(len(tickvals) - 1)]
    return {
        "range": [0.0, float(upper)],
        "tickvals": tickvals,
        "ticktext": [_format_tick(float(x)) for x in tickvals],
        "minor_tickvals": minor_tickvals,
    }


def _fixed_axis_tick_spec(lower: float, upper: float, step: float) -> dict[str, object]:
    """Build a fixed y-axis spec for percent event-study figures."""
    lower = float(lower)
    upper = float(upper)
    step = float(step)
    count = int(round((upper - lower) / step))
    tickvals = [lower + step * i for i in range(count + 1)]
    minor_tickvals = [(tickvals[i] + tickvals[i + 1]) / 2.0 for i in range(len(tickvals) - 1)]
    return {
        "range": [lower, upper],
        "tickvals": [float(x) for x in tickvals],
        "ticktext": [_format_tick(float(x)) for x in tickvals],
        "minor_tickvals": [float(x) for x in minor_tickvals],
    }


EVENT_STUDY_PLOT_RELATIVE_YEARS = list(range(-2, 6))
EVENT_STUDY_PERCENT_Y_SPEC = _fixed_axis_tick_spec(lower=-80.0, upper=20.0, step=20.0)


def _clip_event_window(df: pd.DataFrame, years: list[int] | None = None) -> pd.DataFrame:
    """Keep the shared t-2 to t+5 display window for event-study figures."""
    if df.empty or "relative_year" not in df.columns:
        return df.copy()
    keep_years = EVENT_STUDY_PLOT_RELATIVE_YEARS if years is None else [int(y) for y in years]
    out = df.copy()
    out["relative_year"] = pd.to_numeric(out["relative_year"], errors="coerce")
    out = out[out["relative_year"].isin(keep_years)].copy()
    out["relative_year"] = out["relative_year"].astype(int)
    return out


def _apply_y_tick_spec(fig: go.Figure, row: int, col: int, spec: dict[str, object]) -> None:
    """Helper."""
    fig.update_yaxes(
        range=spec["range"],
        tickmode="array",
        tickvals=spec["tickvals"],
        ticktext=spec["ticktext"],
        minor=dict(
            tickmode="array",
            tickvals=spec["minor_tickvals"],
            ticks="outside",
            ticklen=3,
            showgrid=True,
            gridcolor="#f3f3f3",
        ),
        row=row,
        col=col,
    )


def _stack_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Helper."""
    series_list = [pd.to_numeric(df[col], errors="coerce") for col in columns if col in df.columns]
    if not series_list:
        return pd.Series(dtype=float)
    return pd.concat(series_list, ignore_index=True)


def _add_reference_row(zdf: pd.DataFrame, value_col: str, reference_year: int = -1, value: float = 0.0) -> pd.DataFrame:
    """Helper."""
    if zdf.empty or "relative_year" not in zdf.columns:
        return zdf
    if (zdf["relative_year"].astype(int) == reference_year).any():
        return zdf.sort_values("relative_year")

    reference: dict[str, object] = {}
    for col in zdf.columns:
        if col == "relative_year":
            reference[col] = reference_year
        elif col in {value_col, "att", "ci_low", "ci_high"} or col.endswith(
            ("_ci_low", "_ci_high")
        ):
            reference[col] = value
        elif zdf[col].notna().any():
            reference[col] = zdf[col].dropna().iloc[0]

    out = pd.concat([zdf, pd.DataFrame([reference])], ignore_index=True, sort=False)
    return out.reindex(columns=zdf.columns).sort_values("relative_year")


def _add_reference_guides(fig: go.Figure, row: int, col: int, color: str, y_value: float = 0.0) -> None:
    """Helper."""
    fig.add_hline(y=y_value, line_width=1.3, line_color="black", layer="below", row=row, col=col)
    fig.add_vline(x=-1, line_width=1.3, line_dash="dash", line_color="black", layer="above", row=row, col=col)


def _add_interval_ribbon(
    fig: go.Figure,
    zdf: pd.DataFrame,
    x_col: str,
    low_col: str,
    high_col: str,
    row: int,
    col: int,
    color: str,
    fill_alpha: float = 0.14,
) -> None:
    """Helper."""
    required = {x_col, low_col, high_col}
    if not required.issubset(zdf.columns):
        return
    ci_df = zdf[[x_col, low_col, high_col]].dropna().sort_values(x_col)
    if ci_df.empty:
        return

    high_x, high_y = _smooth_xy(ci_df[x_col], ci_df[high_col])
    low_x, low_y = _smooth_xy(ci_df[x_col], ci_df[low_col])
    fig.add_trace(
        go.Scatter(
            x=high_x + low_x[::-1],
            y=high_y + low_y[::-1],
            fill="toself",
            fillcolor=_hex_to_rgba(color, fill_alpha),
            line=dict(color="rgba(0,0,0,0)", width=0),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    for _, smooth_x, smooth_y in [(high_col, high_x, high_y), (low_col, low_x, low_y)]:
        fig.add_trace(
            go.Scatter(
                x=smooth_x,
                y=smooth_y,
                mode="lines",
                line=dict(color=_hex_to_rgba(color, 0.72), width=1.15, dash="dash", shape="linear"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def _add_error_bars(
    fig: go.Figure,
    zdf: pd.DataFrame,
    x_col: str,
    y_col: str,
    low_col: str,
    high_col: str,
    row: int,
    col: int,
    color: str,
) -> None:
    """Helper."""
    required = {x_col, y_col, low_col, high_col}
    if not required.issubset(zdf.columns):
        return
    err_df = zdf[[x_col, y_col, low_col, high_col]].dropna().sort_values(x_col)
    if err_df.empty:
        return

    y = pd.to_numeric(err_df[y_col], errors="coerce")
    low = pd.to_numeric(err_df[low_col], errors="coerce")
    high = pd.to_numeric(err_df[high_col], errors="coerce")
    upper = (high - y).clip(lower=0).to_numpy()
    lower = (y - low).clip(lower=0).to_numpy()
    fig.add_trace(
        go.Scatter(
            x=err_df[x_col],
            y=y,
            mode="markers",
            marker=dict(size=1, color=_hex_to_rgba(color, 0.0), line=dict(width=0)),
            error_y=dict(
                type="data",
                symmetric=False,
                array=upper,
                arrayminus=lower,
                color=_hex_to_rgba(color, 0.62),
                thickness=1.25,
                width=4,
            ),
            hoverinfo="skip",
            showlegend=False,
            cliponaxis=False,
        ),
        row=row,
        col=col,
    )


def _add_ci_ribbon(fig: go.Figure, zdf: pd.DataFrame, x_col: str, row: int, col: int, color: str) -> None:
    """Helper."""
    _add_interval_ribbon(
        fig,
        zdf,
        x_col=x_col,
        low_col="ci_low",
        high_col="ci_high",
        row=row,
        col=col,
        color=color,
        fill_alpha=0.16,
    )


def _add_event_trace(
    fig: go.Figure,
    zdf: pd.DataFrame,
    row: int,
    col: int,
    color: str,
    outcome: str,
    showlegend: bool,
) -> None:
    """Helper."""
    plot_df = _add_reference_row(zdf, value_col="att", reference_year=-1, value=0.0)
    _add_ci_ribbon(fig, plot_df, x_col="relative_year", row=row, col=col, color=color)

    marker_symbols = ["circle" for _ in plot_df["relative_year"]]
    marker_sizes = [5 for _ in plot_df["relative_year"]]
    _add_smooth_point_line(
        fig,
        plot_df,
        x_col="relative_year",
        y_col="att",
        row=row,
        col=col,
        color=color,
        name=outcome,
        showlegend=showlegend,
        width=2.4,
        marker_symbol=marker_symbols,
        marker_size=marker_sizes,
        marker_line_width=0,
        legendgroup=outcome,
    )
    _add_reference_guides(fig, row=row, col=col, color=color, y_value=0.0)


def _add_event_percent_trace(
    fig: go.Figure,
    zdf: pd.DataFrame,
    row: int,
    col: int,
    color: str,
    outcome: str,
    showlegend: bool,
) -> None:
    """Helper."""
    plot_df = zdf.copy()
    # 百分比 Event Study 保持 DID ATT% 原始方向：生态下降为负值。
    plot_df["att"] = plot_df["att_percent"]
    plot_df["ci_low"] = plot_df["ci_low_percent"]
    plot_df["ci_high"] = plot_df["ci_high_percent"]
    _add_event_trace(fig, plot_df, row=row, col=col, color=color, outcome=outcome, showlegend=showlegend)


def _add_radiation_trace(
    fig: go.Figure,
    zdf: pd.DataFrame,
    row: int,
    col: int,
    color: str,
    outcome: str,
    showlegend: bool,
) -> None:
    """Helper."""
    plot_df = zdf.copy()
    plot_df["distance_mid_m"] = plot_df.apply(_distance_midpoint, axis=1)
    plot_df["distance_label"] = plot_df.apply(_distance_label, axis=1)
    plot_df = plot_df.sort_values("distance_mid_m")
    plot_df["distance_index"] = range(len(plot_df))
    _add_ci_ribbon(fig, plot_df, x_col="distance_index", row=row, col=col, color=color)
    _add_smooth_point_line(
        fig,
        plot_df,
        x_col="distance_index",
        y_col="att",
        row=row,
        col=col,
        color=color,
        name=outcome,
        showlegend=showlegend,
        width=2.4,
        marker_size=6,
        legendgroup=outcome,
        customdata=plot_df["distance_label"],
        hovertemplate="Distance bin: %{customdata}<br>DID ATT: %{y:.4g}<extra></extra>",
    )
    fig.add_hline(y=0, line_width=1.3, line_color="black", layer="below", row=row, col=col)
    fig.update_xaxes(
        tickmode="array",
        tickvals=plot_df["distance_index"].tolist(),
        ticktext=plot_df["distance_label"].tolist(),
        range=[-0.35, max(float(len(plot_df) - 1) + 0.35, 0.65)],
        row=row,
        col=col,
    )


def _add_radiation_year_trace(
    fig: go.Figure,
    zdf: pd.DataFrame,
    row: int,
    col: int,
    color: str,
    label: str,
    legendgroup: str,
    showlegend: bool,
    dash: str = "solid",
) -> None:
    """Helper."""
    plot_df = zdf.copy()
    plot_df["distance_mid_m"] = plot_df.apply(_distance_midpoint, axis=1)
    plot_df["distance_label"] = plot_df.apply(_distance_label, axis=1)
    plot_df = plot_df.sort_values("distance_mid_m")
    plot_df["distance_index"] = range(len(plot_df))
    _add_ci_ribbon(fig, plot_df, x_col="distance_index", row=row, col=col, color=color)
    _add_smooth_point_line(
        fig,
        plot_df,
        x_col="distance_index",
        y_col="att",
        row=row,
        col=col,
        color=color,
        name=label,
        showlegend=showlegend,
        dash=dash,
        width=2.25,
        marker_size=6,
        legendgroup=legendgroup,
        customdata=plot_df["distance_label"],
        hovertemplate="Distance bin: %{customdata}<br>" + label + "<br>DID ATT: %{y:.4g}<extra></extra>",
    )
    fig.add_hline(y=0, line_width=1.15, line_color="black", layer="below", row=row, col=col)
    fig.update_xaxes(
        tickmode="array",
        tickvals=plot_df["distance_index"].tolist(),
        ticktext=plot_df["distance_label"].tolist(),
        range=[0, max(float(len(plot_df) - 1), 0.65)],
        row=row,
        col=col,
    )


def _baseline_radiation_rows(chunk: pd.DataFrame, outcome: str, relative_year: int = -1) -> pd.DataFrame:
    """Helper."""
    template = chunk[chunk["relative_year"].astype(int) != relative_year].copy()
    if template.empty:
        return pd.DataFrame()
    zones = template[["zone_name", "distance_min_m", "distance_max_m"]].drop_duplicates()
    rows = []
    for _, zone in zones.iterrows():
        zone_name = str(zone["zone_name"])
        if float(zone["distance_max_m"]) <= 0 and not _is_direct_zone_name(zone_name):
            continue
        rows.append(
            {
                "outcome": outcome,
                "zone_name": zone_name,
                "relative_year": relative_year,
                "att": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "distance_min_m": zone["distance_min_m"],
                "distance_max_m": zone["distance_max_m"],
            }
        )
    return pd.DataFrame(rows)


def _distance_midpoint(row: pd.Series) -> float:
    """Helper."""
    zone_name = str(row.get("zone_name", ""))
    if zone_name == "core_inner":
        return -1.0
    if zone_name == "core":
        return 0.0
    return (float(row["distance_min_m"]) + float(row["distance_max_m"])) / 2.0


def _distance_label(row: pd.Series) -> str:
    """Helper."""
    zone_name = str(row.get("zone_name", ""))
    if zone_name == "core_inner":
        return "core inner"
    if zone_name == "core":
        return "core"
    dmin = int(float(row["distance_min_m"]))
    dmax = int(float(row["distance_max_m"]))
    return f"{dmin}-{dmax} m"


def _radiation_chunk_with_reference(chunk: pd.DataFrame, outcome: str, relative_years: list[int]) -> pd.DataFrame:
    """Helper."""
    frames: list[pd.DataFrame] = []
    available = set(chunk["relative_year"].astype(int).tolist())
    for rel_year in relative_years:
        if rel_year == -1 and rel_year not in available:
            zdf = _baseline_radiation_rows(chunk, outcome, relative_year=-1)
        else:
            zdf = chunk[chunk["relative_year"].astype(int) == rel_year].copy()
        if not zdf.empty:
            frames.append(zdf)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["distance_mid_m"] = out.apply(_distance_midpoint, axis=1)
    out["distance_label"] = out.apply(_distance_label, axis=1)
    return out.sort_values(["distance_mid_m", "relative_year"])


def _format_ci(row: pd.Series) -> str:
    """Helper."""
    ci_low = row.get("ci_low", pd.NA)
    ci_high = row.get("ci_high", pd.NA)
    p_value = row.get("p_value", pd.NA)
    n_treated = row.get("n_treated", pd.NA)
    n_control = row.get("n_control_mean", row.get("n_control", pd.NA))
    pieces = []
    if pd.notna(ci_low) and pd.notna(ci_high):
        pieces.append(f"95% CI: [{float(ci_low):.4g}, {float(ci_high):.4g}]")
    if pd.notna(p_value):
        pieces.append(f"p: {float(p_value):.3g}")
    if pd.notna(n_treated):
        pieces.append(f"treated zones: {int(float(n_treated))}")
    if pd.notna(n_control):
        pieces.append(f"control mean: {float(n_control):.1f}")
    return "<br>".join(pieces)


def _outcome_heat_colorscale(outcome: str) -> list[list[object]]:
    """
    零中心、不透明、单调的 radiation heatmap 色带。

    旧版使用 rgba 透明度作为负值梯度，浏览器会把颜色再与白色背景叠加，
    视觉上容易出现“浅负值看起来比深负值更重”的错觉。
    这里改为完全不透明的连续色带：
    - zmin -> outcome 主色，代表最大负效应；
    - 0 -> 白色；
    - zmax -> 深灰，代表正效应。
    """
    color = _outcome_color(outcome)
    white = "#ffffff"
    positive_dark = "#4d4d4d"
    return [
        [0.00, color],
        [0.10, _mix_hex(color, white, 0.20)],
        [0.20, _mix_hex(color, white, 0.40)],
        [0.30, _mix_hex(color, white, 0.60)],
        [0.40, _mix_hex(color, white, 0.80)],
        [0.50, white],
        [0.60, _mix_hex(white, positive_dark, 0.20)],
        [0.70, _mix_hex(white, positive_dark, 0.40)],
        [0.80, _mix_hex(white, positive_dark, 0.60)],
        [0.90, _mix_hex(white, positive_dark, 0.80)],
        [1.00, positive_dark],
    ]



def _outcome_intensity_colorscale(outcome: str) -> list[list[object]]:
    """Helper."""
    color = _outcome_color(outcome)
    return [
        [0.0, "#f2f2f2"],
        [0.12, _hex_to_rgba(color, 0.12)],
        [0.28, _hex_to_rgba(color, 0.28)],
        [0.45, _hex_to_rgba(color, 0.48)],
        [0.68, _hex_to_rgba(color, 0.72)],
        [1.0, color],
    ]


def _add_radiation_heatmap(
    fig: go.Figure,
    chunk: pd.DataFrame,
    outcome: str,
    relative_years: list[int],
    row: int,
    col: int,
    show_colorbar: bool,
    colorbar_kwargs: dict[str, object] | None = None,
) -> None:
    """Helper."""
    plot_df = _radiation_chunk_with_reference(chunk, outcome, relative_years)
    if plot_df.empty:
        return

    plot_df["hover_extra"] = plot_df.apply(_format_ci, axis=1)
    distance_order = (
        plot_df[["distance_label", "distance_mid_m"]]
        .drop_duplicates()
        .sort_values("distance_mid_m")["distance_label"]
        .tolist()
    )
    available_years = set(plot_df["relative_year"].astype(int).tolist())
    year_order = [int(y) for y in relative_years if int(y) in available_years]
    z = (
        plot_df.pivot_table(index="distance_label", columns="relative_year", values="att", aggfunc="mean")
        .reindex(index=distance_order, columns=year_order)
    )
    hover = (
        plot_df.pivot_table(index="distance_label", columns="relative_year", values="hover_extra", aggfunc="first")
        .reindex(index=distance_order, columns=year_order)
    )
    # heatmap 颜色必须围绕 0 对称映射：
    # - 最大负效应使用 outcome 主色；
    # - 0 使用白色；
    # - 最大正效应使用深灰色。
    # z_abs 使用当前 outcome 当前图层内的最大绝对值，保证 colorbar 与色块严格对应。
    # cohort heatmap 也使用 0 中心的对称色标，防止不同年份色块与 colorbar 脱节。
    finite_values = z.stack().dropna()
    z_abs = max(abs(float(finite_values.min())), abs(float(finite_values.max()))) if not finite_values.empty else 1.0
    if z_abs == 0:
        z_abs = 1.0

    fig.add_trace(
        go.Heatmap(
            x=[f"t{int(y):+d}" for y in year_order],
            y=distance_order,
            z=z.to_numpy(dtype=float),
            customdata=hover.fillna("").to_numpy(),
            colorscale=_outcome_heat_colorscale(outcome),
            zauto=False,
            zmid=0,
            zmin=-z_abs,
            zmax=z_abs,
            colorbar=(
                {
                    "title": dict(text="DID ATT", side="right"),
                    "len": 0.34,
                    "thickness": 12,
                    "tickmode": "array",
                    "tickvals": [-z_abs, z_abs],
                    "ticktext": [f"{-z_abs:.2g}", f"{z_abs:.2g}"],
                    **(colorbar_kwargs or {}),
                }
                if show_colorbar
                else None
            ),
            showscale=show_colorbar,
            hovertemplate=(
                "Relative year: %{x}<br>"
                "Distance bin: %{y}<br>"
                "DID ATT: %{z:.4g}<br>"
                "%{customdata}<extra></extra>"
            ),
            name=outcome,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(title_text="Relative year", row=row, col=col)
    fig.update_yaxes(title_text="Distance bin" if col == 1 else "", autorange="reversed", row=row, col=col)


def plot_event_study_percent(percent_event_study: pd.DataFrame, figures_dir: str | Path) -> list[Path]:
    """Helper."""
    figures_dir = ensure_dir(figures_dir)
    paths: list[Path] = []
    required = {"att_percent", "ci_low_percent", "ci_high_percent", "counterfactual_mean"}
    if percent_event_study.empty or not required.issubset(percent_event_study.columns):
        return paths

    df = _clip_event_window(percent_event_study).dropna(subset=["att_percent", "counterfactual_mean"]).copy()
    outcomes = _main_outcomes(df["outcome"].dropna().unique().tolist())
    if not outcomes:
        return paths
    zones = _ordered_zones(df["zone_name"].dropna().unique().tolist())
    # 百分比 Event Study 用固定尺度，便于横向比较不同 outcome 和不同 zone。
    # 保持 DID ATT% 原始方向：下降为负值，恢复向 0 靠近。
    y_spec = EVENT_STUDY_PERCENT_Y_SPEC
    fig = make_subplots(
        rows=len(outcomes),
        cols=len(zones),
        horizontal_spacing=0.065,
        vertical_spacing=0.085,
    )
    for r, outcome in enumerate(outcomes, start=1):
        color = _outcome_color(str(outcome))
        for c, zone_name in enumerate(zones, start=1):
            zdf = df[(df["outcome"] == outcome) & (df["zone_name"] == zone_name)].sort_values("relative_year")
            if zdf.empty:
                continue
            _add_event_percent_trace(fig, zdf, row=r, col=c, color=color, outcome=str(outcome), showlegend=(c == 1))
            fig.update_xaxes(range=[-2, 5], row=r, col=c)
            _apply_y_tick_spec(fig, row=r, col=c, spec=y_spec)
    _article_layout(fig, "Event Study Percent Change (all outcomes)", max(620, 285 * len(outcomes)))
    _apply_subplot_axes(
        fig,
        rows=len(outcomes),
        cols=len(zones),
        x_title="Years since landslide",
        y_title="DID ATT (%)",
    )
    _add_panel_letters(fig, rows=len(outcomes), cols=len(zones))
    _add_caption(
        fig,
        _caption_text(
            "Percent event study in town Jiangwan. Values are DID ATT relative to the counterfactual level; ecological decline is negative, t-1 is the reference period, and the display window is t-2 to t+5",
            [f"{outcome} {_zone_label(zone)} percent ATT trajectory" for outcome in outcomes for zone in zones],
        ),
    )
    dashboard_path = figures_dir / "Dashboard_01_Event_Study_Percent.html"
    _write_html(fig, dashboard_path)
    paths.append(dashboard_path)
    return paths


def plot_radiation_profiles(
    event_study: pd.DataFrame,
    figures_dir: str | Path,
    relative_years: list[int],
) -> list[Path]:
    """Helper."""
    figures_dir = ensure_dir(figures_dir)
    paths: list[Path] = []
    df = event_study[
        event_study["zone_name"].map(_is_direct_zone_name) | (event_study["distance_max_m"] > 0)
    ].copy()
    df = df[df["zone_name"].astype(str) != "core_weighted"].copy()
    if df.empty:
        return paths
    df = _clip_event_window(df)
    if df.empty:
        return paths

    outcomes = _main_outcomes(_ordered_outcomes(df["outcome"].dropna().unique().tolist()))
    if not outcomes:
        return paths
    available_all = set(df["relative_year"].astype(int).tolist())
    present_years = [int(y) for y in EVENT_STUDY_PLOT_RELATIVE_YEARS if int(y) in available_all or int(y) == -1]
    if not present_years:
        return paths

    rows = len(outcomes)
    fig = make_subplots(rows=rows, cols=1, vertical_spacing=0.10)
    for row, outcome in enumerate(outcomes, start=1):
        _, y_domain = _subplot_domain(fig, row, 1, 1)
        _add_radiation_heatmap(
            fig,
            df[df["outcome"] == outcome].copy(),
            str(outcome),
            present_years,
            row=row,
            col=1,
            show_colorbar=True,
            colorbar_kwargs={
                "x": 1.02,
                "y": (y_domain[0] + y_domain[1]) / 2,
                "len": y_domain[1] - y_domain[0],
                "thickness": 18,
            },
        )

    _article_layout(fig, "Radiation Profile Matrix (all outcomes)", max(680, 285 * rows))
    _apply_subplot_axes(fig, rows=rows, cols=1, x_title="Relative year", y_title="Distance bin")
    _add_panel_letters(fig, rows=rows, cols=1)
    _add_caption(
        fig,
        _caption_text(
            "Radiation profile heatmap in town Jiangwan. Rows are distance bins from the landslide body, columns are relative years from t-2 to t+5, and colors show DID ATT",
            [f"{outcome} all-cohort radiation profile" for outcome in outcomes],
        ),
    )
    fig.update_layout(margin=dict(l=78, r=190, t=78, b=155), showlegend=False)
    dashboard_path = figures_dir / "Dashboard_02_Radiation_Profile.html"
    _write_html(fig, dashboard_path)
    paths.append(dashboard_path)
    return paths

