from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_snap_lai_qc as audit  # noqa: E402


def add_counts(store: dict[int, dict[str, int]], scl_code: int, qc: np.ndarray, lai: np.ndarray, mask: np.ndarray) -> None:
    if not mask.any():
        return
    qc_i = np.rint(qc[mask]).astype(np.int64, copy=False)
    lai_v = lai[mask]
    high = lai_v > 0.8
    qc0 = qc_i == 0
    input_out = (qc_i & 1) != 0
    too_high = (qc_i & (1 << 4)) != 0
    thresholded_high = (qc_i & (1 << 2)) != 0

    row = store[scl_code]
    row["pixels"] += int(mask.sum())
    row["lai_gt_0_8"] += int(high.sum())
    row["qc0"] += int(qc0.sum())
    row["qc0_lai_gt_0_8"] += int((qc0 & high).sum())
    row["qc_nonzero_lai_gt_0_8"] += int(((~qc0) & high).sum())
    row["input_out_of_range"] += int(input_out.sum())
    row["input_out_of_range_lai_gt_0_8"] += int((input_out & high).sum())
    row["output_too_high"] += int(too_high.sum())
    row["output_too_high_lai_gt_0_8"] += int((too_high & high).sum())
    row["output_thresholded_high"] += int(thresholded_high.sum())
    row["output_thresholded_high_lai_gt_0_8"] += int((thresholded_high & high).sum())


def main() -> None:
    audit.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    _, raw_index = audit.build_raw_inventory()
    rows: list[dict[str, Any]] = []

    for tif_path in sorted(audit.BIOPHYSICAL_DIR.glob("Biophysical_*.tif")):
        meta = audit.parse_biophysical_name(tif_path.name)
        raw = raw_index.get((meta["mission"], meta["date"], meta["tile"]))
        if not raw or not raw.get("has_scl_20m"):
            continue

        print(f"crosschecking {tif_path.name}")
        counts: dict[int, dict[str, int]] = defaultdict(
            lambda: {
                "pixels": 0,
                "lai_gt_0_8": 0,
                "qc0": 0,
                "qc0_lai_gt_0_8": 0,
                "qc_nonzero_lai_gt_0_8": 0,
                "input_out_of_range": 0,
                "input_out_of_range_lai_gt_0_8": 0,
                "output_too_high": 0,
                "output_too_high_lai_gt_0_8": 0,
                "output_thresholded_high": 0,
                "output_thresholded_high_lai_gt_0_8": 0,
            }
        )

        with rasterio.open(tif_path) as src:
            with rasterio.open(audit.vsizip_path(raw["path"], raw["scl_20m_inner_path"])) as scl_src:
                with WarpedVRT(
                    scl_src,
                    crs=src.crs,
                    transform=src.transform,
                    width=src.width,
                    height=src.height,
                    resampling=Resampling.nearest,
                ) as scl_vrt:
                    for _, window in src.block_windows(1):
                        lai_ma = src.read(1, window=window, masked=True)
                        qc_ma = src.read(2, window=window, masked=True)
                        scl_ma = scl_vrt.read(1, window=window, masked=True)

                        lai = np.ma.getdata(lai_ma)
                        qc = np.ma.getdata(qc_ma)
                        scl = np.ma.getdata(scl_ma)
                        valid = (
                            ~np.ma.getmaskarray(lai_ma)
                            & ~np.ma.getmaskarray(qc_ma)
                            & ~np.ma.getmaskarray(scl_ma)
                            & np.isfinite(lai)
                            & np.isfinite(qc)
                        )
                        if not valid.any():
                            continue
                        for scl_code_raw in np.unique(scl[valid]):
                            scl_code = int(scl_code_raw)
                            add_counts(counts, scl_code, qc, lai, valid & (scl == scl_code))

        for scl_code, values in sorted(counts.items()):
            pixels = values["pixels"]
            qc0 = values["qc0"]
            row = {
                **meta,
                "file_name": tif_path.name,
                "scl_code": scl_code,
                "scl_class": audit.SCL_CLASSES.get(scl_code, "UNKNOWN"),
                **values,
                "lai_gt_0_8_pct": values["lai_gt_0_8"] / pixels if pixels else None,
                "qc0_pct": qc0 / pixels if pixels else None,
                "qc0_lai_gt_0_8_pct_of_all": values["qc0_lai_gt_0_8"] / pixels if pixels else None,
                "qc0_lai_gt_0_8_pct_of_qc0": values["qc0_lai_gt_0_8"] / qc0 if qc0 else None,
                "high_lai_with_qc_nonzero_pct_of_high_lai": values["qc_nonzero_lai_gt_0_8"] / values["lai_gt_0_8"]
                if values["lai_gt_0_8"]
                else None,
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    out_path = audit.AUDIT_DIR / "lai_scl_qc_crosscheck.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    lines = ["# LAI SCL/QC crosscheck", ""]
    if out.empty:
        lines.append("No matched SCL scenes were available.")
    else:
        scl5 = out[out["scl_code"] == 5]
        if not scl5.empty:
            total = int(scl5["pixels"].sum())
            high = int(scl5["lai_gt_0_8"].sum())
            qc0 = int(scl5["qc0"].sum())
            qc0_high = int(scl5["qc0_lai_gt_0_8"].sum())
            qcnz_high = int(scl5["qc_nonzero_lai_gt_0_8"].sum())
            lines.extend(
                [
                    f"- SCL=5 total pixels: {total:,}",
                    f"- SCL=5 LAI > 0.8 pixels: {high:,} ({high / total:.2%})",
                    f"- SCL=5 QC=0 pixels: {qc0:,} ({qc0 / total:.2%})",
                    f"- SCL=5 QC=0 and LAI > 0.8 pixels: {qc0_high:,} ({qc0_high / qc0:.2%} of QC=0 SCL=5)",
                    f"- SCL=5 high LAI with nonzero QC: {qcnz_high:,} ({qcnz_high / high:.2%} of high-LAI SCL=5)",
                ]
            )
        lines.extend(["", f"Full table: `{out_path.name}`"])
    (audit.AUDIT_DIR / "README_LAI_SCL_QC_CROSSCHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
