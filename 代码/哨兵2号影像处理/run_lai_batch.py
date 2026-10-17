"""
==============================================================================
Project       : Slide_dect
Module Name   : run_lai_batch.py
Description   : 批量调用 SNAP Biophysical10mOp 反演 LAI/FVC/FAPAR，并拆分单波段栅格。
==============================================================================
"""
from __future__ import annotations

import argparse
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import rasterio


# ==================== 配置区域 ====================
# SNAP gpt.exe 绝对路径。
GPT_PATH = r"D:\esa-snap\bin\gpt.exe"

# SNAP Graph XML 模板路径。
TEMPLATE_XML = r"E:\SNAP_base\SR2_LAI_cal\Graphs\myBiophysical10mOp.xml"

# Sentinel-2 L2A 原始数据目录。
INPUT_DIR = r"E:\Raster_data\Sentinel-2\韶关市江湾镇\L2A_raw"

# SNAP 多波段 Biophysical 输出目录。
OUTPUT_DIR = r"E:\python_base\Slide_dect\Output\SNAP_Results\Biophysical"

# 单波段 LAI/FVC/FAPAR 输出目录。
SINGLE_BAND_OUTPUT_DIR = r"E:\python_base\Slide_dect\Output\SNAP_Results\Biophysical_SingleBand"

# SNAP Biophysical10mOp 的 GeoTIFF band 顺序。现有 tif 没有 band description，因此这里显式固定。
# 注意：SNAP UI 和 BiophysicalVariable 枚举顺序均为 LAI -> FAPAR -> FVC/FCOVER。
BIOPHYSICAL_BAND_ORDER = {
    1: "LAI",
    2: "LAIQC",
    3: "FAPAR",
    4: "FAPARQC",
    5: "FVC",
    6: "FVCQC",
}

# 生态服务分析只使用这三个主变量，QC band 暂不单独输出。
PRIMARY_PRODUCT_BANDS = {
    "LAI": 1,
    "FAPAR": 3,
    "FVC": 5,
}
# ==================================================


def _extract_s2_metadata(file_name: str) -> dict[str, str]:
    """从 Sentinel-2 文件名中提取日期、基线、轨道、瓦片和传感器信息。"""
    sensor_type = "S2A_10m"
    if "S2A" in file_name:
        sensor_type = "S2A_10m"
    elif "S2B" in file_name:
        sensor_type = "S2B_10m"

    parts = file_name.split("_")
    return {
        "date": parts[2].split("T")[0] if len(parts) > 2 else "UnknownDate",
        "baseline": parts[3] if len(parts) > 3 else "UnknownBaseline",
        "relative_orbit": parts[4] if len(parts) > 4 else "UnknownOrbit",
        "tile_id": parts[5] if len(parts) > 5 else "UnknownTile",
        "sensor_type": sensor_type,
    }


def _build_output_name(meta: dict[str, str]) -> str:
    """构造多波段 Biophysical 输出文件名。"""
    return (
        f"Biophysical_{meta['date']}_{meta['baseline']}_{meta['relative_orbit']}_"
        f"{meta['tile_id']}_{meta['sensor_type']}.tif"
    )


def _patch_snap_graph(template_xml: str | Path, input_path: str | Path, output_path: str | Path, sensor_type: str) -> ET.ElementTree:
    """读取 SNAP XML 模板，并把输入路径、传感器类型和输出路径改成当前影像。"""
    tree = ET.parse(template_xml)
    root = tree.getroot()

    read_node = root.find(".//node[@id='Read']")
    if read_node is not None:
        file_tag = read_node.find(".//file")
        if file_tag is not None:
            file_tag.text = str(input_path)

    bio_node = root.find(".//node[@id='Biophysical10mOp']")
    if bio_node is not None:
        sensor_tag = bio_node.find(".//sensor")
        if sensor_tag is not None:
            sensor_tag.text = sensor_type

    write_node = root.find(".//node[@id='Write']")
    if write_node is not None:
        source_product = write_node.find(".//sources/sourceProduct")
        if source_product is not None:
            source_product.set("refid", "Biophysical10mOp")

        file_tag = write_node.find(".//file")
        if file_tag is not None:
            file_tag.text = str(output_path)

        format_tag = write_node.find(".//formatName")
        if format_tag is not None:
            format_tag.text = "GeoTIFF-BigTIFF"
        else:
            params = write_node.find(".//parameters")
            if params is not None:
                new_format = ET.SubElement(params, "formatName")
                new_format.text = "GeoTIFF-BigTIFF"

    return tree


def validate_biophysical_tif(tif_path: str | Path) -> None:
    """检查 SNAP 多波段结果是否至少包含 LAI/LAIQC/FVC/FVCQC/FAPAR/FAPARQC 六个 band。"""
    tif_path = Path(tif_path)
    with rasterio.open(tif_path) as src:
        if src.count < len(BIOPHYSICAL_BAND_ORDER):
            raise ValueError(f"{tif_path} 只有 {src.count} 个 band，少于期望的 6 个 Biophysical band。")
        descriptions = src.descriptions
        print(f"  -> band count: {src.count}")
        for band_index, band_name in BIOPHYSICAL_BAND_ORDER.items():
            desc = descriptions[band_index - 1] if band_index <= len(descriptions) else None
            desc_text = desc if desc else "未写入 description，使用固定 SNAP band 顺序"
            print(f"     band {band_index}: {band_name} ({desc_text})")


def split_primary_bands(
    multiband_path: str | Path,
    single_band_root: str | Path = SINGLE_BAND_OUTPUT_DIR,
    overwrite: bool = True,
) -> list[Path]:
    """把多波段 Biophysical tif 拆成 LAI、FVC、FAPAR 三个单波段 GeoTIFF。"""
    multiband_path = Path(multiband_path)
    single_band_root = Path(single_band_root)
    written: list[Path] = []

    validate_biophysical_tif(multiband_path)
    with rasterio.open(multiband_path) as src:
        profile = src.profile.copy()
        profile.update(
            count=1,
            driver="GTiff",
            BIGTIFF="IF_SAFER",
            compress="deflate",
            predictor=2,
            tiled=True,
        )

        for product_name, source_band in PRIMARY_PRODUCT_BANDS.items():
            product_dir = single_band_root / product_name
            product_dir.mkdir(parents=True, exist_ok=True)
            out_name = multiband_path.name.replace("Biophysical_", f"{product_name}_", 1)
            out_path = product_dir / out_name
            if out_path.exists() and not overwrite:
                print(f"  -> 已存在，跳过: {out_path}")
                written.append(out_path)
                continue

            data = src.read(source_band)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data, 1)
                dst.set_band_description(1, product_name)
                dst.update_tags(
                    1,
                    product=product_name,
                    source_band=str(source_band),
                    source_band_order="1=LAI,2=LAIQC,3=FAPAR,4=FAPARQC,5=FVC,6=FVCQC",
                    source_file=multiband_path.name,
                )
                dst.update_tags(product=product_name, source_file=multiband_path.name)
            print(f"  -> 写出 {product_name}: {out_path}")
            written.append(out_path)
    return written


def split_existing_outputs(output_dir: str | Path = OUTPUT_DIR, single_band_root: str | Path = SINGLE_BAND_OUTPUT_DIR) -> None:
    """不重新运行 SNAP，只拆分已有的 Biophysical_*.tif。"""
    output_dir = Path(output_dir)
    tif_paths = sorted(output_dir.glob("Biophysical_*.tif"))
    if not tif_paths:
        raise FileNotFoundError(f"没有找到可拆分的 Biophysical_*.tif: {output_dir}")

    print(f"发现 {len(tif_paths)} 个多波段 Biophysical tif，开始拆分单波段产品。")
    for tif_path in tif_paths:
        print(f"\n正在拆分: {tif_path.name}")
        split_primary_bands(tif_path, single_band_root=single_band_root, overwrite=True)


def run_snap_batch(
    input_dir: str | Path = INPUT_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    single_band_root: str | Path = SINGLE_BAND_OUTPUT_DIR,
    split_after_snap: bool = True,
) -> None:
    """批量调用 SNAP 生成多波段 Biophysical tif，并可选拆分单波段产品。"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in sorted(os.listdir(input_dir)):
        if not (file_name.endswith(".SAFE") or file_name.endswith(".zip")):
            continue

        meta = _extract_s2_metadata(file_name)
        input_file_path = input_dir / file_name
        output_file_path = output_dir / _build_output_name(meta)
        temp_xml_path = output_dir / "temp_run.xml"

        print(f"\n正在处理: {file_name}")
        print(f"-> 自动适配传感器: {meta['sensor_type']}")
        print(f"-> 输出多波段结果: {output_file_path.name}")

        tree = _patch_snap_graph(TEMPLATE_XML, input_file_path, output_file_path, meta["sensor_type"])
        tree.write(temp_xml_path, encoding="utf-8", xml_declaration=False)

        cmd = [GPT_PATH, str(temp_xml_path)]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            if output_file_path.exists():
                print(f"【成功】多波段影像已保存: {output_file_path}")
                if split_after_snap:
                    split_primary_bands(output_file_path, single_band_root=single_band_root, overwrite=True)
            else:
                data_folder = output_file_path.with_suffix(".data")
                if data_folder.exists():
                    print(f"【成功但注意】文件生成在数据夹内: {data_folder}")
                else:
                    print("【警告】未检测到输出文件。")
        except subprocess.CalledProcessError as exc:
            print(f"【失败】SNAP 报错信息:\n{exc.stderr}\n")
        finally:
            if temp_xml_path.exists():
                temp_xml_path.unlink()

    print("\n江湾镇 Biophysical 批量处理结束。")


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="Batch run SNAP Biophysical10mOp and split LAI/FVC/FAPAR rasters.")
    parser.add_argument("--input-dir", default=INPUT_DIR, help="Sentinel-2 L2A input directory.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Multiband Biophysical output directory.")
    parser.add_argument("--single-band-dir", default=SINGLE_BAND_OUTPUT_DIR, help="Single-band LAI/FVC/FAPAR output root.")
    parser.add_argument("--split-existing", action="store_true", help="Only split existing Biophysical_*.tif; do not run SNAP.")
    parser.add_argument("--no-split", action="store_true", help="Run SNAP only; do not split single-band outputs.")
    args = parser.parse_args()

    if args.split_existing:
        split_existing_outputs(args.output_dir, args.single_band_dir)
    else:
        run_snap_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            single_band_root=args.single_band_dir,
            split_after_snap=not args.no_split,
        )


if __name__ == "__main__":
    main()
