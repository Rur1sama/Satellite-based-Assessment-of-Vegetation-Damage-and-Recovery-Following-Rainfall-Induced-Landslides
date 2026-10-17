# 仓库介绍
Author：Yukhang Lee

Affiliation：Sun yat-sen Univerisity

运行环境 ： Linux

Python ： 3.9.23

项目背景：算滑坡的。

# 数据下载
降水数据来源于 NASA 的[全球卫星降水计划](https://gpm.nasa.gov/data)；原始数据分辨率为0.1°逐半小时；在GEE批量下载得到。

光学数据来源于欧空局的 [哨兵2号](https://sentinels.copernicus.eu/copernicus/sentinel-2)系列卫星，L2级数据产品。在GEE批量下载得到。

高程数据来源于 JAXA 的[高级陆地观测卫星-1，ALOS](https://www.eorc.jaxa.jp/ALOS/en/index_e.htm)项目卫星；可在GEE上下载得到。

标签数据需要自己制作，请运行[伪标签生成.py](代码/机器学习训练/训练主程序/伪标签生成.py)；标签含义请参照[手动标签数据验证序号表.txt](手动标签数据验证序号表.txt)；需要自己手动开一个叫''Ls_Bg''的字段来存储标签。

# 环境设置
1. python环境配置
```bash
conda create -n slide_dect python=3.9
```

2. 配置库
```bash
pip install requirements.txt
```
3. 输入路径设置 

在Slide_dect根目录创建Input文件夹，再创建Input文件目录以及子目录。
```
+---Input
|   +---GPM_Time_Series(手动创建)
|   |   +---2018（范例,手动创建）
|   |   |       GPM_Daily_20180101.tif
|   |   \---2025（范例,手动创建）
|   |           GPM_Daily_20250101.tif
|   +---光学栅格（手动创建）
|   |   +---SR2_20181007
|   |   |       SR2_20181007T025601_20181007T030202_T49RGH.tif
|   |   |       SR2_20181007T025601_20181007T030202_T49RGH.tif.aux.xml
|   |   |       
|   |   \---SR2_20251121
|   |           SR2_20251121T030151_20251121T030152_T49RGH.tif
|   |           SR2_20251121T030151_20251121T030152_T49RGH.tif.aux.xml
|   |           
|   +---样本面手动标注结果（手动创建）
|   |   +---面18-19（手动创建）
|   |   |       面18-19.cpg
|   |   |       面18-19.dbf
|   |   |       面18-19.prj
|   |   |       面18-19.qmd
|   |   |       面18-19.shp
|   |   |       面18-19.shx
|   |   |       
|   \---高程数据（手动创建）
|           xxx.tif
\
```
4. 在[__init__.py](代码/机器学习训练/训练主程序/utils/__init__.py)中配置模型和数据路径参数。
5. 运行[训练主程序.py](代码/机器学习训练/训练主程序/训练主程序.py)，开始训练。
6. 运行[SHAP_interp.py](代码/机器学习训练/训练主程序/SHAP_interp.py)，开始特征解释归因。
