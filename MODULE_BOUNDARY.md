# Coordinate-To-Parcel Module Boundary

本文是给后续 AI/开发者看的模块边界说明。目标是让 coordinate-to-parcel 可以持续优化，但不破坏 file-browser 端到端流程。

## 模块职责

coordinate-to-parcel 只负责把 EPSG:27700 点转换成每个 `oachargeid` 的 0 到 1 个最终地块多边形。

它不负责 geocoding，不负责解释原始用户 CSV/XLSX，也不负责 council 自动识别。

## 稳定输入契约

端到端流程给 coordinate-to-parcel 的稳定输入来自：

```bash
python -m spatial_pipeline.adapters geocoding-points
```

稳定输入图层为：

- layer: `stable_geocoding_points`
- required field: `oachargeid`
- optional fields: `oachargeid_sub`, `x_27700`, `y_27700`
- compatibility fields: `api_x_27700`, `api_y_27700`
- crs: `EPSG:27700`
- geometry: Point

coordinate-to-parcel 内部不要依赖 geocoding 的其它临时字段。需要新信号时，应作为可选增强输入处理，不能成为端到端必需条件。

## 稳定输出契约

端到端流程只认以下输出语义：

- 每个 `oachargeid` 输出 0 到 1 个 polygon。
- 有 polygon 时，必须能被 adapter 标准化为：
  - layer: `stable_oachargeid_polygons`
  - required field: `oachargeid`
  - geometry: Polygon 或 MultiPolygon
  - crs: `EPSG:27700`
- 没有 polygon 是合法结果，不应让整批任务失败。

当前标准化命令是：

```bash
python -m spatial_pipeline.adapters selected-polygons
```

## Council 无关性

除输入端和输出端外，中间步骤不应写 council 分支。所有 council/area 相关资源都应来自 `spatial_pipeline.area_profile.AreaProfile`：

- WFS raw GPKG 和 layer
- council reference land GPKG 和 layer
- UPRN GPKG 和 layer
- model bundle 路径
- native merge layer

不要在 `point_surrounding.py`、`wfs_merge_native/run_pipeline.py` 或 `generate_auto_polygon.py` 里硬编码某个 council。

## 修改守则

- 可以优化 WFS 覆盖检测、下载策略、merge 模型、候选 polygon 选择、fallback 策略。
- 不要让 coordinate-to-parcel 读取原始用户 CSV/XLSX。
- 不要依赖 geocoding 的内部诊断字段作为必需字段。
- 输出字段可以增删，但 `oachargeid -> 0/1 polygon` 的语义必须保留。
- 如果最终输出层名或字段变化，必须同步 `spatial_pipeline.adapters selected-polygons`，不要让 file-browser 跟着改。
- 新增 area/council 支持时，优先修 `AreaProfile` resource resolution，不要在算法脚本里加 council if/else。

## 提交前检查

至少运行：

```bash
PYTHONPATH=/env/code:/env/code/geocoding python3 -m unittest discover -s tests
```

如果改了 GPKG 输入/输出层或字段，还要用 `/env/venv/textual/bin/python3.11` 跑一次 `spatial_pipeline.adapters` 的 geocoding-points 和 selected-polygons smoke test。
