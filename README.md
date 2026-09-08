# 一种面向城市环境的无人机射线探测启发式三维航迹规划方法

本代码包仅包含 SRD-HS 本文算法、三种对比算法，以及生成三维航迹图、50 次独立运行对比曲线和定量汇总表所需的代码。

## 环境要求

- Python 3.9 或更高版本
- Windows、Linux 或 macOS
- 三维图形显示需要可用的 Matplotlib 图形后端

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 代码范围

### 本文算法

| 文件 | 说明 |
| --- | --- |
| `srd_hs.py` | 本文提出的 SRD-HS 三维航迹规划算法 |

### 对比算法

| 文件 | 说明 |
| --- | --- |
| `astar.py` | 三维 A* 航迹规划算法 |
| `grey_wolf_optimizer.py` | GWO 灰狼优化航迹规划算法 |
| `bidirectional_rrt_star.py` | Bi-RRT* 双向快速扩展随机树算法 |
| `rrt_star.py` | Bi-RRT* 使用的父节点优选与邻域重连逻辑 |
| `rrt.py` | RRT 基础节点、采样、扩展和碰撞检测逻辑 |

### 公共模块

| 文件 | 说明 |
| --- | --- |
| `environment_models.py` | 简单与复杂城市环境、障碍物模型及碰撞检测 |
| `flight_constraints.py` | 限高、禁飞区及航段合规性校验 |

### 结果生成代码

| 文件 | 对应结果 |
| --- | --- |
| `path_planning_comparison.py` | SRD-HS、Bi-RRT*、A*、GWO 的三维航迹对比图 |
| `benchmark_planners.py` | 简单和复杂城市环境下各算法 50 次独立运行数据 |
| `visualize_benchmark_results.py` | 根据 CSV 生成耗时与航迹长度对比曲线 |
| `summarize_benchmark_results.py` | 根据 CSV 生成 Time、Length、Nodes 定量汇总表 |

## 结果生成顺序

1. 运行三维航迹对比：

```powershell
python path_planning_comparison.py
```

输入 `1` 生成简单城市环境结果，输入 `2` 生成复杂城市环境结果。

2. 执行两个环境下各 50 次独立运行：

```powershell
python benchmark_planners.py
```

原始数据将写入 `results/` 目录下的 CSV 文件。

3. 生成 50 次独立运行对比曲线：

```powershell
python visualize_benchmark_results.py
```

4. 生成定量实验汇总表：

```powershell
python summarize_benchmark_results.py
```

## 数据约定

- 三维点格式为 `(x, y, z)`，距离单位为米。
- 长方体障碍物格式为 `(x, y, z, dx, dy, dz)`。
- 禁飞区至少包含 `(xmin, xmax, ymin, ymax)`，也可追加垂直范围。
- 未搜索到合规航迹时，算法返回 `None`，批量评测将该次运行记为失败。

`results/` 为运行时生成目录，不属于源码文件。
