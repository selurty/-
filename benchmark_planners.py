"""批量评测各航迹规划算法的耗时、路径长度及约束合规性 """

import os
import time
import copy
import csv
import numpy as np
import gc
from datetime import datetime
import environment_models as environment
from flight_constraints import validate_path_constraints
from srd_hs import SRDHSPlanner
from astar import AStar3DPlanner
from grey_wolf_optimizer import GreyWolfPathPlanner
from bidirectional_rrt_star import BidirectionalRRTStar

NUM_TESTS = 50
OUTPUT_DIR = "results"
SAFE_MARGIN = 7.0
WARMUP_ITERATIONS = 3


def calc_path_length(path):
    """使用向量化计算获得航迹总长度 """
    if path is None or len(path) < 2:
        return 0.0
    path = np.array(path)
    dist = np.linalg.norm(path[1:] - path[:-1], axis=1)
    return np.sum(dist)


def normalize_forbidden_zones_for_algorithm(forbidden_zones, max_height):
    algorithm_forbidden_zones = []

    for zone in forbidden_zones or []:
        xmin, xmax, ymin, ymax = zone[:4]
        algorithm_forbidden_zones.append((xmin, xmax, ymin, ymax, 0, max_height))

    return algorithm_forbidden_zones


def forbidden_zones_to_obstacles(forbidden_zones, max_height):
    nofly_obstacles = []

    for zone in forbidden_zones or []:
        xmin, xmax, ymin, ymax = zone[:4]
        nofly_obstacles.append((xmin, ymin, 0, xmax - xmin, ymax - ymin, max_height))

    return nofly_obstacles


def write_csv(file_path, rows):
    if not rows:
        return

    with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def get_all_test_configs():
    configs = [
        {
            "name": "Simple_Env",
            "env_obj": environment.SimpleEnvironment(),
            "start": [100, 900, 190],
            "goal": [900, 100, 60],
            "max_height": 500.0,
        },
        {
            "name": "Complex_Env",
            "env_obj": environment.ComplexStaggeredEnvironment(),
            "start": [100, 900, 190],
            "goal": [950, 100, 60],
            "max_height": 500.0,
        }
    ]

    for cfg in configs:
        cfg["forbidden_zones"] = getattr(cfg["env_obj"], 'forbidden_zones', [])

    return configs


def run_benchmarks():
    """对配置中的环境和算法执行预热、重复测试及结果导出 """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    test_configs = get_all_test_configs()

    print("=" * 75)
    print("UAV Path Planning Automated Batch Test")
    print(f" Total Environments: {len(test_configs)} | Iterations per Env: {NUM_TESTS}")
    print("=" * 75)

    for cfg in test_configs:
        env_name = cfg["name"]
        env_obj = cfg["env_obj"]
        start = cfg["start"]
        goal = cfg["goal"]
        max_height = cfg.get("max_height", 500.0)
        forbidden_zones = cfg.get("forbidden_zones", [])
        algorithm_forbidden_zones = normalize_forbidden_zones_for_algorithm(
            forbidden_zones,
            max_height
        )

        obs_raw = []
        for o in env_obj.obstacles:
            x, y, z = o['pos']
            dx, dy, dz = o['size']
            obs_raw.append((x, y, z, dx, dy, dz))

        baseline_obstacles = list(obs_raw)
        baseline_obstacles.extend(
            forbidden_zones_to_obstacles(algorithm_forbidden_zones, max_height)
        )

        baseline_obs_inflated = [(bx - SAFE_MARGIN, by - SAFE_MARGIN, bz - SAFE_MARGIN,
                                  dx + 2 * SAFE_MARGIN, dy + 2 * SAFE_MARGIN, dz + 2 * SAFE_MARGIN)
                                 for (bx, by, bz, dx, dy, dz) in baseline_obstacles]

        results_data = []
        print(f"\n>>> Current Environment: {env_name}")

        print(f"  [System Warm-up] Ramping up CPU and pre-allocating memory ({WARMUP_ITERATIONS} iterations)", end='',
              flush=True)

        for w_iter in range(WARMUP_ITERATIONS):
            warmup_planners = [
                SRDHSPlanner(copy.deepcopy(start), copy.deepcopy(goal), copy.deepcopy(obs_raw), margin=SAFE_MARGIN,
                       max_height=max_height, forbidden_zones=copy.deepcopy(algorithm_forbidden_zones)),
                BidirectionalRRTStar(copy.deepcopy(start), copy.deepcopy(goal), copy.deepcopy(baseline_obs_inflated), [0, 1000],
                          max_iter=1500),
                AStar3DPlanner(copy.deepcopy(start), copy.deepcopy(goal), copy.deepcopy(baseline_obstacles),
                        margin=SAFE_MARGIN, resolution=25.0),
                GreyWolfPathPlanner(copy.deepcopy(start), copy.deepcopy(goal), copy.deepcopy(baseline_obstacles),
                               margin=SAFE_MARGIN)
            ]

            for wp in warmup_planners:
                try:
                    if hasattr(wp, 'planning'):
                        wp.planning()
                    elif hasattr(wp, 'find_path'):
                        wp.find_path()
                except Exception:
                    pass
            print(".", end='', flush=True)

        gc.collect()
        time.sleep(0.5)
        print(" Done.")

        for i in range(1, NUM_TESTS + 1):

            planners = [
                ("SRD-HS (Proposed)",
                 SRDHSPlanner(copy.deepcopy(start), copy.deepcopy(goal), copy.deepcopy(obs_raw), margin=SAFE_MARGIN,
                        max_height=max_height, forbidden_zones=copy.deepcopy(algorithm_forbidden_zones))),
                ("Bi-RRT*",
                 BidirectionalRRTStar(copy.deepcopy(start), copy.deepcopy(goal), copy.deepcopy(baseline_obs_inflated), [0, 1000],
                           max_iter=1500)),
                ("A*", AStar3DPlanner(copy.deepcopy(start), copy.deepcopy(goal), copy.deepcopy(baseline_obstacles),
                               margin=SAFE_MARGIN, resolution=25.0)),
                ("GWO", GreyWolfPathPlanner(copy.deepcopy(start), copy.deepcopy(goal), copy.deepcopy(baseline_obstacles),
                                       margin=SAFE_MARGIN))
            ]

            for name, planner in planners:
                path = None
                num_nodes = 0
                t_start = time.time()
                found = False
                constraint_ok = False
                violation_count = 0
                success = False
                path_max_z = 0.0

                print(f"\r  Iteration {i}/{NUM_TESTS} [Evaluating: {name:<20}]...", end='', flush=True)

                try:
                    if "SRD-HS" in name:
                        res = planner.planning()
                        if res:
                            wp, _, path = res
                            num_nodes = len(wp)
                    elif "A*" in name:
                        path = planner.find_path()
                        if path is not None:
                            num_nodes = len(path)
                    elif "GWO" in name:
                        path = planner.planning()
                        if path is not None:
                            num_nodes = len(path)
                    elif "Bi-RRT*" in name:
                        path = planner.planning()
                        if path is not None:
                            num_nodes = len(planner.trees[0]) + len(planner.trees[1])

                    duration = time.time() - t_start

                    if path is not None and len(path) > 0:
                        path = np.array(path)
                        path_len = calc_path_length(path)
                        path_max_z = float(np.max(path[:, 2]))
                        found = True
                        constraint_ok, violation_count = validate_path_constraints(
                            path, max_height=max_height, forbidden_zones=algorithm_forbidden_zones
                        )
                        success = constraint_ok
                    else:
                        path_len = 0.0
                        duration = time.time() - t_start

                except Exception as e:
                    print(f"\n  [!] {name} failed on Iteration {i}. Error: {e}")
                    duration = time.time() - t_start
                    path_len = 0.0
                    num_nodes = 0
                    found = False
                    constraint_ok = False
                    violation_count = 0
                    success = False
                    path_max_z = 0.0

                results_data.append({
                    "Iteration": i,
                    "Algorithm": name,
                    "Time_s": round(duration, 4),
                    "Length_m": round(path_len, 2),
                    "Nodes": num_nodes,
                    "Path_Max_Z": round(path_max_z, 2),
                    "PathFound": int(found),
                    "Constraint_OK": int(constraint_ok),
                    "Constraint_Violations": violation_count,
                    "Max_Height": max_height,
                    "Success": int(success)
                })

        print(f"\r  Iteration {NUM_TESTS}/{NUM_TESTS} completed." + " " * 30)

        file_path = os.path.join(OUTPUT_DIR, f"results_{env_name}_{datetime.now().strftime('%m%d_%H%M')}.csv")
        write_csv(file_path, results_data)

        print(f"\n  {env_name} Completed. Summary:")
        print(
            f"  {'Algorithm':<20} | {'SR':<8} | {'CR':<8} | {'Avg Time(s)':<12} | {'Avg Len(m)':<12} | {'Avg Nodes':<10} | {'Avg MaxZ':<9}")

        algorithms = []
        for row in results_data:
            if row["Algorithm"] not in algorithms:
                algorithms.append(row["Algorithm"])

        for alg in algorithms:
            alg_rows = [row for row in results_data if row["Algorithm"] == alg]
            sr = mean(row["Success"] for row in alg_rows) * 100
            cr = mean(row["Constraint_OK"] for row in alg_rows) * 100
            success_rows = [row for row in alg_rows if row["Success"] == 1]

            if success_rows:
                avg_t = mean(row["Time_s"] for row in success_rows)
                avg_l = mean(row["Length_m"] for row in success_rows)
                avg_n = mean(row["Nodes"] for row in success_rows)
                avg_z = mean(row["Path_Max_Z"] for row in success_rows)
                print(
                    f"  {alg:<20} | {sr:>6.1f}% | {cr:>6.1f}% | {avg_t:>10.4f}   | "
                    f"{avg_l:>10.2f}   | {avg_n:>9.1f} | {avg_z:>8.1f}"
                )
            else:
                print(
                    f"  {alg:<20} | {sr:>6.1f}% | {cr:>6.1f}% | {'N/A':>10}   | "
                    f"{'N/A':>10}   | {'N/A':>9} | {'N/A':>8}"
                )

    print("\n" + "=" * 75)
    print("All tests finished. Results are in the 'results' folder.")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmarks()
