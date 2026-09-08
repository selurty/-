"""对比多种三维航迹规划算法并绘制规划结果 """
import numpy as np
import time
import environment_models as environment
from flight_constraints import validate_path_constraints
from srd_hs import SRDHSPlanner
from astar import AStar3DPlanner
from grey_wolf_optimizer import GreyWolfPathPlanner
from bidirectional_rrt_star import BidirectionalRRTStar

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.lines import Line2D
    PLOTTING_AVAILABLE = True
except ModuleNotFoundError:
    matplotlib = None
    plt = None
    Poly3DCollection = None
    Line2D = None
    PLOTTING_AVAILABLE = False

FONT_SIZE = 12
if PLOTTING_AVAILABLE:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams['font.size'] = FONT_SIZE
    plt.rcParams['axes.labelsize'] = FONT_SIZE
    plt.rcParams['xtick.labelsize'] = FONT_SIZE
    plt.rcParams['ytick.labelsize'] = FONT_SIZE
    plt.rcParams['legend.fontsize'] = FONT_SIZE
    plt.rcParams['axes.titlesize'] = FONT_SIZE

COLORS = {
    'obs_face': '#708090',
    'obs_edge': '#2F4F4F',
    'start': '#2ca02c',
    'goal': '#d62728',
    'ours': '#D62728',
    'astar': '#FF7F0E',
    'gwo': '#9467bd',
    'rrt': '#1F77B4',
    'nofly': '#FF0000'
}


def calc_path_length(path):
    """计算三维折线航迹的累计欧氏长度 """
    if path is None or len(path) < 2:
        return 0.0
    length = 0.0
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i + 1])
        length += np.linalg.norm(p2 - p1)
    return length


def draw_cuboid(ax, x, y, z, dx, dy, dz, color_override=None):

    vertices = [
        [x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z],
        [x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [x, y + dy, z + dz]
    ]
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[2], vertices[3]]
    ]
    face_c = color_override if color_override else COLORS['obs_face']
    edge_c = 'black' if color_override else COLORS['obs_edge']
    poly = Poly3DCollection(faces, alpha=0.4, linewidths=0.6, edgecolors=edge_c)
    poly.set_facecolor(face_c)
    ax.add_collection3d(poly)


def normalize_forbidden_zones_for_algorithm(forbidden_zones, max_height):
    """为算法补全禁飞区的垂直边界 """
    algorithm_forbidden_zones = []

    for zone in forbidden_zones or []:
        xmin, xmax, ymin, ymax = zone[:4]
        algorithm_forbidden_zones.append((xmin, xmax, ymin, ymax, 0, max_height))

    return algorithm_forbidden_zones


def forbidden_zones_to_obstacles(forbidden_zones, max_height):
    """将二维禁飞区转换为贯穿限高范围的长方体障碍物 """
    nofly_obstacles = []

    for zone in forbidden_zones or []:
        xmin, xmax, ymin, ymax = zone[:4]

        bx = xmin
        by = ymin
        bz = 0
        bdx = xmax - xmin
        bdy = ymax - ymin
        bdz = max_height

        nofly_obstacles.append((bx, by, bz, bdx, bdy, bdz))

    return nofly_obstacles


def draw_forbidden_zone(ax, zone, max_height):

    xmin, xmax, ymin, ymax = zone[:4]

    z0 = 2.0

    verts = [[
        [xmin, ymin, z0],
        [xmax, ymin, z0],
        [xmax, ymax, z0],
        [xmin, ymax, z0]
    ]]

    poly = Poly3DCollection(
        verts,
        alpha=0.28,
        linewidths=1.5,
        edgecolors=COLORS['nofly']
    )
    poly.set_facecolor(COLORS['nofly'])
    ax.add_collection3d(poly)

    xs = [xmin, xmax, xmax, xmin, xmin]
    ys = [ymin, ymin, ymax, ymax, ymin]
    zs = [z0 + 0.5] * 5

    ax.plot(
        xs, ys, zs,
        color=COLORS['nofly'],
        linewidth=2.0,
        linestyle='--'
    )


def setup_environment_plot(ax, obstacles, start, goal, forbidden_zones=None, max_height=500.0):
    for obs in obstacles:
        if len(obs) >= 6:
            draw_cuboid(ax, *obs[:6])

    for zone in forbidden_zones or []:
        draw_forbidden_zone(ax, zone, max_height)

    ax.scatter(start[0], start[1], start[2], c=COLORS['start'], marker='o', s=150,
               depthshade=False, edgecolors='white', linewidth=1.5, label='Start', zorder=100)
    ax.scatter(goal[0], goal[1], goal[2], c=COLORS['goal'], marker='*', s=250,
               depthshade=False, edgecolors='white', linewidth=1.5, label='Goal', zorder=100)

    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(0, 500)
    ax.set_box_aspect((1, 1, 0.8))
    ax.set_xlabel('X/m', labelpad=10)
    ax.set_ylabel('Y/m', labelpad=10)
    ax.set_zlabel('Z/m', labelpad=10)
    ax.set_zticks(np.arange(0, 501, 100))
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    ax.grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.3)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


def main():
    """选择测试环境，执行算法对比并显示三维规划结果 """
    print("=" * 60)
    print("3D UAV Path Planning Evaluation")
    print("=" * 60)
    print("1: Simple Environment")
    print("2: Complex Staggered Environment")
    print("=" * 60)

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == '2':
        env_obj = environment.ComplexStaggeredEnvironment()
        start_pos = [100, 900, 190]
        goal_pos = [950, 100, 60]
    else:
        env_obj = environment.SimpleEnvironment()
        start_pos = [100, 900, 190]
        goal_pos = [900, 100, 60]

    max_height = getattr(env_obj, 'max_height', 500.0)

    forbidden_zones = getattr(env_obj, 'forbidden_zones', [])

    algorithm_forbidden_zones = normalize_forbidden_zones_for_algorithm(
        forbidden_zones,
        max_height
    )

    all_obstacles = []
    if hasattr(env_obj, 'obstacles'):
        for o in env_obj.obstacles:
            x, y, z = o['pos']
            dx, dy, dz = o['size']
            all_obstacles.append((x, y, z, dx, dy, dz))

    print(f"\n>>> Scenario Loaded. Obstacles: {len(all_obstacles)}")
    print(f">>> Start: {start_pos}, Goal: {goal_pos}")
    if forbidden_zones:
        print(f">>> Flight constraints: max height {max_height} m, no-fly zones {len(forbidden_zones)}")
        print(">>> No-fly zone logic: ground footprint extends upward to max height for planning")

    safe_margin = 7.0

    baseline_obstacles = list(all_obstacles)
    baseline_obstacles.extend(
        forbidden_zones_to_obstacles(algorithm_forbidden_zones, max_height)
    )

    baseline_obs_inflated = [(bx - safe_margin, by - safe_margin, bz - safe_margin,
                              dx + 2 * safe_margin, dy + 2 * safe_margin, dz + 2 * safe_margin)
                             for (bx, by, bz, dx, dy, dz) in baseline_obstacles]

    planners = [
        ("SRD-HS (Proposed)",
         SRDHSPlanner(start_pos, goal_pos, all_obstacles, margin=safe_margin,
                max_height=max_height, forbidden_zones=algorithm_forbidden_zones),
         COLORS['ours'], '-', 2.5),
        ("Bi-RRT*",
         BidirectionalRRTStar(start_pos, goal_pos, baseline_obs_inflated, [0, 1000], max_iter=1500),
         COLORS['rrt'], '-.', 2.0),
        ("A*",
         AStar3DPlanner(start_pos, goal_pos, baseline_obstacles, margin=safe_margin, resolution=25.0),
         COLORS['astar'], '--', 2.0),
        ("GWO",
         GreyWolfPathPlanner(start_pos, goal_pos, baseline_obstacles, margin=safe_margin),
         COLORS['gwo'], ':', 2.0)
    ]

    results = []

    print("\n" + "=" * 95)
    print(f"{'Algorithm':<28} | {'Time(s)':<12} | {'Length(m)':<12} | {'Nodes/Status':<15} | {'Constraint':<10}")
    print("-" * 95)

    for name, planner, color, linestyle, linewidth in planners:
        path = None
        tree_nodes_or_status = "0"
        constraint_status = "OK"
        t_start = time.time()
        try:
            if "SRD-HS" in name:
                res = planner.planning()
                if res:
                    wp, _, path = res
                    tree_nodes_or_status = str(len(wp))
            elif "A*" in name:
                path = planner.find_path()
                if path is not None:
                    tree_nodes_or_status = str(len(path))
            elif "GWO" in name:
                path = planner.planning()
                if path is not None:
                    tree_nodes_or_status = str(len(path))
            elif "Bi-RRT*" in name:
                path = planner.planning()
                if path is not None:
                    tree_nodes_or_status = str(len(planner.trees[0]) + len(planner.trees[1]))
        except Exception as e:
            print(f"[{name}] Error: {e}")
            path = None

        duration = time.time() - t_start
        if path is not None and len(path) > 0:
            path = np.array(path)
            path_len = calc_path_length(path)

            constraint_ok, violation_count = validate_path_constraints(
                path,
                max_height=max_height,
                forbidden_zones=algorithm_forbidden_zones
            )
            constraint_status = "OK" if constraint_ok else f"Viol.{violation_count}"
        else:
            path_len = float('inf')
            tree_nodes_or_status = "Failed"
            constraint_status = "Failed"

        print(f"{name:<28} | {duration:<12.4f} | {path_len:<12.2f} | {tree_nodes_or_status:<15} | {constraint_status:<10}")

        if path is not None:
            results.append({
                "name": name, "path": path, "color": color,
                "style": linestyle, "width": linewidth,
                "constraint_status": constraint_status
            })

    print("=" * 95)

    if not PLOTTING_AVAILABLE:
        print("\n>>> matplotlib is not installed; skipped 3D trajectory rendering.")
        return

    print("\n>>> Rendering 3D Trajectories and Shadows...")
    fig = plt.figure(figsize=(14, 10), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    setup_environment_plot(ax, all_obstacles, start_pos, goal_pos,
                           forbidden_zones=forbidden_zones, max_height=max_height)

    legend_proxies = []
    for res in results:
        path = res["path"]
        ax.plot(path[:, 0], path[:, 1], path[:, 2],
                c=res["color"], linestyle=res["style"], linewidth=res["width"],
                zorder=200, alpha=0.95)
        ax.plot(path[:, 0], path[:, 1], np.zeros_like(path[:, 2]),
                c=res["color"], linestyle=res["style"], linewidth=res["width"] * 0.6,
                alpha=0.25, zorder=0)
        label = res["name"] if res["constraint_status"] == "OK" else f"{res['name']} ({res['constraint_status']})"
        legend_proxies.append(Line2D([0], [0], color=res["color"], linewidth=res["width"],
                                     linestyle=res["style"], label=label))

    ax.legend(handles=legend_proxies, loc='upper left', frameon=True, edgecolor='black', framealpha=0.9)
    plt.show()


if __name__ == "__main__":
    main()
