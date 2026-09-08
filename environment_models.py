"""无人机三维航迹规划的环境模型与碰撞检测工具 """

import numpy as np

try:
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ModuleNotFoundError:
    matplotlib = None
    plt = None
    Poly3DCollection = None
import math

MAP_SIZE = [1000, 1000, 500]
FIXED_SEED = 2024

STYLE = {
    'face_color': '#708090',   # 普通建筑填充
    'edge_color': '#2F4F4F',   # 边框颜色
    'sphere_color': '#2C3E50',  # 空中障碍物颜色
    'nofly_color': '#FF0000',  # 禁飞区颜色
    'nofly_edge': '#B30000',   # 禁飞区边界颜色
    'axis_edge': '#808080',    # 坐标轴
    'grid_color': '#D3D3D3',   # 网格
    'alpha_block': 0.4,        # 建筑透明度
    'alpha_sphere': 0.4,       # 球体透明度
    'alpha_nofly': 0.28,       # 禁飞区透明度
    'nofly_z_offset': 2.0,     # 禁飞区略高于地面，避免重叠闪烁
}


def is_point_in_obstacle(point, obstacles, margin=2.0):
    """检查点是否在障碍物内 """
    px, py, pz = point
    for b in obstacles:
        if len(b) >= 6:
            bx, by, b_base_z, bdx, bdy, bdz = b[:6]
            if (bx - margin <= px <= bx + bdx + margin) and \
                    (by - margin <= py <= by + bdy + margin):
                if b_base_z - margin <= pz <= (b_base_z + bdz + margin):
                    return True
        elif len(b) == 4:
            sx, sy, sz, r = b
            dist = math.sqrt((px - sx) ** 2 + (py - sy) ** 2 + (pz - sz) ** 2)
            if dist <= (r + margin):
                return True
    return False


def check_line_collision(p1, p2, obstacles):
    """检查线段是否与障碍物碰撞 """
    safe_margin = 2.0
    step_size = 10.0

    x1, y1, z1 = p1
    x2, y2, z2 = p2

    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    if dist < step_size:
        return is_point_in_obstacle(p2, obstacles, safe_margin)

    steps = int(dist / step_size) + 1
    for i in range(steps + 1):
        t = i / steps
        cx = x1 + (x2 - x1) * t
        cy = y1 + (y2 - y1) * t
        cz = z1 + (z2 - z1) * t
        if is_point_in_obstacle((cx, cy, cz), obstacles, safe_margin):
            return True

    return False


class SimpleEnvironment:
    """用于基础算法验证的稀疏障碍环境。"""

    def __init__(self):
        self.obstacles = []
        self.spheres = []

        self.forbidden_zones = [
            (420, 520, 450, 550),  # 中心禁飞区
            (580, 700, 250, 350),  # 右下前置禁飞区
            (250, 350, 680, 800)   # 左上后置禁飞区
        ]
        self.generate_fixed_from_table()

    def generate_fixed_from_table(self):

        # --- 1. 高层建筑 ---
        h_size = (120, 120, 350)
        h_coords = [
            (298, 588, 0), (303, 723, 0), (582, 151, 0),
            (586, 872, 0), (726, 301, 0), (727, 157, 0)
        ]
        for pos in h_coords:
            self.obstacles.append({'pos': pos, 'size': h_size})

        # --- 2. 中层建筑  ---
        m_size = (100, 80, 180)
        m_coords = [
            (16, 900, 0), (152, 310, 0), (157, 467, 0), (295, 911, 0),
            (312, 332, 0), (453, 610, 0), (739, 451, 0), (743, 622, 0),
            (870, 601, 0), (885, 188, 0)
        ]
        for pos in m_coords:
            self.obstacles.append({'pos': pos, 'size': m_size})

        # --- 3. 底层建筑 ---
        l_size = (60, 60, 80)
        l_coords = [
            (25, 332, 0), (47, 438, 0), (60, 641, 0), (196, 607, 0),
            (197, 33, 0), (201, 758, 0), (356, 466, 0), (451, 450, 0),
            (469, 926, 0), (491, 732, 0), (496, 314, 0), (645, 18, 0),
            (723, 785, 0), (773, 931, 0), (876, 454, 0), (898, 356, 0),
            (913, 49, 0), (924, 884, 0)
        ]
        for pos in l_coords:
            self.obstacles.append({'pos': pos, 'size': l_size})

        # --- 4. 空中障碍物  ---
        radius = 50
        aerial_raw_coords = [
            (80, 223, 185), (350, 74, 312), (492, 209, 254),
            (652, 351, 277), (928, 792, 183)
        ]
        for (x, y, z) in aerial_raw_coords:
            center = (x + radius, y + radius, z + radius)
            self.spheres.append({'pos': center, 'radius': radius})


class ComplexStaggeredEnvironment:
    """用于综合性能验证的交错密集障碍环境 """

    def __init__(self):
        self.obstacles = []
        self.spheres = []
        self.forbidden_zones = [
            (380, 650, 380, 640),
            (100, 250, 700, 850)
        ]
        self.generate_from_table()

    def generate_from_table(self):
        # 1. 高层建筑
        high_size = (140, 140, 380)
        high_coords = [(223, 469, 0), (339, 686, 0), (369, 0, 0), (420, 527, 0), (488, 288, 0), (688, 461, 0)]
        for pos in high_coords:
            self.obstacles.append({'pos': pos, 'size': high_size})

        # 2. 中层建筑
        mid_size = (110, 100, 220)
        mid_coords = [
            (48, 467, 0), (54, 616, 0), (119, 261, 0), (177, 621, 0), (183, 133, 0), (218, 743, 0),
            (258, 286, 0), (329, 848, 0), (358, 162, 0), (497, 875, 0), (519, 682, 0), (519, 177, 0),
            (538, 65, 0), (627, 808, 0), (638, 688, 0), (653, 166, 0), (701, 347, 0), (770, 680, 0),
            (790, 228, 0), (823, 352, 0), (851, 517, 0)
        ]
        for pos in mid_coords:
            self.obstacles.append({'pos': pos, 'size': mid_size})

        # 3. 底层建筑
        low_size = (80, 80, 100)
        low_coords = [
            (14, 900, 0), (18, 782, 0), (28, 325, 0), (30, 67, 0), (63, 157, 0), (123, 751, 0),
            (125, 375, 0), (130, 13, 0), (137, 907, 0), (233, 895, 0), (249, 25, 0), (373, 334, 0),
            (391, 421, 0), (521, 787, 0), (566, 454, 0), (585, 576, 0), (677, 27, 0), (759, 805, 0),
            (766, 80, 0), (769, 909, 0), (851, 60, 0), (860, 870, 0), (895, 735, 0), (906, 640, 0),
            (907, 166, 0)
        ]
        for pos in low_coords:
            self.obstacles.append({'pos': pos, 'size': low_size})

        # 4. 空中障碍物
        radius = 45
        aerial_coords = [
            (269, 750, 319), (148, 893, 273), (351, 723, 440), (226, 404, 292), (523, 203, 383),
            (651, 898, 291), (861, 729, 323), (818, 136, 344), (638, 58, 385), (119, 59, 305)
        ]
        for (x, y, z) in aerial_coords:
            center = (x + radius, y + radius, z + radius)
            self.spheres.append({'pos': center, 'radius': radius, 'type': 'aerial'})


# ==========================================
# 2. 绘图辅助函数
# ==========================================
def plot_scene(ax, env_obj):
    if Poly3DCollection is None:
        raise RuntimeError("matplotlib is required for plotting scenes.")

    # 1. 绘制实体建筑物
    for o in env_obj.obstacles:
        x, y, z = o['pos']
        dx, dy, dz = o['size']
        V = [[x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z],
             [x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [x, y + dy, z + dz]]
        faces = [[V[0], V[1], V[5], V[4]], [V[2], V[3], V[7], V[6]],
                 [V[0], V[3], V[7], V[4]], [V[1], V[2], V[6], V[5]],
                 [V[4], V[5], V[6], V[7]], [V[0], V[1], V[2], V[3]]]
        poly = Poly3DCollection(faces, alpha=STYLE['alpha_block'], linewidths=0.6, edgecolors=STYLE['edge_color'])
        poly.set_facecolor(STYLE['face_color'])
        ax.add_collection3d(poly)

    # 2. 绘制空中球体障碍
    for s in env_obj.spheres:
        x_c, y_c, z_c = s['pos']
        r = s['radius']
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        x = r * np.outer(np.cos(u), np.sin(v)) + x_c
        y = r * np.outer(np.sin(u), np.sin(v)) + y_c
        z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z_c
        ax.plot_surface(x, y, z, color=STYLE['sphere_color'], alpha=STYLE['alpha_sphere'], edgecolor='none')

    # 3. 绘制禁飞区：地面红色半透明区域
    if hasattr(env_obj, 'forbidden_zones'):
        z0 = STYLE['nofly_z_offset']
        for fz in env_obj.forbidden_zones:
            xmin, xmax, ymin, ymax = fz[:4]

            # 地面投影面
            verts = [[
                [xmin, ymin, z0],
                [xmax, ymin, z0],
                [xmax, ymax, z0],
                [xmin, ymax, z0]
            ]]

            poly = Poly3DCollection(
                verts,
                alpha=STYLE['alpha_nofly'],
                linewidths=1.8,
                edgecolors=STYLE['nofly_edge']
            )
            poly.set_facecolor(STYLE['nofly_color'])
            ax.add_collection3d(poly)

            # 红色虚线边界
            xs = [xmin, xmax, xmax, xmin, xmin]
            ys = [ymin, ymin, ymax, ymax, ymin]
            zs = [z0 + 0.5] * 5
            ax.plot(xs, ys, zs,
                    color=STYLE['nofly_edge'],
                    linewidth=2.0,
                    linestyle='--')


def configure_axis(ax):
    ax.set_xlim(0, MAP_SIZE[0])
    ax.set_ylim(0, MAP_SIZE[1])
    ax.set_zlim(0, MAP_SIZE[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect((MAP_SIZE[0], MAP_SIZE[1], MAP_SIZE[2]))


# ==========================================
# 3. 主函数执行块
# ==========================================
if __name__ == "__main__":

    fig = plt.figure(figsize=(16, 8))

    # === 绘制简单环境 ===
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Simple Environment with No-Fly Zone", fontsize=14)
    env_simple = SimpleEnvironment()
    plot_scene(ax1, env_simple)
    configure_axis(ax1)
    ax1.view_init(elev=35, azim=-45)

    # === 绘制复杂环境 ===
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Complex Environment with No-Fly Zones", fontsize=14)
    env_complex = ComplexStaggeredEnvironment()
    plot_scene(ax2, env_complex)
    configure_axis(ax2)
    ax2.view_init(elev=35, azim=-45)

    plt.tight_layout()
    plt.show()

