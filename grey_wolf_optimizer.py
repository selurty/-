"""用于三维无人机航迹规划的灰狼优化算法 """

import numpy as np


class GreyWolfPathPlanner:
    """将灰狼优化算法应用于三维无人机中间航迹点寻优 """

    def __init__(self, start, goal, obstacle_list, margin=5.0, agents_no=40, max_iter=80, n_waypoints=3):
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.obstacles = obstacle_list
        self.margin = margin

        # GWO 超参数
        self.agents_no = agents_no
        self.max_iter = max_iter
        self.n_waypoints = n_waypoints  # 用 N 个中间控制点来表示一条航迹
        self.dim = n_waypoints * 3  # 每个候选解包含全部中间航迹点的三维坐标

        # 搜索空间边界 (x:0~1000, y:0~1000, z:0~500)
        self.lb = np.array([0, 0, 0] * n_waypoints)
        self.ub = np.array([1000, 1000, 500] * n_waypoints)

    def get_intersection_info(self, p1, p2, obs):
        """使用射线与轴对齐包围盒求交判断航段碰撞 """
        bx, by, bz, dx, dy, dz = obs
        m = self.margin
        b_min = np.array([bx - m, by - m, bz - m])
        b_max = np.array([bx + dx + m, by + dy + m, bz + dz + m])

        d = p2 - p1
        t_near, t_far = -1e9, 1e9

        for i in range(3):
            if abs(d[i]) < 1e-9:
                if p1[i] < b_min[i] or p1[i] > b_max[i]: return None
            else:
                t1 = (b_min[i] - p1[i]) / d[i]
                t2 = (b_max[i] - p1[i]) / d[i]
                if t1 > t2: t1, t2 = t2, t1
                if t1 > t_near: t_near = t1
                t_far = min(t_far, t2)

        if t_near <= t_far and 0 <= t_near <= 1.0:
            return True
        return None

    def check_collision(self, p1, p2):
        for obs in self.obstacles:
            if self.get_intersection_info(p1, p2, obs):
                return True
        return False

    def fobj(self, position):
        """以航迹长度为目标，并对碰撞航段施加高额惩罚 """
        pts = position.reshape(self.n_waypoints, 3)
        path = np.vstack((self.start, pts, self.goal))

        cost = 0.0
        collision_penalty = 1e6  # 严厉惩罚碰撞

        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            cost += np.linalg.norm(p2 - p1)
            if self.check_collision(p1, p2):
                cost += collision_penalty
        return cost

    def planning(self):
        # 初始化三个层级的最优候选解及其适应度
        Alpha_pos = np.zeros(self.dim)
        Alpha_score = float('inf')

        Beta_pos = np.zeros(self.dim)
        Beta_score = float('inf')

        Delta_pos = np.zeros(self.dim)
        Delta_score = float('inf')

        # 初始化种群位置
        Positions = np.random.uniform(0, 1, (self.agents_no, self.dim)) * (self.ub - self.lb) + self.lb

        # 主循环
        for l in range(self.max_iter):
            for i in range(self.agents_no):
                # 边界处理 (越界拉回)
                Positions[i, :] = np.clip(Positions[i, :], self.lb, self.ub)

                # 计算适应度
                fitness = self.fobj(Positions[i, :])

                # 更新 Alpha, Beta, Delta
                if fitness < Alpha_score:
                    Alpha_score = fitness
                    Alpha_pos = Positions[i, :].copy()
                elif fitness < Beta_score:
                    Beta_score = fitness
                    Beta_pos = Positions[i, :].copy()
                elif fitness < Delta_score:
                    Delta_score = fitness
                    Delta_pos = Positions[i, :].copy()

            # 线性衰减因子 a (从 2 到 0)
            a = 2 - l * (2 / self.max_iter)

            # 根据三个最优候选解更新种群位置
            for i in range(self.agents_no):
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                    X1 = Alpha_pos[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                    X2 = Beta_pos[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                    X3 = Delta_pos[j] - A3 * D_delta

                    Positions[i, j] = (X1 + X2 + X3) / 3

        # 如果最终得分仍带有惩罚项，说明未能找到无碰路径
        if Alpha_score >= 1e5:
            return None

        best_pts = Alpha_pos.reshape(self.n_waypoints, 3)
        final_path = np.vstack((self.start, best_pts, self.goal))
        return final_path
