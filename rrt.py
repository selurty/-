"""三维快速扩展随机树（RRT）基础实现 """

import math
import random
import numpy as np
import environment_models as environment


class Node:
    """表示随机树中的三维节点 """

    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.cost = 0.0


class RRT:
    """在连续三维空间中执行采样、扩展和碰撞检测 """

    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=80.0,
                 goal_sample_rate=0.2,
                 max_iter=3000):
        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])
        self.obstacle_list = obstacle_list
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.z_min = 0.0
        self.z_max = 500.0

        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.threshold = 30.0

        self.node_list = []

    def planning(self):
        self.node_list = [self.start]

        for i in range(self.max_iter):
            # 1. 采样
            rnd_node = self.sample_free()

            # 2. 最近邻
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            # 3. 扩展
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # 4. 边界检查
            new_node = self.bound_point(new_node)

            # 5. 碰撞检测
            if self.check_collision(nearest_node, new_node):
                self.node_list.append(new_node)

                # 6. 检查是否到达目标
                dist_to_goal, _, _, _ = self.calc_distance_and_vector(new_node, self.goal)

                if dist_to_goal <= self.threshold or dist_to_goal <= self.expand_dis:
                    # 尝试最后一步连接
                    if self.check_collision(new_node, self.goal):
                        # 当前节点可直接连接目标点，结束本轮规划
                        self.goal.parent = new_node
                        self.goal.cost = new_node.cost + dist_to_goal
                        return self.reconstruct_path(self.goal)

        return None

    def sample_free(self):
        # 目标偏置采样
        if random.random() > self.goal_sample_rate:
            return Node(random.uniform(self.min_rand, self.max_rand),
                        random.uniform(self.min_rand, self.max_rand),
                        random.uniform(self.z_min, self.z_max))
        else:
            # 直接返回目标点坐标
            return Node(self.goal.x, self.goal.y, self.goal.z)

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y, from_node.z)
        d, dx, dy, dz = self.calc_distance_and_vector(from_node, to_node)

        if d == 0:
            return from_node

        if d <= extend_length:
            actual_extend = d
            new_node.x = to_node.x
            new_node.y = to_node.y
            new_node.z = to_node.z
        else:
            actual_extend = extend_length
            new_node.x += (dx / d) * actual_extend
            new_node.y += (dy / d) * actual_extend
            new_node.z += (dz / d) * actual_extend

        new_node.parent = from_node
        new_node.cost = from_node.cost + actual_extend
        return new_node

    def get_nearest_node_index(self, node_list, rnd_node):
        if not node_list:
            return 0
        node_coords = np.array([[n.x, n.y, n.z] for n in node_list])
        rnd_coords = np.array([rnd_node.x, rnd_node.y, rnd_node.z])
        dists_sq = np.sum((node_coords - rnd_coords) ** 2, axis=1)
        return np.argmin(dists_sq)

    def bound_point(self, node):
        node.x = max(self.min_rand, min(self.max_rand, node.x))
        node.y = max(self.min_rand, min(self.max_rand, node.y))
        node.z = max(self.z_min, min(self.z_max, node.z))
        return node

    def check_collision(self, node1, node2):
        return not environment.check_line_collision((node1.x, node1.y, node1.z),
                                            (node2.x, node2.y, node2.z),
                                            self.obstacle_list)

    def reconstruct_path(self, goal_node):
        path = [[goal_node.x, goal_node.y, goal_node.z]]
        node = goal_node.parent
        while node is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        return path[::-1]

    def calc_distance_and_vector(self, n1, n2):
        dx = n2.x - n1.x
        dy = n2.y - n1.y
        dz = n2.z - n1.z
        d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return d, dx, dy, dz

