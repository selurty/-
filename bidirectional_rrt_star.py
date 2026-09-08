"""双向三维 RRT* 航迹规划算法 """

import random
import numpy as np
from rrt_star import RRTStar, Node


class BidirectionalRRTStar(RRTStar):
    """从起点和终点同步扩展，并尝试连接两棵随机树 """
    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=30.0,
                 max_iter=2000,
                 connect_circle_dist=80.0):
        # 初始化父类
        super().__init__(start, goal, obstacle_list, rand_area,
                         expand_dis, goal_sample_rate=0.0, max_iter=max_iter,
                         connect_circle_dist=connect_circle_dist)

        self.start_node = Node(start[0], start[1], start[2])
        self.start_node.cost = 0.0

        self.goal_node = Node(goal[0], goal[1], goal[2])
        self.goal_node.cost = 0.0

        self.trees = [[self.start_node], [self.goal_node]]

        # 最优路径记录
        self.best_path = None
        self.c_best = float('inf')

        self.tree_orientation = 0

    @property
    def tree_start(self):
        return self.trees[0] if self.tree_orientation == 0 else self.trees[1]

    @property
    def tree_goal(self):
        return self.trees[1] if self.tree_orientation == 0 else self.trees[0]

    def planning(self):
        """
        主循环
        """
        self.node_list = self.trees[0]

        for i in range(self.max_iter):

            rnd = self.sample_free_base()

            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd, self.expand_dis)
            new_node = self.bound_point(new_node)

            if self.check_collision(nearest_node, new_node):

                near_inds = self.find_near_nodes(new_node)

                new_node = self.choose_parent(new_node, near_inds)

                if new_node.parent:
                    # 将新节点加入当前树
                    self.node_list.append(new_node)

                    self.rewire(new_node, near_inds)

                    self.connect_to_other_tree(new_node, self.trees[1])

            self.swap_trees()

        return self.best_path

    def connect_to_other_tree(self, node_in_current_tree, other_tree):

        other_coords = np.array([[n.x, n.y, n.z] for n in other_tree])
        other_costs = np.array([n.cost for n in other_tree])

        curr_pos = np.array([node_in_current_tree.x, node_in_current_tree.y, node_in_current_tree.z])

        dists = np.sqrt(np.sum((other_coords - curr_pos) ** 2, axis=1))

        radius = self.connect_circle_dist
        near_mask = dists <= radius

        if not np.any(near_mask):
            candidates_indices = [np.argmin(dists)]
        else:
            candidates_indices = np.where(near_mask)[0]

        curr_cost = node_in_current_tree.cost

        candidate_list = []
        for idx in candidates_indices:
            dist_bridge = dists[idx]
            total_cost = curr_cost + dist_bridge + other_costs[idx]

            if total_cost < self.c_best:
                candidate_list.append((total_cost, idx, dist_bridge))

        candidate_list.sort(key=lambda x: x[0])

        for total_cost, idx, dist_bridge in candidate_list:
            node_other = other_tree[idx]

            if self.check_collision(node_in_current_tree, node_other):
                self.c_best = total_cost
                self.best_path = self.generate_path(node_in_current_tree, node_other)
                break

    def sample_free_base(self):
        """纯随机采样"""
        return Node(random.uniform(self.min_rand, self.max_rand),
                    random.uniform(self.min_rand, self.max_rand),
                    random.uniform(self.z_min, self.z_max))

    def swap_trees(self):
        """交换 trees[0] 和 trees[1] 的引用，以及 node_list 指针"""
        self.trees[0], self.trees[1] = self.trees[1], self.trees[0]
        self.node_list = self.trees[0]
        self.tree_orientation = 1 - self.tree_orientation

    def generate_path(self, node_curr, node_other):
        """
        生成路径：拼接两棵树的路径
        node_curr: 当前扩展树(trees[0])中的节点
        node_other: 另一棵树(trees[1])中的节点
        """
        # 1. 回溯当前树
        path_curr = []
        node = node_curr
        while node is not None:
            path_curr.append([node.x, node.y, node.z])
            node = node.parent

        # 2. 回溯另一棵树
        path_other = []
        node = node_other
        while node is not None:
            path_other.append([node.x, node.y, node.z])
            node = node.parent

        # 3. 根据树的方向拼接
        if self.tree_orientation == 0:
            full_path = path_curr[::-1] + path_other
        else:
            full_path = path_other[::-1] + path_curr

        return full_path
