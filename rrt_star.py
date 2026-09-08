"""带父节点优选和邻域重连的三维 RRT* 算法 """

import numpy as np
from rrt import RRT, Node


class RRTStar(RRT):
    """通过邻域重连持续降低随机树的路径代价 """

    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=80.0,
                 goal_sample_rate=0.1,
                 max_iter=3000,
                 connect_circle_dist=100.0):
        super().__init__(start, goal, obstacle_list, rand_area,
                         expand_dis, goal_sample_rate, max_iter)
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = Node(goal[0], goal[1], goal[2])
        self.goal_node.cost = float("inf")

    def planning(self):
        """构建随机树，通过邻域重连优化代价并尝试连接目标点 """
        self.node_list = [self.start]
        self.start.cost = 0.0

        for i in range(self.max_iter):

            rnd = self.sample_free()

            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd, self.expand_dis)
            new_node = self.bound_point(new_node)

            if self.check_collision(nearest_node, new_node):

                near_inds = self.find_near_nodes(new_node)

                new_node = self.choose_parent(new_node, near_inds)

                if new_node.parent:
                    self.node_list.append(new_node)

                    self.rewire(new_node, near_inds)

                    self.try_connect_to_goal(new_node)

        if self.goal_node.parent:
            return self.reconstruct_path(self.goal_node)

        return None

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list)
        r = self.connect_circle_dist
        node_coords = np.array([[node.x, node.y, node.z] for node in self.node_list])
        point = np.array([new_node.x, new_node.y, new_node.z])
        dists_sq = np.sum((node_coords - point) ** 2, axis=1)
        near_inds = np.where(dists_sq <= r ** 2)[0]
        return near_inds

    def choose_parent(self, new_node, near_inds):
        """从无碰撞邻居中选择累计代价最低的父节点 """
        if len(near_inds) == 0:
            return new_node

        costs = []
        valid_inds = []

        for i in near_inds:
            near_node = self.node_list[i]
            d, _, _, _ = self.calc_distance_and_vector(near_node, new_node)
            if self.check_collision(near_node, new_node):
                costs.append(near_node.cost + d)
                valid_inds.append(i)

        if not costs:
            new_node.parent = None
            return new_node

        min_cost = min(costs)
        min_ind = valid_inds[costs.index(min_cost)]
        new_node.cost = min_cost
        new_node.parent = self.node_list[min_ind]
        return new_node

    def rewire(self, new_node, near_inds):
        """使用新节点降低邻域节点代价并更新其父节点 """
        for i in near_inds:
            near_node = self.node_list[i]
            if near_node == new_node.parent:
                continue

            d, _, _, _ = self.calc_distance_and_vector(new_node, near_node)
            new_cost = new_node.cost + d

            if new_cost < near_node.cost:
                if self.check_collision(new_node, near_node):
                    near_node.parent = new_node
                    near_node.cost = new_cost

    def try_connect_to_goal(self, node):
        dist, _, _, _ = self.calc_distance_and_vector(node, self.goal_node)
        if self.check_collision(node, self.goal_node):
            potential_cost = node.cost + dist

            if potential_cost < self.goal_node.cost:
                self.goal_node.parent = node
                self.goal_node.cost = potential_cost

