"""基于规则栅格的三维 A* 航迹规划器 """

import math
import heapq
import numpy as np


class Heuristic:
    """提供 A* 搜索使用的三维距离启发函数 """

    @staticmethod
    def get_delta(source, target):
        return abs(source[0] - target[0]), abs(source[1] - target[1]), abs(source[2] - target[2])

    @staticmethod
    def manhattan(source, target):
        dx, dy, dz = Heuristic.get_delta(source, target)
        return 10 * (dx + dy + dz)

    @staticmethod
    def euclidean(source, target):
        dx, dy, dz = Heuristic.get_delta(source, target)
        return 10 * math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


class Node:
    """记录搜索节点的坐标、父节点及累计代价 """

    def __init__(self, coordinates, parent=None):
        self.coordinates = coordinates
        self.parent = parent
        self.g = 0
        self.h = 0

    @property
    def score(self):
        return self.g + self.h

    def __lt__(self, other):
        return self.score < other.score


class AStar3DPlanner:
    """在三维占据空间中搜索无碰撞航迹 """

    def __init__(self, start, goal, obstacles, margin=5.0, resolution=25.0):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.margin = margin
        self.resolution = resolution

        self.heuristic = Heuristic.euclidean

        self.direction = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.direction.append((dx, dy, dz))

    def _pos_to_grid(self, pos):
        """将连续空间坐标量化为整数栅格坐标 """
        return tuple(int(round(p / self.resolution)) for p in pos)

    def _grid_to_pos(self, grid_coord):
        """将栅格坐标还原为连续空间坐标 """
        return np.array([c * self.resolution for c in grid_coord])

    def detect_collision(self, grid_coord):
        """判断栅格节点是否越界或进入膨胀后的障碍物 """
        rx, ry, rz = self._grid_to_pos(grid_coord)
        m = self.margin

        if not (0 <= rx <= 1000 and 0 <= ry <= 1000 and 0 <= rz <= 500):
            return True

        for obs in self.obstacles:
            bx, by, bz, dx, dy, dz = obs

            if (bx - m <= rx <= bx + dx + m and
                    by - m <= ry <= by + dy + m and
                    bz - m <= rz <= bz + dz + m):
                return True
        return False

    def find_path(self):
        """执行 A* 搜索并返回从起点到终点的三维坐标序列 """
        source_grid = self._pos_to_grid(self.start)
        target_grid = self._pos_to_grid(self.goal)

        open_set = []
        closed_set = set()
        open_dict = {}

        start_node = Node(source_grid)
        heapq.heappush(open_set, start_node)
        open_dict[source_grid] = start_node

        # 使用优先队列按 f = g + h 的升序扩展候选节点
        while open_set:
            current = heapq.heappop(open_set)

            if current.coordinates in closed_set:
                continue

            if current.coordinates == target_grid:
                path = []
                while current is not None:
                    path.append(self._grid_to_pos(current.coordinates))
                    current = current.parent
                path.reverse()

                path[0] = np.array(self.start)
                path[-1] = np.array(self.goal)
                return np.array(path)

            closed_set.add(current.coordinates)

            # 采用 26 邻域，使搜索能够沿三维对角方向扩展
            for d in self.direction:
                new_coord = (
                    current.coordinates[0] + d[0],
                    current.coordinates[1] + d[1],
                    current.coordinates[2] + d[2]
                )

                if new_coord in closed_set or self.detect_collision(new_coord):
                    continue

                dist_factor = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
                total_cost = current.g + 10 * dist_factor

                successor = open_dict.get(new_coord)
                if successor is None or total_cost < successor.g:
                    if successor is None:
                        successor = Node(new_coord, current)
                    else:
                        successor.parent = current

                    successor.g = total_cost
                    successor.h = self.heuristic(successor.coordinates, target_grid)

                    open_dict[new_coord] = successor
                    heapq.heappush(open_set, successor)

        return None
