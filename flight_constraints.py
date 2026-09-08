"""航迹限高、禁飞区及航段合规性校验工具 """

import numpy as np


def is_point_in_forbidden_zone(point, forbidden_zones=None):
    """检查点是否落入禁飞区 """
    if not forbidden_zones:
        return False

    px, py, pz = point
    for zone in forbidden_zones:
        if len(zone) < 4:
            continue

        xmin, xmax, ymin, ymax = zone[:4]
        zmin = zone[4] if len(zone) >= 6 else -float("inf")
        zmax = zone[5] if len(zone) >= 6 else float("inf")

        if xmin <= px <= xmax and ymin <= py <= ymax and zmin <= pz <= zmax:
            return True

    return False


def check_point_constraints(point, max_height=500.0, forbidden_zones=None):
    """检查单个航迹点是否满足限高与禁飞区约束 """
    px, py, pz = point
    if pz < 0 or pz > max_height:
        return False
    if is_point_in_forbidden_zone((px, py, pz), forbidden_zones):
        return False
    return True


def check_segment_constraints(p1, p2, max_height=500.0, forbidden_zones=None, step_size=5.0):
    """沿航段采样检查限高与禁飞区约束 """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    if not check_point_constraints(p1, max_height, forbidden_zones):
        return False
    if not check_point_constraints(p2, max_height, forbidden_zones):
        return False

    dist = np.linalg.norm(p2 - p1)
    steps = max(1, int(dist / step_size))
    for i in range(1, steps):
        t = i / steps
        p = p1 + (p2 - p1) * t
        if not check_point_constraints(p, max_height, forbidden_zones):
            return False

    return True


def validate_path_constraints(path, max_height=500.0, forbidden_zones=None, step_size=5.0):
    """返回航迹是否满足限高/禁飞区约束，以及违规航段数量 """
    if path is None or len(path) == 0:
        return False, 0

    path = np.array(path, dtype=float)
    violations = 0
    for i in range(len(path) - 1):
        if not check_segment_constraints(path[i], path[i + 1], max_height, forbidden_zones, step_size):
            violations += 1

    if len(path) == 1 and not check_point_constraints(path[0], max_height, forbidden_zones):
        violations += 1

    return violations == 0, violations
