"""SRD-HS 三维航迹规划算法 """

import numpy as np
import heapq
import itertools


class SRDHSPlanner:
    """SRD-HS 航迹规划器，结合射线检测与局部候选扩展完成搜索 """
    def __init__(
        self,
        start,
        goal,
        obstacle_list,
        margin=5.0,
        max_height=400.0,
        forbidden_zones=None,
        local_candidate_cap=18,
        max_expand=200
    ):
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.margin = margin
        self.max_height = max_height
        self.forbidden_zones = forbidden_zones if forbidden_zones else []

        self.local_candidate_cap = local_candidate_cap
        self.max_expand = max_expand

        self.counter = itertools.count()
        self.motion_cache = {}
        self.point_cache = {}

        obs_array = np.array(obstacle_list, dtype=float)
        if obs_array.size == 0:
            obs_array = np.empty((0, 6), dtype=float)

        self.raw_obs = obs_array
        self.N_obs = len(obs_array)

        if self.N_obs > 0:
            self.b_min = obs_array[:, 0:3] - self.margin
            self.b_max = obs_array[:, 0:3] + obs_array[:, 3:6] + self.margin
        else:
            self.b_min = np.empty((0, 3), dtype=float)
            self.b_max = np.empty((0, 3), dtype=float)

        self.xy_bounds = self._infer_xy_bounds()

    def _infer_xy_bounds(self):
        xs = [0.0, 1000.0, self.start[0], self.goal[0]]
        ys = [0.0, 1000.0, self.start[1], self.goal[1]]

        for bx, by, _, dx, dy, _ in self.raw_obs:
            xs.extend([bx, bx + dx])
            ys.extend([by, by + dy])

        for zone in self.forbidden_zones:
            xmin, xmax, ymin, ymax = zone[:4]
            xs.extend([xmin, xmax])
            ys.extend([ymin, ymax])

        pad = max(2.0 * self.margin, 20.0)
        return (
            min(xs) - pad,
            max(xs) + pad,
            min(ys) - pad,
            max(ys) + pad
        )

    def _point_key(self, p):
        return tuple(np.round(np.array(p, dtype=float), 2))

    def _motion_key(self, p1, p2):
        a = self._point_key(p1)
        b = self._point_key(p2)
        return (a, b) if a <= b else (b, a)

    def check_constraints(self, point):
        """检查航迹点是否满足高度和禁飞区约束，并缓存结果 """
        key = self._point_key(point)
        if key in self.point_cache:
            return self.point_cache[key]

        p = np.array(point, dtype=float)

        valid = True

        if p[2] < 0.0 or p[2] > self.max_height:
            valid = False
        elif self._point_in_any_forbidden_zone(p):
            valid = False

        self.point_cache[key] = valid
        return valid

    def _point_in_any_forbidden_zone(self, point):

        x, y = point[0], point[1]

        for zone in self.forbidden_zones:
            xmin, xmax, ymin, ymax = zone[:4]
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True

        return False

    def check_motion(self, p1, p2):
        """检查航段的端点、建筑碰撞及禁飞区穿越情况 """

        key = self._motion_key(p1, p2)
        if key in self.motion_cache:
            return self.motion_cache[key]

        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)

        valid = True

        if not self.check_constraints(p1):
            valid = False
        elif not self.check_constraints(p2):
            valid = False
        elif self._segment_hits_obstacle(p1, p2):
            valid = False
        elif self._segment_hits_forbidden_zone(p1, p2):
            valid = False

        self.motion_cache[key] = valid
        return valid

    # =====================================================
    # 建筑物 AABB 射线检测
    # =====================================================
    def _segment_hits_obstacle(self, p1, p2):
        return self._first_obstacle_hit(p1, p2) is not None

    def _first_obstacle_hit(self, p1, p2):

        if self.N_obs == 0:
            return None

        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        d = p2 - p1

        if np.dot(d, d) < 1e-9:
            return None

        inv_d = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)

        t1 = (self.b_min - p1) * inv_d
        t2 = (self.b_max - p1) * inv_d

        t_near_all = np.minimum(t1, t2)
        t_far_all = np.maximum(t1, t2)

        t_entry = np.max(t_near_all, axis=1)
        t_exit = np.min(t_far_all, axis=1)

        hit_mask = (
            (t_entry <= t_exit)
            & (t_entry >= 0.0)
            & (t_entry <= 1.0)
            & (t_exit >= 0.0)
        )

        hit_indices = np.where(hit_mask)[0]

        if len(hit_indices) == 0:
            return None

        idx = hit_indices[np.argmin(t_entry[hit_indices])]
        axis = int(np.argmax(t_near_all[idx]))
        t_hit = float(t_entry[idx])
        p_hit = p1 + t_hit * d

        return {
            "type": "obstacle",
            "t": t_hit,
            "point": p_hit,
            "axis": axis,
            "index": int(idx)
        }

    # =====================================================
    # 禁飞区解析检测
    # =====================================================
    def _segment_hits_forbidden_zone(self, p1, p2):
        return self._first_forbidden_zone_hit(p1, p2) is not None

    def _first_forbidden_zone_hit(self, p1, p2):

        if not self.forbidden_zones:
            return None

        best_hit = None

        for zone_index, zone in enumerate(self.forbidden_zones):
            hit = self._segment_rectangle_hit_xy(p1, p2, zone, use_margin=True)

            if hit is None:
                continue

            hit["type"] = "forbidden"
            hit["zone_index"] = zone_index
            hit["zone"] = zone

            if best_hit is None or hit["t"] < best_hit["t"]:
                best_hit = hit

        return best_hit

    def _segment_rectangle_hit_xy(self, p1, p2, zone, use_margin=True):

        xmin, xmax, ymin, ymax = zone[:4]

        if use_margin:
            xmin -= self.margin
            xmax += self.margin
            ymin -= self.margin
            ymax += self.margin

        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)

        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]

        dx = x2 - x1
        dy = y2 - y1

        t_min = 0.0
        t_max = 1.0
        entry_axis = 0

        if abs(dx) < 1e-9:
            if x1 < xmin or x1 > xmax:
                return None
        else:
            tx1 = (xmin - x1) / dx
            tx2 = (xmax - x1) / dx

            t_low = min(tx1, tx2)
            t_high = max(tx1, tx2)

            if t_low > t_min:
                entry_axis = 0

            t_min = max(t_min, t_low)
            t_max = min(t_max, t_high)

            if t_min > t_max:
                return None

        if abs(dy) < 1e-9:
            if y1 < ymin or y1 > ymax:
                return None
        else:
            ty1 = (ymin - y1) / dy
            ty2 = (ymax - y1) / dy

            t_low = min(ty1, ty2)
            t_high = max(ty1, ty2)

            if t_low > t_min:
                entry_axis = 1

            t_min = max(t_min, t_low)
            t_max = min(t_max, t_high)

            if t_min > t_max:
                return None

        if t_max < 0.0 or t_min > 1.0:
            return None

        t_hit = max(t_min, 0.0)
        p_hit = p1 + t_hit * (p2 - p1)

        return {
            "t": float(t_hit),
            "point": p_hit,
            "axis": entry_axis
        }

    def _segment_rectangle_span_xy(self, p1, p2, zone, use_margin=True):

        xmin, xmax, ymin, ymax = zone[:4]

        if use_margin:
            xmin -= self.margin
            xmax += self.margin
            ymin -= self.margin
            ymax += self.margin

        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)

        x1, y1 = p1[0], p1[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        t_min = 0.0
        t_max = 1.0
        entry_axis = 0
        exit_axis = 0

        if abs(dx) < 1e-9:
            if x1 < xmin or x1 > xmax:
                return None
        else:
            tx1 = (xmin - x1) / dx
            tx2 = (xmax - x1) / dx
            t_low = min(tx1, tx2)
            t_high = max(tx1, tx2)

            if t_low > t_min:
                entry_axis = 0
            if t_high < t_max:
                exit_axis = 0

            t_min = max(t_min, t_low)
            t_max = min(t_max, t_high)

            if t_min > t_max:
                return None

        if abs(dy) < 1e-9:
            if y1 < ymin or y1 > ymax:
                return None
        else:
            ty1 = (ymin - y1) / dy
            ty2 = (ymax - y1) / dy
            t_low = min(ty1, ty2)
            t_high = max(ty1, ty2)

            if t_low > t_min:
                entry_axis = 1
            if t_high < t_max:
                exit_axis = 1

            t_min = max(t_min, t_low)
            t_max = min(t_max, t_high)

            if t_min > t_max:
                return None

        if t_max < 0.0 or t_min > 1.0:
            return None

        t_in = max(t_min, 0.0)
        t_out = min(t_max, 1.0)
        d = p2 - p1

        return {
            "t": float(t_in),
            "t_in": float(t_in),
            "t_out": float(t_out),
            "p_in": p1 + t_in * d,
            "p_out": p1 + t_out * d,
            "axis_in": entry_axis,
            "axis_out": exit_axis
        }

    def _first_blocker(self, p_from, p_to):

        obs_hit = self._first_obstacle_hit(p_from, p_to)
        nfz_hit = self._first_forbidden_zone_hit(p_from, p_to)

        if obs_hit is None and nfz_hit is None:
            return None

        if obs_hit is None:
            return nfz_hit

        if nfz_hit is None:
            return obs_hit

        return obs_hit if obs_hit["t"] <= nfz_hit["t"] else nfz_hit

    def _all_obstacle_hits(self, p1, p2):

        if self.N_obs == 0:
            return []

        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        d = p2 - p1

        if np.dot(d, d) < 1e-9:
            return []

        inv_d = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)

        t1 = (self.b_min - p1) * inv_d
        t2 = (self.b_max - p1) * inv_d

        t_near_all = np.minimum(t1, t2)
        t_far_all = np.maximum(t1, t2)

        t_entry = np.max(t_near_all, axis=1)
        t_exit = np.min(t_far_all, axis=1)

        hit_mask = (
            (t_entry <= t_exit)
            & (t_exit >= 0.0)
            & (t_entry <= 1.0)
        )

        hits = []

        for idx in np.where(hit_mask)[0]:
            t_in = max(0.0, float(t_entry[idx]))
            t_out = min(1.0, float(t_exit[idx]))

            hits.append({
                "type": "obstacle",
                "t": t_in,
                "t_in": t_in,
                "t_out": t_out,
                "p_in": p1 + t_in * d,
                "p_out": p1 + t_out * d,
                "axis_in": int(np.argmax(t_near_all[idx])),
                "axis_out": int(np.argmin(t_far_all[idx])),
                "index": int(idx)
            })

        hits.sort(key=lambda h: h["t"])
        return hits

    def _all_forbidden_zone_hits(self, p1, p2):
        hits = []

        for zone_index, zone in enumerate(self.forbidden_zones):
            span = self._segment_rectangle_span_xy(
                p1,
                p2,
                zone,
                use_margin=True
            )

            if span is None:
                continue

            span["type"] = "forbidden"
            span["zone_index"] = zone_index
            span["zone"] = zone
            hits.append(span)

        hits.sort(key=lambda h: h["t"])
        return hits

    # =====================================================
    # 局部候选点生成
    # =====================================================
    def _generate_local_candidates(self, curr, goal):
        """围绕当前视线遇到的首个阻挡体生成局部绕行候选点 """

        blocker = self._first_blocker(curr, goal)

        if blocker is None:
            return []

        if blocker["type"] == "obstacle":
            candidates = self._generate_obstacle_candidates(blocker)
        else:
            candidates = self._generate_forbidden_zone_candidates(blocker)

        return self._rank_and_filter_candidates(candidates, curr, goal)

    def _generate_obstacle_candidates(self, blocker):

        candidates = []

        idx = blocker["index"]
        p_hit = blocker["point"]

        bx, by, bz, dx, dy, dz = self.raw_obs[idx]

        m_extra = self.margin + 1.0

        xmin = bx - m_extra
        xmax = bx + dx + m_extra
        ymin = by - m_extra
        ymax = by + dy + m_extra

        z_side = min(max(p_hit[2], 0.0), self.max_height)
        z_top = bz + dz + m_extra

        side_candidates = [

            (xmin, ymin, z_side),
            (xmin, ymax, z_side),
            (xmax, ymin, z_side),
            (xmax, ymax, z_side),


            (xmin, p_hit[1], z_side),
            (xmax, p_hit[1], z_side),
            (p_hit[0], ymin, z_side),
            (p_hit[0], ymax, z_side),
        ]

        candidates.extend(side_candidates)

        if z_top <= self.max_height:
            top_candidates = [
                (p_hit[0], p_hit[1], z_top),

                (xmin, p_hit[1], z_top),
                (xmax, p_hit[1], z_top),
                (p_hit[0], ymin, z_top),
                (p_hit[0], ymax, z_top),

                (xmin, ymin, z_top),
                (xmin, ymax, z_top),
                (xmax, ymin, z_top),
                (xmax, ymax, z_top),
            ]

            candidates.extend(top_candidates)

        return candidates

    def _generate_forbidden_zone_candidates(self, blocker):

        candidates = []

        zone = blocker["zone"]
        p_hit = blocker["point"]

        xmin, xmax, ymin, ymax = zone[:4]

        m_extra = self.margin + 1.0

        z = min(max(p_hit[2], 0.0), self.max_height)

        hx = min(max(p_hit[0], xmin), xmax)
        hy = min(max(p_hit[1], ymin), ymax)

        left = xmin - m_extra
        right = xmax + m_extra
        bottom = ymin - m_extra
        top = ymax + m_extra

        candidates.extend([
            (left, hy, z),
            (right, hy, z),
            (hx, bottom, z),
            (hx, top, z),
        ])

        candidates.extend([
            (left, bottom, z),
            (left, top, z),
            (right, bottom, z),
            (right, top, z),
        ])

        extra = self.margin + 10.0

        candidates.extend([
            (left - extra, bottom - extra, z),
            (left - extra, top + extra, z),
            (right + extra, bottom - extra, z),
            (right + extra, top + extra, z),
        ])

        return candidates

    def _rank_and_filter_candidates(self, candidates, curr, goal, cap=None):

        curr = np.array(curr, dtype=float)
        goal = np.array(goal, dtype=float)

        unique = {}

        for c in candidates:
            p = np.array(c, dtype=float)

            if not self.check_constraints(p):
                continue

            if np.linalg.norm(p - curr) < 1e-6:
                continue

            key = tuple(np.round(p, 2))
            unique[key] = p

        ranked = list(unique.values())

        if not ranked:
            return []

        direct_dist = np.linalg.norm(curr - goal)
        xmin, xmax, ymin, ymax = self.xy_bounds

        ranked = [
            p for p in ranked
            if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax
        ]

        if not ranked:
            return []

        def score(p):
            step = np.linalg.norm(curr - p)
            remain = np.linalg.norm(p - goal)

            progress = direct_dist - remain

            return step + remain - 0.15 * max(progress, 0.0)

        ranked.sort(key=score)

        effective_cap = self.local_candidate_cap if cap is None else cap

        if effective_cap is not None and len(ranked) > effective_cap:
            ranked = ranked[:effective_cap]

        return ranked

    def _generate_ray_candidates(self, curr, goal, first_only=False, cap=None):

        hits = self._all_obstacle_hits(curr, goal)
        hits.extend(self._all_forbidden_zone_hits(curr, goal))
        hits.sort(key=lambda h: h["t"])

        if first_only and hits:
            hits = hits[:1]

        candidates = []

        for hit in hits:
            if hit["type"] == "obstacle":
                entry_blocker = {
                    "type": "obstacle",
                    "point": hit["p_in"],
                    "axis": hit["axis_in"],
                    "index": hit["index"]
                }
                exit_blocker = {
                    "type": "obstacle",
                    "point": hit["p_out"],
                    "axis": hit["axis_out"],
                    "index": hit["index"]
                }
                candidates.extend(self._generate_obstacle_candidates(entry_blocker))
                candidates.extend(self._generate_obstacle_candidates(exit_blocker))
            else:
                entry_blocker = {
                    "type": "forbidden",
                    "point": hit["p_in"],
                    "zone": hit["zone"]
                }
                exit_blocker = {
                    "type": "forbidden",
                    "point": hit["p_out"],
                    "zone": hit["zone"]
                }
                candidates.extend(self._generate_forbidden_zone_candidates(entry_blocker))
                candidates.extend(self._generate_forbidden_zone_candidates(exit_blocker))

        ranked = self._rank_and_filter_candidates(
            candidates,
            curr,
            goal,
            cap=len(candidates)
        )

        reachable = []
        deferred = []

        for p in ranked:
            if self.check_motion(curr, p):
                reachable.append(p)
            else:
                deferred.append(p)

        ordered = reachable + deferred

        if cap is not None and len(ordered) > cap:
            ordered = ordered[:cap]

        return ordered

    def _build_candidate_cloud(
        self,
        depth=3,
        per_ray_cap=42,
        frontier_cap=100,
        total_cap=420,
        branch_keep=10
    ):
        goal_key = self._point_key(self.goal)
        candidates = {goal_key: self.goal}
        protected_keys = {goal_key}
        frontier = [self.start]

        for depth_index in range(depth):
            next_frontier = []
            priority_frontier = []

            for p in frontier:
                ray_candidates = self._generate_ray_candidates(
                    p,
                    self.goal,
                    first_only=False,
                    cap=per_ray_cap
                )

                for candidate_rank, c in enumerate(ray_candidates):
                    key = self._point_key(c)
                    if key in candidates:
                        continue

                    candidates[key] = np.array(c, dtype=float)
                    next_frontier.append(np.array(c, dtype=float))

                    if len(priority_frontier) < frontier_cap:
                        if candidate_rank < branch_keep:
                            priority_frontier.append(np.array(c, dtype=float))

            if not next_frontier:
                break

            ranked_frontier = self._rank_and_filter_candidates(
                next_frontier,
                self.start,
                self.goal,
                cap=len(next_frontier)
            )

            ordered_frontier = []
            ordered_keys = set()

            for source in (priority_frontier, ranked_frontier):
                for p in source:
                    key = self._point_key(p)
                    if key in ordered_keys:
                        continue

                    ordered_keys.add(key)
                    ordered_frontier.append(np.array(p, dtype=float))

                    if len(ordered_frontier) >= frontier_cap:
                        break

                if len(ordered_frontier) >= frontier_cap:
                    break

            next_frontier = ordered_frontier

            for p in next_frontier:
                protected_keys.add(self._point_key(p))

            frontier = next_frontier

            if len(candidates) > total_cap:
                protected = {
                    key: candidates[key]
                    for key in protected_keys
                    if key in candidates
                }
                remaining = [
                    p for key, p in candidates.items()
                    if key not in protected
                ]
                ranked = self._rank_and_filter_candidates(
                    remaining,
                    self.start,
                    self.goal,
                    cap=max(0, total_cap - len(protected))
                )
                candidates = dict(protected)
                candidates.update({
                    self._point_key(p): np.array(p, dtype=float)
                    for p in ranked
                })
                candidates[goal_key] = self.goal

        return list(candidates.values())

    def _planning_candidate_graph(self):

        candidate_sets = [
            self._build_candidate_cloud(
                depth=3,
                per_ray_cap=46,
                frontier_cap=120,
                total_cap=520
            ),
            self._build_candidate_cloud(
                depth=4,
                per_ray_cap=58,
                frontier_cap=160,
                total_cap=760
            )
        ]

        for candidates in candidate_sets:
            result = self._search_candidate_graph(candidates)
            if result is not None:
                return result

        return None

    def _planning_layered_ray(
        self,
        candidate_cap=30,
        beam_width=40,
        max_depth=4,
        prune_factor=1.08,
        height_weight=0.15
    ):

        if not self.check_constraints(self.start):
            return None

        if not self.check_constraints(self.goal):
            return None

        states = [(0.0, self.start, [self.start])]
        best = None

        for depth in range(max_depth + 1):
            for g, curr, path in states:
                if self.check_motion(curr, self.goal):
                    total_cost = g + np.linalg.norm(curr - self.goal)
                    if best is None or total_cost < best[0]:
                        best = (total_cost, path + [self.goal])

            if depth == max_depth:
                break

            next_states = []

            for g, curr, path in states:
                ray_candidates = self._generate_ray_candidates(
                    curr,
                    self.goal,
                    first_only=False,
                    cap=candidate_cap
                )

                for p_next in ray_candidates:
                    p_next = np.array(p_next, dtype=float)

                    if not self.check_motion(curr, p_next):
                        continue

                    if any(np.linalg.norm(p_next - old_p) < 1e-6 for old_p in path):
                        continue

                    new_g = g + np.linalg.norm(p_next - curr)

                    if self.check_motion(p_next, self.goal):
                        total_cost = new_g + np.linalg.norm(p_next - self.goal)
                        if best is None or total_cost < best[0]:
                            best = (total_cost, path + [p_next, self.goal])

                    optimistic_cost = new_g + np.linalg.norm(p_next - self.goal)
                    if best is not None and optimistic_cost >= best[0] * prune_factor:
                        continue

                    next_states.append((new_g, p_next, path + [p_next]))

            deduped = {}

            for g, curr, path in next_states:
                key = self._point_key(curr)

                if key not in deduped or g < deduped[key][0]:
                    deduped[key] = (g, curr, path)

            states = list(deduped.values())

            states.sort(
                key=lambda item: (
                    item[0]
                    + np.linalg.norm(item[1] - self.goal)
                    + height_weight * max(item[1][2] - self.goal[2], 0.0)
                )
            )

            if len(states) > beam_width:
                states = states[:beam_width]

            if not states and best is not None:
                break

        if best is None:
            return None

        return self._optimize_and_smooth(best[1])

    def _preferred_corridor_height(self):
        base_height = max(float(self.start[2]), float(self.goal[2]))

        if self.N_obs == 0:
            return min(self.max_height, base_height + 40.0)

        tops = sorted({
            float(bz + dz + self.margin + 1.0)
            for _, _, bz, _, _, dz in self.raw_obs
            if 0.0 <= bz + dz + self.margin + 1.0 <= self.max_height
        })

        for top in tops:
            if top >= base_height + 5.0:
                return min(top, base_height + 60.0, self.max_height)

        return min(self.max_height, base_height + 60.0)

    def _build_boundary_corridor_candidates(self):

        candidates = {}
        m = self.margin + 1.0
        cruise_z = self._preferred_corridor_height()
        xmin_bound, xmax_bound, ymin_bound, ymax_bound = self.xy_bounds

        def add(point):
            p = np.array(point, dtype=float)

            if p[2] < 0.0 or p[2] > self.max_height:
                return

            if not (xmin_bound <= p[0] <= xmax_bound and ymin_bound <= p[1] <= ymax_bound):
                return

            if not self.check_constraints(p):
                return

            candidates[self._point_key(p)] = p

        for bx, by, bz, dx, dy, dz in self.raw_obs:
            xmin = bx - m
            xmax = bx + dx + m
            ymin = by - m
            ymax = by + dy + m
            cx = (xmin + xmax) * 0.5
            cy = (ymin + ymax) * 0.5
            z_top = bz + dz + m

            z_layers = [z_top]

            if z_top > cruise_z + 80.0:
                z_layers.insert(0, cruise_z)

            for z in z_layers:
                if z < 0.0 or z > self.max_height:
                    continue

                for point in (
                    (xmin, ymin, z),
                    (xmin, ymax, z),
                    (xmax, ymin, z),
                    (xmax, ymax, z),
                    (xmin, cy, z),
                    (xmax, cy, z),
                    (cx, ymin, z),
                    (cx, ymax, z),
                ):
                    add(point)

        for zone in self.forbidden_zones:
            xmin, xmax, ymin, ymax = zone[:4]
            xmin -= m
            xmax += m
            ymin -= m
            ymax += m
            cx = (xmin + xmax) * 0.5
            cy = (ymin + ymax) * 0.5

            for z in (float(self.start[2]), cruise_z):
                for point in (
                    (xmin, ymin, z),
                    (xmin, ymax, z),
                    (xmax, ymin, z),
                    (xmax, ymax, z),
                    (xmin, cy, z),
                    (xmax, cy, z),
                    (cx, ymin, z),
                    (cx, ymax, z),
                ):
                    add(point)

        return list(candidates.values())

    def _planning_boundary_corridor(self, neighbor_cap=60, bridge_cap=50, max_expand=400):

        if not self.check_constraints(self.start):
            return None

        if not self.check_constraints(self.goal):
            return None

        candidates = self._build_boundary_corridor_candidates()

        if not candidates:
            return None

        start_key = self._point_key(self.start)
        goal_key = self._point_key(self.goal)

        candidate_map = {
            self._point_key(p): np.array(p, dtype=float)
            for p in candidates
            if self._point_key(p) != start_key
        }
        candidate_map[goal_key] = self.goal

        goal_visible_keys = {
            key for key, p in candidate_map.items()
            if self.check_motion(p, self.goal)
        }

        pq = [
            (
                np.linalg.norm(self.start - self.goal),
                0.0,
                next(self.counter),
                start_key,
                [self.start]
            )
        ]

        visited = {start_key: 0.0}
        expand_count = 0
        base_height = max(float(self.start[2]), float(self.goal[2]))

        while pq and expand_count < max_expand:
            _, g, _, curr_key, path = heapq.heappop(pq)

            if g > visited.get(curr_key, float("inf")):
                continue

            curr = np.array(curr_key, dtype=float)
            expand_count += 1

            if self.check_motion(curr, self.goal):
                return self._optimize_and_smooth(path + [self.goal])

            bridge_ranked = []
            regular_ranked = []

            for next_key, p_next in candidate_map.items():
                if next_key == curr_key:
                    continue

                step_dist = np.linalg.norm(p_next - curr)
                h = np.linalg.norm(p_next - self.goal)
                alt_penalty = 0.25 * max(p_next[2] - base_height, 0.0)
                item = (step_dist + h + alt_penalty, step_dist, h, next_key, p_next)

                if next_key in goal_visible_keys:
                    bridge_ranked.append(item)
                else:
                    regular_ranked.append(item)

            bridge_ranked.sort(key=lambda item: item[0])
            regular_ranked.sort(key=lambda item: item[0])

            for _, step_dist, h, next_key, p_next in (
                bridge_ranked[:bridge_cap] + regular_ranked[:neighbor_cap]
            ):
                new_g = g + step_dist

                if new_g >= visited.get(next_key, float("inf")):
                    continue

                if not self.check_motion(curr, p_next):
                    continue

                visited[next_key] = new_g
                heapq.heappush(
                    pq,
                    (
                        new_g + h,
                        new_g,
                        next(self.counter),
                        next_key,
                        path + [p_next]
                    )
                )

        return None

    def _search_candidate_graph(self, candidates, neighbor_cap=180, max_expand=900):
        start_key = self._point_key(self.start)
        candidate_map = {}

        for p in candidates:
            key = self._point_key(p)
            if key != start_key:
                candidate_map[key] = np.array(p, dtype=float)

        goal_visible_keys = {
            key for key, p in candidate_map.items()
            if self.check_motion(p, self.goal)
        }

        pq = [
            (
                np.linalg.norm(self.start - self.goal),
                0.0,
                next(self.counter),
                start_key,
                [self.start]
            )
        ]

        visited = {start_key: 0.0}
        expand_count = 0

        while pq and expand_count < max_expand:
            _, g, _, curr_key, path = heapq.heappop(pq)

            if g > visited.get(curr_key, float("inf")):
                continue

            curr = np.array(curr_key, dtype=float)
            expand_count += 1

            if self.check_motion(curr, self.goal):
                return self._optimize_and_smooth(path + [self.goal])

            ranked_goal_visible = []
            ranked_regular = []

            for next_key, p_next in candidate_map.items():
                if next_key == curr_key:
                    continue

                step_dist = np.linalg.norm(p_next - curr)
                h = np.linalg.norm(p_next - self.goal)
                item = (step_dist + h, step_dist, h, next_key, p_next)

                if next_key in goal_visible_keys:
                    ranked_goal_visible.append(item)
                else:
                    ranked_regular.append(item)

            ranked_goal_visible.sort(key=lambda item: item[0])
            ranked_regular.sort(key=lambda item: item[0])

            curr_neighbor_cap = len(ranked_regular) if expand_count <= 2 else neighbor_cap
            ranked = ranked_goal_visible + ranked_regular[:curr_neighbor_cap]

            for _, step_dist, h, next_key, p_next in ranked:
                new_g = g + step_dist

                if new_g >= visited.get(next_key, float("inf")):
                    continue

                if not self.check_motion(curr, p_next):
                    continue

                visited[next_key] = new_g
                heapq.heappush(
                    pq,
                    (
                        new_g + h,
                        new_g,
                        next(self.counter),
                        next_key,
                        path + [p_next]
                    )
                )

        return None


    def _planning_dynamic_local(self):

        if not self.check_constraints(self.start):
            return None

        if not self.check_constraints(self.goal):
            return None

        start_key = self._point_key(self.start)

        pq = [
            (
                np.linalg.norm(self.start - self.goal),
                0.0,
                next(self.counter),
                start_key,
                [self.start]
            )
        ]

        visited = {start_key: 0.0}
        expand_count = 0

        while pq and expand_count < self.max_expand:
            _, g, _, curr_key, path = heapq.heappop(pq)

            if g > visited.get(curr_key, float("inf")):
                continue

            curr = np.array(curr_key, dtype=float)
            expand_count += 1

            if self.check_motion(curr, self.goal):
                final_path = path + [self.goal]
                return self._optimize_and_smooth(final_path)

            candidates = self._generate_local_candidates(curr, self.goal)

            pushed = 0

            for p_next in candidates:
                p_next = np.array(p_next, dtype=float)
                p_next_key = self._point_key(p_next)

                if p_next_key == curr_key:
                    continue

                step_dist = np.linalg.norm(p_next - curr)
                new_g = g + step_dist

                if new_g >= visited.get(p_next_key, float("inf")):
                    continue

                if not self.check_motion(curr, p_next):
                    continue

                visited[p_next_key] = new_g

                h = np.linalg.norm(p_next - self.goal)
                f = new_g + h

                heapq.heappush(
                    pq,
                    (
                        f,
                        new_g,
                        next(self.counter),
                        p_next_key,
                        path + [p_next]
                    )
                )

                pushed += 1

            if pushed == 0:
                continue

        return None

    def planning(self):
        """按场景复杂度依次执行分层射线、边界走廊和局部搜索 """
        # 密集环境优先使用边界走廊，减少无效射线候选的扩展量
        if self.N_obs >= 45:
            result = self._planning_boundary_corridor(
                neighbor_cap=48,
                bridge_cap=38,
                max_expand=340
            )

            if result is not None:
                return result

        result = self._planning_layered_ray(
            candidate_cap=30,
            beam_width=40,
            max_depth=3,
            prune_factor=1.08,
            height_weight=0.15
        )

        if result is not None:
            return result

        result = self._planning_boundary_corridor(
            neighbor_cap=48,
            bridge_cap=38,
            max_expand=340
        )

        if result is not None:
            return result

        result = self._planning_layered_ray(
            candidate_cap=40,
            beam_width=80,
            max_depth=5,
            prune_factor=1.10,
            height_weight=0.12
        )

        if result is not None:
            return result

        return self._planning_dynamic_local()

    # =====================================================
    # 双向视线剪枝
    # =====================================================
    def _line_of_sight_pruning(self, path):
        """双向删除可被直连航段替代的冗余航迹点 """

        if len(path) <= 2:
            return path

        def single_pass(p_list):
            res = [p_list[0]]
            i = 0

            while i < len(p_list) - 1:
                found = False

                for j in range(len(p_list) - 1, i, -1):
                    if self.check_motion(p_list[i], p_list[j]):
                        res.append(p_list[j])
                        i = j
                        found = True
                        break

                if not found:
                    i += 1
                    if i < len(p_list):
                        res.append(p_list[i])

            return res

        path = single_pass(path)
        path = single_pass(path[::-1])[::-1]

        return path

    # =====================================================
    # 航迹优化和平滑
    # =====================================================
    def _optimize_and_smooth(self, path):
        """先执行视线剪枝，再生成平滑且满足约束的连续航迹 """
        waypoints = self._line_of_sight_pruning(path)
        waypoints = np.array(waypoints, dtype=float)

        if len(waypoints) <= 2:
            dist = np.linalg.norm(waypoints[1] - waypoints[0])
            num_pts = max(2, int(dist / 2.0))
            trajectory = np.linspace(waypoints[0], waypoints[1], num_pts)
            return waypoints, None, trajectory

        result = self._final_trajectory_generation(waypoints)

        if self._path_is_feasible(result[2]):
            return result

        return waypoints, None, self._densify_polyline(waypoints)

    def _path_is_feasible(self, path):

        if path is None or len(path) < 2:
            return False

        for i in range(len(path) - 1):
            if not self.check_motion(path[i], path[i + 1]):
                return False

        return True

    def _densify_polyline(self, waypoints):
        trajectory = [waypoints[0]]

        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]

            dist = np.linalg.norm(p2 - p1)
            pts = np.linspace(p1, p2, max(2, int(dist / 2.0)))

            trajectory.extend(pts[1:])

        return np.array(trajectory)

    def _final_trajectory_generation(self, waypoints):
        """
        三次贝塞尔曲线平滑
        """
        trajectory = []

        turn_dist = 50.0

        current_pos = waypoints[0]
        trajectory.append(current_pos)

        for i in range(1, len(waypoints) - 1):
            prev_wp = waypoints[i - 1]
            curr_wp = waypoints[i]
            next_wp = waypoints[i + 1]

            v_in = curr_wp - prev_wp
            v_out = next_wp - curr_wp

            l_in = np.linalg.norm(v_in)
            l_out = np.linalg.norm(v_out)

            if l_in < 1e-9 or l_out < 1e-9:
                continue

            d = min(turn_dist, l_in * 0.4, l_out * 0.4)

            dir_in = v_in / l_in
            dir_out = v_out / l_out

            P0 = curr_wp - d * dir_in
            P3 = curr_wp + d * dir_out
            P1 = curr_wp - d * 0.5 * dir_in
            P2 = curr_wp + d * 0.5 * dir_out

            seg_dist = np.linalg.norm(P0 - current_pos)

            if seg_dist > 0.1:
                trajectory.extend(
                    np.linspace(
                        current_pos,
                        P0,
                        max(2, int(seg_dist / 2.0))
                    )[1:]
                )

            t = np.linspace(0, 1, 10)[:, np.newaxis]

            curve = (
                (1 - t) ** 3 * P0
                + 3 * (1 - t) ** 2 * t * P1
                + 3 * (1 - t) * t ** 2 * P2
                + t ** 3 * P3
            )

            trajectory.extend(curve[1:])
            current_pos = P3

        final_dist = np.linalg.norm(waypoints[-1] - current_pos)

        if final_dist > 0.1:
            trajectory.extend(
                np.linspace(
                    current_pos,
                    waypoints[-1],
                    max(2, int(final_dist / 2.0))
                )[1:]
            )

        return waypoints, None, np.array(trajectory)
