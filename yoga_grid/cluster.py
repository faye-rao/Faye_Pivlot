"""把候选帧按体式聚类，保证九宫格里是九个不同体式而不是九张下犬式。

用平均连接层次聚类，距离超过阈值就停止合并。候选帧通常只有几十到两百个，
O(n^3) 的朴素实现完全够快，因此不引入 scipy/sklearn 依赖。
"""

from __future__ import annotations

import numpy as np

from . import landmarks as L
from .score import Candidate


def pose_distance(a: np.ndarray, b: np.ndarray, mirror_same: bool = True) -> float:
    """两个归一化姿态的距离，单位是躯干长度。

    ``mirror_same`` 为真时，左右版本的同一体式（左战士二 / 右战士二）算作
    同一个体式 —— 这样九宫格的体式种类更丰富。
    """
    core_a = a[L.CORE]
    direct = float(np.linalg.norm(core_a - b[L.CORE], axis=1).mean())
    if not mirror_same:
        return direct
    mirrored = float(
        np.linalg.norm(core_a - L.mirror_pose(b)[L.CORE], axis=1).mean()
    )
    return min(direct, mirrored)


def distance_matrix(
    candidates: list[Candidate], mirror_same: bool = True
) -> np.ndarray:
    n = len(candidates)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dist = pose_distance(
                candidates[i].frame.norm, candidates[j].frame.norm, mirror_same
            )
            d[i, j] = d[j, i] = dist
    return d


def agglomerate(d: np.ndarray, threshold: float) -> list[int]:
    """平均连接层次聚类，返回每个样本的簇标号。"""
    n = d.shape[0]
    if n == 0:
        return []

    clusters: list[list[int]] = [[i] for i in range(n)]

    while len(clusters) > 1:
        best = (np.inf, -1, -1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # 平均连接：两簇成员两两距离的均值。
                block = d[np.ix_(clusters[i], clusters[j])]
                avg = float(block.mean())
                if avg < best[0]:
                    best = (avg, i, j)

        if best[0] > threshold:
            break

        _, i, j = best
        clusters[i].extend(clusters[j])
        clusters.pop(j)

    labels = [0] * n
    # 按簇内最早出现的样本排序，让簇号大致跟随时间，读 JSON 时更直观。
    for label, members in enumerate(sorted(clusters, key=min)):
        for m in members:
            labels[m] = label
    return labels


def cluster_candidates(
    candidates: list[Candidate], threshold: float = 0.35, mirror_same: bool = True
) -> int:
    """就地写入 ``candidate.cluster``，返回簇数量。"""
    if not candidates:
        return 0
    labels = agglomerate(distance_matrix(candidates, mirror_same), threshold)
    for cand, label in zip(candidates, labels):
        cand.cluster = label
    return len(set(labels))
