from __future__ import annotations
import os, json, random
from collections import deque, defaultdict
from typing import Dict, Any, List
import numpy as np

def write_ndjson(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_graph(path: str) -> Dict[str, Any]:
    """Load a causal graph from a JSON file.

    Expected format is a JSON object. The function does minimal validation
    and returns the parsed dict which selectors or models can interpret.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_c_from_dag(
        graph: Dict[str, List[str]] | None,
        variables: List[str],
        target: str,
        unreachable_penalty: float = 10.0,
        min_c: float = 0.5
    ) -> np.ndarray | None:
    """
    根據 DAG 的 adjacency list graph，計算每個變數對 target 的距離。
    再把距離轉換成 weighted L1 的 c_i。

    graph: adjacency list (有向)
        {
            "A": ["B", "C"],   # A → B, A → C
            "B": ["D"],
            "C": [],
            "D": []
        }

    variables:
        需要計算 c_i 權重的特徵名稱（順序對應到 X 的 column）

    target:
        label 的名稱，例如 "Y" 或 "Outcome"

    unreachable_penalty:
        如果該變數無法到達 target（任意方向），則 c_i 設為這個值

    min_c:
        距離最小值的 offset，例如距離0 → min_c，距離1 → min_c+1

    Returns:
        np.ndarray shape (len(variables),)
    """

    if graph is None:
        return None
    
    # --- 1. 建立「無向版」的 graph，用 BFS 找最短距離 ---
    undirected = defaultdict(set)

    for u, nbrs in graph.items():
        for v in nbrs:
            undirected[u].add(v)
            undirected[v].add(u)  # 反向加邊（因為我們要算最短路徑）

    # --- 2. BFS 從 target 出發計算距離 ---
    dist = {target: 0}
    queue = deque([target])

    while queue:
        node = queue.popleft()
        for nei in undirected[node]:
            if nei not in dist:
                dist[nei] = dist[node] + 1
                queue.append(nei)

    # --- 3. 計算每個 variable 的 c_i ---
    c_list = []
    for var in variables:
        if var in dist:
            d = dist[var]
            c_list.append(min_c + d)
        else:
            c_list.append(unreachable_penalty)

    return np.array(c_list, dtype=float)