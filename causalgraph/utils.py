from __future__ import annotations
import os, json, random
from collections import deque, defaultdict
from typing import Dict, Any, List, Optional
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
    graph: Optional[Dict[str, List[str]]],
    base_variables: List[str],
    design_columns: List[str],
    target: str,
    unreachable_penalty: float = 1.0,
    min_c: float = 0.2,
) -> Optional[np.ndarray]:
    """
    根據 DAG 的 adjacency list graph，計算每個「設計矩陣欄位」的加權 L1 係數 c_i。

    graph: adjacency list (有向)
        {
            "A": ["B", "C"],   # A → B, A → C
            "B": ["D"],
            "C": [],
            "D": []
        }

    base_variables:
        原始特徵名稱（跟 causal graph 對得上的那一份），例如:
        ["LotArea", "OverallQual", "Neighborhood", ...]

    design_columns:
        one-hot 之後的設計矩陣欄位名稱，順序對應到 X 的 column。
        例如:
        ["LotArea",
         "OverallQual",
         "Neighborhood_RL",
         "Neighborhood_CollgCr",
         ...]

        對於像 "Neighborhood_RL" 這種欄位，會把它視為 base variable "Neighborhood" 的一個 dummy。

    target:
        label 的名稱，例如 "SalePrice" 或 "Y"。

    unreachable_penalty:
        如果該變數無法到達 target（在無向圖裡也找不到），則 base c 設為這個值。

    min_c:
        距離最小值的 offset，例如距離 0 → min_c，距離 1 → min_c+1。

    Returns:
        np.ndarray shape (len(design_columns),)，對每個設計矩陣欄位給一個 c_i。
    """

    if graph is None:
        return None

    # --- 1. 建立「無向版」的 graph，用 BFS 找最短距離 ---
    undirected = defaultdict(set)
    for u, nbrs in graph.items():
        for v in nbrs:
            undirected[u].add(v)
            undirected[v].add(u)  # 反向加邊（因為我們要算最短路徑）

    # 確保 target 至少存在於 undirected 的 key 中
    if target not in undirected:
        undirected[target] = set()

    # --- 2. BFS 從 target 出發計算距離 ---
    dist = {target: 0}
    queue = deque([target])

    while queue:
        node = queue.popleft()
        for nei in undirected[node]:
            if nei not in dist:
                dist[nei] = dist[node] + 1
                queue.append(nei)

    # --- 3. 先對「原始變數」算 base c ---
    base_c_dict: Dict[str, float] = {}
    for var in base_variables:
        if var in dist:
            d = dist[var]
            base_c_dict[var] =  pow(min_c, 1 / (d + 1))  # 距離越遠 c 越小
        else:
            base_c_dict[var] = 1.0

    base_set = set(base_variables)

    # --- 小工具：把設計矩陣欄位 mapping 回 base variable ---
    def get_base_var(col: str) -> str:
        # 1) 欄名剛好就是原始變數
        if col in base_set:
            return col

        # 2) 嘗試找 prefix: var + "_" 的形式
        candidates = [v for v in base_variables if col.startswith(v + "_")]
        if candidates:
            # 避免有 prefix 包含關係，取最長的 match
            return max(candidates, key=len)

        # 3) 找不到就當成自己一個 node（可能不在 graph 裡）
        return col

    # --- 4. 對每個設計矩陣欄位給 c_i ---
    c_list = []
    for col in design_columns:
        base_var = get_base_var(col)
        if base_var in base_c_dict:
            c_list.append(base_c_dict[base_var])
        else:
            # 這個 base_var 根本沒有在 base_variables 裡，或不在 graph 中
            # 當成 unreachable
            c_list.append(unreachable_penalty)
    
    assert np.all(np.array(c_list) > 0.0), "All c_i should be positive."

    return np.array(c_list, dtype=float)
