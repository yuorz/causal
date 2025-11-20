
from __future__ import annotations
from typing import List, Dict, Callable, Tuple, Any, Optional
import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

try:
    # lazy import transformers when llm selector is used
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except Exception:  # pragma: no cover - optional dependency
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None

def drop_excluded(df: pd.DataFrame, exclude_cols: List[str]) -> pd.DataFrame:
    to_drop = [c for c in exclude_cols if c in df.columns]
    return df.drop(columns=to_drop, errors="ignore")

def select_random(X: pd.DataFrame, y: pd.Series, task: str, ratio: float, random_state: int = 42, graph: Optional[Dict[str, Any]] = None) -> List[str]:
    rng = np.random.default_rng(random_state)
    candidates = list(X.columns)
    k = max(1, int(round(len(candidates) * ratio)))
    rng.shuffle(candidates)
    return candidates[:k]

def _safe_corr(x: pd.Series, y: np.ndarray) -> float:
    x = pd.to_numeric(x, errors="coerce")
    mask = x.notna() & ~pd.isna(y)
    if mask.sum() < 2 or x[mask].nunique() <= 1:
        return 0.0
    try:
        return float(np.abs(np.corrcoef(x[mask].values, y[mask].values)[0,1]))
    except Exception:
        return 0.0

def select_by_corr(X: pd.DataFrame, y: pd.Series, task: str, ratio: float, random_state: int = 42, graph: Optional[Dict[str, Any]] = None) -> List[str]:
    y_arr = y.values
    if task == "classification":
        le = LabelEncoder()
        y_arr = le.fit_transform(y.astype(str).values)

    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    if not numeric_cols:
        return select_random(X, y, task, ratio, random_state=random_state)

    scores = {col: _safe_corr(X[col], y_arr) for col in numeric_cols}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    k = max(1, int(round(len(numeric_cols) * ratio)))
    selected = [col for col, s in ranked[:k]]
    return selected


def select_by_llm(X: pd.DataFrame, y: pd.Series, task: str, ratio: float, random_state: int = 42, graph: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Prefer CUDA if available. If bitsandbytes is present, try 4-bit.
    Use accelerate if installed; otherwise fall back to single-GPU .to('cuda').
    On any error, fall back to correlation-based selection.
    """
    if pipeline is None:
        logging.warning("transformers not available; falling back to corr selector for llm method")
        return select_by_corr(X, y, task, ratio, random_state=random_state)

    candidates = list(X.columns)
    k = max(1, int(round(len(candidates) * ratio)))

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, "simple_fs_model", "llm_models")
    os.makedirs(models_dir, exist_ok=True)

    model_name = "microsoft/Phi-3-mini-4k-instruct"

    import importlib.util
    def _has(pkg: str) -> bool:
        return importlib.util.find_spec(pkg) is not None

    try:
        import torch
        use_cuda = torch.cuda.is_available()
        has_accel = _has("accelerate")
        has_bnb = _has("bitsandbytes")

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=models_dir, use_fast=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = dict(cache_dir=models_dir, low_cpu_mem_usage=True)
        # 用新的參數名 dtype=
        if use_cuda:
            if has_bnb:
                # 4-bit 優先
                load_kwargs.update(
                    dict(
                        load_in_4bit=True,
                        dtype="auto",  # 取代舊的 torch_dtype
                    )
                )
            else:
                load_kwargs.update(dict(dtype=torch.float16))

            if has_accel:
                load_kwargs.update(device_map="auto")
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            else:
                # 沒有 accelerate：避免 device_map，自行 to('cuda')
                load_kwargs.pop("device_map", None)
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
                model.to("cuda")
            device_arg = 0
        else:
            load_kwargs.update(dict(dtype="auto"))
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            device_arg = -1  # CPU

        gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device_arg)

        prompt_lines = [
            f"Task: {task}",
            f"Return the top {k} most important features as a JSON array.",
            "Features:",
        ]
        for c in candidates:
            prompt_lines.append(f"- {c}")
        # If a causal graph is provided, include a brief summary to the LLM prompt.
        if graph is not None:
            try:
                # Keep the graph summary concise: list edges if present
                edges = graph.get("edges") or graph.get("adjacency") or None
                if edges:
                    prompt_lines.append("\nCausal edges:")
                    # only include first 50 edges to avoid huge prompts
                    for e in edges[:50]:
                        prompt_lines.append(f"- {e}")
            except Exception:
                # ignore any graph formatting issues
                pass
        prompt_lines.append('Respond with only a JSON array like ["f1","f2"]. No other text.')
        prompt = "\n".join(prompt_lines)

        out = gen(
            prompt,
            max_new_tokens=128,
            do_sample=False,        # 貪婪解碼，格式最穩
            temperature=0.0,        # 會被忽略但語義清楚
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = out[0]["generated_text"]

        # 解析 JSON
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1 and end > start:
            import json as _json
            arr_text = text[start:end + 1]
            parsed = _json.loads(arr_text)
            selected = [f for f in parsed if f in candidates]
            if len(selected) >= k:
                return selected[:k]
            else:
                remaining = [c for c in candidates if c not in selected]
                try:
                    corr_rest = select_by_corr(X[remaining], y, task, 1.0, random_state=random_state)
                    filled = selected + [c for c in corr_rest if c not in selected]
                    return filled[:k]
                except Exception:
                    return (selected + remaining)[:k]

        logging.warning("LLM did not return a valid JSON array; falling back to corr selector.")
        return select_by_corr(X, y, task, ratio, random_state=random_state)

    except Exception as e:
        logging.warning("LLM feature selection failed (%s). Falling back to corr selector.", str(e))
        return select_by_corr(X, y, task, ratio, random_state=random_state)



FEATURE_SELECTORS: Dict[str, Callable[[pd.DataFrame, pd.Series, str, float, int, Optional[Dict[str, Any]]], List[str]]] = {
    "random": select_random,
    "corr": select_by_corr,
    "llm": select_by_llm,
}

def prepare_design_matrices(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    selected_features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_df[selected_features].copy()
    va = valid_df[selected_features].copy()

    cat_cols = list(tr.select_dtypes(include=["object", "category", "bool"]).columns)
    for c in cat_cols:
        if tr[c].dtype == "bool":
            tr[c] = tr[c].astype("object")
            va[c] = va[c].astype("object")

    num_cols = [c for c in tr.columns if c not in cat_cols]
    if num_cols:
        tr[num_cols] = tr[num_cols].apply(lambda s: s.fillna(s.median()), axis=0)
        va[num_cols] = va[num_cols].apply(lambda s: s.fillna(tr[s.name].median()), axis=0)

    for c in cat_cols:
        tr[c] = tr[c].astype("object").fillna("NA")
        va[c] = va[c].astype("object").fillna("NA")

    tr["_split_flag"] = 1
    va["_split_flag"] = 0
    comb = pd.concat([tr, va], axis=0, ignore_index=True)
    comb_oh = pd.get_dummies(comb, columns=cat_cols, dummy_na=False)

    tr_oh = comb_oh[comb_oh["_split_flag"] == 1].drop(columns=["_split_flag"])
    va_oh = comb_oh[comb_oh["_split_flag"] == 0].drop(columns=["_split_flag"])
    return tr_oh, va_oh
