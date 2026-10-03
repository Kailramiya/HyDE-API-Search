# ============================================================
# Phase 1: Retrieval Baseline for LLM-Based API Service Discovery
# Goal: Replicate paper's task setup and beat their retrieval baselines
# (Sentence-BERT: Hit@20=0.64, MRR@20=0.37)
#
# Dataset: StableToolBench ToolEnv2404 (RapidAPI dump, same domain as paper)
# Retrievers: BM25 + BGE-large-en-v1.5 + RRF fusion
# Compute: Kaggle T4 x2 (only need 1 GPU)
# Expected runtime: ~25-35 min end-to-end
# ============================================================

# %% [cell 1] Installs (run once, ~3 min)
# ============================================================
# Kaggle mein yeh cell first run kar
"""
!pip install -q sentence-transformers==3.0.1 faiss-cpu==1.8.0 rank_bm25==0.2.2 \
                groq==0.11.0 huggingface_hub==0.24.0 datasets==2.20.0 \
                FlagEmbedding==1.2.11
!pip install -q --upgrade transformers==4.44.2
"""

# %% [cell 2] Imports & Config
# ============================================================
import os
import json
import random
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Paths (Kaggle working dir)
WORK_DIR = Path("/kaggle/working")
CACHE_DIR = WORK_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# Groq key from Kaggle secrets
# Kaggle: Add-ons > Secrets > Add Secret: GROQ_API_KEY = gsk_...
try:
    from kaggle_secrets import UserSecretsClient
    GROQ_API_KEY = UserSecretsClient().get_secret("GROQ_API_KEY")
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    print("Groq key loaded from Kaggle secret")
except Exception as e:
    print(f"WARN: Groq key not loaded ({e}). Set it for Phase 3.")

# %% [cell 3] Load ToolEnv2404 (paper's RapidAPI proxy)
# ============================================================
# ToolEnv2404 structure:
#   toolenv/tools/<Category>/<tool_name>/api.py + files
#   Also has a tools.json index
#
# We want to build a corpus of:
#   category -> list of APIs -> list of endpoints
# Each endpoint has: name, description, method, parameters

from huggingface_hub import snapshot_download

print("Downloading ToolEnv2404 (RapidAPI dump ~300MB)...")
DATA_DIR = snapshot_download(
    repo_id="stabletoolbench/ToolEnv2404",
    repo_type="dataset",
    local_dir=str(CACHE_DIR / "toolenv"),
    local_dir_use_symlinks=False,
)
print(f"Downloaded to: {DATA_DIR}")

# Inspect structure
for p in Path(DATA_DIR).rglob("*.json"):
    print(p.relative_to(DATA_DIR))
    break  # just first one
print("Top-level:", list(Path(DATA_DIR).iterdir())[:5])

# %% [cell 4] Parse corpus into categories / APIs / endpoints
# ============================================================
@dataclass
class Endpoint:
    endpoint_id: str          # unique: cat/tool/endpoint
    category: str
    tool_name: str            # the "API" in paper terms
    tool_description: str
    endpoint_name: str
    endpoint_description: str
    method: str = ""
    parameters: List[Dict] = field(default_factory=list)

    def text_for_embedding(self) -> str:
        """Full text representation for dense/sparse retrieval."""
        parts = [
            f"Category: {self.category}",
            f"API: {self.tool_name}",
            f"API description: {self.tool_description}",
            f"Endpoint: {self.endpoint_name}",
            f"Endpoint description: {self.endpoint_description}",
        ]
        if self.method:
            parts.append(f"Method: {self.method}")
        if self.parameters:
            param_str = ", ".join(p.get("name", "") for p in self.parameters[:10])
            parts.append(f"Parameters: {param_str}")
        return " | ".join(parts)


def parse_toolenv(data_dir: str) -> List[Endpoint]:
    """Walk ToolEnv2404 and extract every endpoint as a retrieval doc."""
    root = Path(data_dir)
    # ToolEnv2404 uses 'toolenv/tools/<category>/<tool>/' structure
    tools_root = root / "toolenv" / "tools"
    if not tools_root.exists():
        # fallback: some dumps use 'tools/' at root
        tools_root = root / "tools"
    if not tools_root.exists():
        raise FileNotFoundError(f"Could not find tools dir under {root}")

    endpoints: List[Endpoint] = []
    categories = [p for p in tools_root.iterdir() if p.is_dir()]
    print(f"Found {len(categories)} categories")

    for cat_dir in tqdm(categories, desc="Parsing categories"):
        category = cat_dir.name
        for tool_dir in cat_dir.iterdir():
            if not tool_dir.is_dir():
                continue
            tool_name = tool_dir.name
            # Try to read tool-level metadata (name may vary)
            meta_file = None
            for candidate in [tool_dir / f"{tool_name}.json", tool_dir / "api.json"]:
                if candidate.exists():
                    meta_file = candidate
                    break
            if meta_file is None:
                # fallback: any .json in the tool dir
                jsons = list(tool_dir.glob("*.json"))
                if not jsons:
                    continue
                meta_file = jsons[0]

            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            tool_desc = meta.get("tool_description") or meta.get("description") or ""
            api_list = meta.get("api_list", meta.get("apis", []))

            for api in api_list:
                ep_name = api.get("name", api.get("api_name", ""))
                ep_desc = api.get("description", api.get("api_description", ""))
                method = api.get("method", "")
                params = api.get("required_parameters", []) + api.get("optional_parameters", [])
                if not ep_name:
                    continue
                eid = f"{category}||{tool_name}||{ep_name}"
                endpoints.append(Endpoint(
                    endpoint_id=eid,
                    category=category,
                    tool_name=tool_name,
                    tool_description=tool_desc,
                    endpoint_name=ep_name,
                    endpoint_description=ep_desc,
                    method=method,
                    parameters=params,
                ))
    return endpoints


endpoints = parse_toolenv(DATA_DIR)
print(f"Total endpoints: {len(endpoints)}")
print(f"Unique categories: {len(set(e.category for e in endpoints))}")
print(f"Unique tools (APIs): {len(set((e.category, e.tool_name) for e in endpoints))}")

# Sample
for e in endpoints[:3]:
    print("---")
    print(e.endpoint_id)
    print(e.endpoint_description[:120])

# Persist corpus
with open(CACHE_DIR / "endpoints.pkl", "wb") as f:
    pickle.dump(endpoints, f)

# %% [cell 5] Build test set (paper-style: 10 cats x 5 endpoints = 50 queries)
# ============================================================
# Paper's method: pick 10 representative categories with high API counts,
# pick 5 endpoints per category, synthesize queries via LLM.
# We'll do same with Groq + Llama-3.3-70B.

from collections import Counter

cat_counts = Counter(e.category for e in endpoints)
# Pick top 10 categories by endpoint count (paper used large categories)
top_cats = [c for c, _ in cat_counts.most_common(15)][:10]
print("Selected categories:")
for c in top_cats:
    print(f"  {c}: {cat_counts[c]} endpoints")

# Pick 5 endpoints per category (stratified by tool to get diversity)
def sample_test_endpoints(endpoints, categories, per_cat=5):
    by_cat = defaultdict(list)
    for e in endpoints:
        if e.category in categories and e.endpoint_description.strip():
            by_cat[e.category].append(e)
    test = []
    rng = random.Random(SEED)
    for cat in categories:
        pool = by_cat[cat]
        rng.shuffle(pool)
        # Prefer diverse tools
        seen_tools = set()
        picked = []
        for e in pool:
            if e.tool_name not in seen_tools:
                picked.append(e)
                seen_tools.add(e.tool_name)
            if len(picked) >= per_cat:
                break
        # If not enough diverse tools, fill from pool
        while len(picked) < per_cat and len(pool) > len(picked):
            for e in pool:
                if e not in picked:
                    picked.append(e)
                    break
        test.extend(picked[:per_cat])
    return test


test_endpoints = sample_test_endpoints(endpoints, top_cats, per_cat=5)
print(f"Test endpoints: {len(test_endpoints)}")

# %% [cell 6] Synthesize queries via Groq (paper's exact method)
# ============================================================
# Paper prompt: "Rewrite the following sentence into a concise phrase that
# summarizes its core meaning. Sentence: <endpoint_description>"

from groq import Groq
import time

client = Groq()

SYNTH_PROMPT = (
    "Rewrite the following sentence into a concise phrase "
    "(5-10 words) that summarizes its core meaning as a user query. "
    "Return only the phrase, no quotes.\n\nSentence: {desc}"
)

def synthesize_query(desc: str, model="llama-3.3-70b-versatile", retries=3) -> str:
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": SYNTH_PROMPT.format(desc=desc)}],
                temperature=0.2,
                max_tokens=40,
            )
            return resp.choices[0].message.content.strip().strip('"').strip("'")
        except Exception as e:
            if i == retries - 1:
                return desc[:80]  # fallback
            time.sleep(2 ** i)
    return desc[:80]


test_queries = []
for e in tqdm(test_endpoints, desc="Synthesizing queries"):
    q = synthesize_query(e.endpoint_description)
    test_queries.append({
        "query": q,
        "gold_endpoint_id": e.endpoint_id,
        "gold_category": e.category,
        "gold_tool": e.tool_name,
        "gold_endpoint": e.endpoint_name,
        "gold_description": e.endpoint_description,
    })

print("\nSample synthesized queries:")
for t in test_queries[:5]:
    print(f"  Q: {t['query']}")
    print(f"  Gold desc: {t['gold_description'][:80]}")
    print()

with open(CACHE_DIR / "test_queries.json", "w") as f:
    json.dump(test_queries, f, indent=2)

# %% [cell 7] BM25 retriever (lexical baseline)
# ============================================================
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> List[str]:
    import re
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

corpus_texts = [e.text_for_embedding() for e in endpoints]
tokenized_corpus = [tokenize(t) for t in tqdm(corpus_texts, desc="Tokenizing")]
bm25 = BM25Okapi(tokenized_corpus)
print(f"BM25 index built over {len(corpus_texts)} docs")


def bm25_search(query: str, top_k: int = 50) -> List[Tuple[int, float]]:
    scores = bm25.get_scores(tokenize(query))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]


# %% [cell 8] BGE-large-en-v1.5 dense retriever + FAISS
# ============================================================
from sentence_transformers import SentenceTransformer
import faiss

print("Loading BGE-large-en-v1.5...")
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=DEVICE)
# BGE-large output dim = 1024

print(f"Embedding {len(corpus_texts)} docs (batch=64, fp16)...")
corpus_embs = embedder.encode(
    corpus_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,  # cosine via inner product
)
print(f"Embedding shape: {corpus_embs.shape}, dtype: {corpus_embs.dtype}")

# FAISS inner product (since embeddings are normalized = cosine)
index = faiss.IndexFlatIP(corpus_embs.shape[1])
index.add(corpus_embs.astype(np.float32))
print(f"FAISS index size: {index.ntotal}")

# Persist
np.save(CACHE_DIR / "corpus_embs.npy", corpus_embs)


def bge_search(query: str, top_k: int = 50) -> List[Tuple[int, float]]:
    # BGE recommends adding instruction to queries
    q_emb = embedder.encode(
        [f"Represent this sentence for searching relevant passages: {query}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    scores, idxs = index.search(q_emb, top_k)
    return list(zip(idxs[0].tolist(), scores[0].tolist()))


# %% [cell 9] RRF fusion
# ============================================================
def rrf_fuse(results_list: List[List[Tuple[int, float]]], k: int = 60, top_k: int = 50) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion across multiple retrievers."""
    scores = defaultdict(float)
    for results in results_list:
        for rank, (doc_id, _score) in enumerate(results):
            scores[doc_id] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return fused


def hybrid_search(query: str, top_k: int = 50) -> List[Tuple[int, float]]:
    bm25_res = bm25_search(query, top_k=top_k)
    bge_res = bge_search(query, top_k=top_k)
    return rrf_fuse([bm25_res, bge_res], top_k=top_k)


# %% [cell 10] Evaluation (Hit@K, MRR@K)
# ============================================================
def evaluate(retriever_fn, test_queries, endpoints, ks=(5, 10, 20)) -> Dict[str, float]:
    id_to_idx = {e.endpoint_id: i for i, e in enumerate(endpoints)}
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    mrrs = {k: 0.0 for k in ks}
    n = len(test_queries)

    for tq in tqdm(test_queries, desc="Evaluating"):
        gold_idx = id_to_idx.get(tq["gold_endpoint_id"])
        if gold_idx is None:
            continue
        results = retriever_fn(tq["query"], top_k=max_k)
        ranked_ids = [r[0] for r in results]
        rank = None
        for r, doc_id in enumerate(ranked_ids, start=1):
            if doc_id == gold_idx:
                rank = r
                break
        for k in ks:
            if rank is not None and rank <= k:
                hits[k] += 1
                mrrs[k] += 1.0 / rank

    out = {}
    for k in ks:
        out[f"Hit@{k}"] = hits[k] / n
        out[f"MRR@{k}"] = mrrs[k] / n
    return out


print("\n===== BASELINE: BM25 =====")
res_bm25 = evaluate(bm25_search, test_queries, endpoints)
print(res_bm25)

print("\n===== OURS: BGE-large-en-v1.5 =====")
res_bge = evaluate(bge_search, test_queries, endpoints)
print(res_bge)

print("\n===== OURS: BM25 + BGE RRF =====")
res_hybrid = evaluate(hybrid_search, test_queries, endpoints)
print(res_hybrid)

# Paper baselines for comparison
paper_sbert = {"Hit@5": 0.48, "MRR@5": 0.3503, "Hit@10": 0.56, "MRR@10": 0.3609,
               "Hit@20": 0.64, "MRR@20": 0.3668}
paper_gpt4  = {"Hit@5": 0.60, "MRR@5": 0.5073, "Hit@10": 0.66, "MRR@10": 0.5673,
               "Hit@20": 0.74, "MRR@20": 0.6773}

print("\n===== COMPARISON TABLE =====")
print(f"{'Method':<25} {'Hit@5':<8} {'MRR@5':<8} {'Hit@10':<8} {'MRR@10':<8} {'Hit@20':<8} {'MRR@20':<8}")
for name, r in [("Paper SBERT", paper_sbert), ("Paper GPT-4 (target)", paper_gpt4),
                ("Ours BM25", res_bm25), ("Ours BGE-large", res_bge),
                ("Ours BM25+BGE RRF", res_hybrid)]:
    print(f"{name:<25} "
          f"{r['Hit@5']:<8.4f} {r['MRR@5']:<8.4f} "
          f"{r['Hit@10']:<8.4f} {r['MRR@10']:<8.4f} "
          f"{r['Hit@20']:<8.4f} {r['MRR@20']:<8.4f}")

# Save results for phase 2
with open(CACHE_DIR / "phase1_results.json", "w") as f:
    json.dump({"bm25": res_bm25, "bge": res_bge, "hybrid": res_hybrid}, f, indent=2)

print("\n✅ Phase 1 complete. Save notebook, then message me the comparison table.")
