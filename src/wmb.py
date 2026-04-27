

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np


try:
    from config import CIRAConfig as _cfg

    _CAPACITY        : int   = _cfg.WMB_CAPACITY
    _ALPHA           : float = _cfg.WMB_ALPHA
    _BETA            : float = _cfg.WMB_BETA
    _DECAY_HALF_LIFE : float = _cfg.WMB_DECAY_HALF_LIFE_S
    _EMBED_DIM       : int   = _cfg.EMBED_DIM
except (ImportError, AttributeError):
    _CAPACITY        = 7
    _ALPHA           = 0.4
    _BETA            = 0.6
    _DECAY_HALF_LIFE = 3600.0
    _EMBED_DIM       = 1024

_ALPHA_BETA_TOL: float = 1e-9

logging.basicConfig(level=logging.INFO, format="%(levelname)s | wmb | %(message)s")
_log = logging.getLogger(__name__)

assert abs(_ALPHA + _BETA - 1.0) < _ALPHA_BETA_TOL, (
    f"Config error: ALPHA ({_ALPHA}) + BETA ({_BETA}) = {_ALPHA + _BETA:.10f} ≠ 1.0"
)




@dataclass
class MemoryItem:
    

    text:         str
    embedding:    np.ndarray
    salience:     float
    timestamp:    float
    relevance:    float = 0.5   
    access_count: int   = 0

    def __lt__(self, other: "MemoryItem") -> bool:
        
        return self.salience < other.salience

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryItem):
            return NotImplemented
        return self.salience == other.salience

    def __le__(self, other: "MemoryItem") -> bool:
        return self.salience <= other.salience

    def __gt__(self, other: "MemoryItem") -> bool:
        return self.salience > other.salience

    def __ge__(self, other: "MemoryItem") -> bool:
        return self.salience >= other.salience




class WMB:
    

    def __init__(
        self,
        capacity:        int   = _CAPACITY,
        alpha:           float = _ALPHA,
        beta:            float = _BETA,
        decay_half_life: float = _DECAY_HALF_LIFE,
        embed_dim:       int   = _EMBED_DIM,
    ) -> None:
        
        if not (1 <= capacity <= 50):
            raise ValueError(
                f"[WMB.__init__] capacity must be in [1, 50], got {capacity}."
            )
        if alpha < 0.0 or beta < 0.0:
            raise ValueError(
                f"[WMB.__init__] alpha={alpha} and beta={beta} must both be >= 0.0."
            )
        if abs(alpha + beta - 1.0) > _ALPHA_BETA_TOL:
            raise ValueError(
                f"[WMB.__init__] α + β must equal 1.0, got {alpha + beta:.10f}."
            )
        if decay_half_life <= 0.0:
            raise ValueError(
                f"[WMB.__init__] decay_half_life must be > 0.0, got {decay_half_life}."
            )
        if embed_dim < 1:
            raise ValueError(
                f"[WMB.__init__] embed_dim must be >= 1, got {embed_dim}."
            )

        self.capacity        : int   = capacity
        self.alpha           : float = alpha
        self.beta            : float = beta
        self.decay_half_life : float = decay_half_life
        self.embed_dim       : int   = embed_dim

        
        self.items: List[MemoryItem] = []

        _log.info(
            "WMB initialised | capacity=%d | α=%.3f | β=%.3f | τ=%.0fs",
            capacity, alpha, beta, decay_half_life,
        )

    

    def compute_salience(self, relevance: float, timestamp: float) -> float:
        
        now: float = time.time()
        age: float = max(0.0, now - timestamp)   # negative age guard for NTP/clock skew

        recency: float = math.exp(-age / self.decay_half_life)
        
        if not math.isfinite(relevance):
            relevance = 0.0
        relevance = float(np.clip(relevance, 0.0, 1.0))

        salience: float = self.alpha * recency + self.beta * relevance
        return float(np.clip(salience, 0.0, 1.0))

    def add(self, item: MemoryItem) -> None:
        
        if not isinstance(item, MemoryItem):
            raise TypeError(
                f"[WMB.add] Expected MemoryItem, got {type(item).__name__}."
            )
        _validate_embedding(item.embedding, dim=self.embed_dim, context="WMB.add")

        
        if not math.isfinite(item.relevance):
            item.relevance = 0.0
        item.relevance = float(np.clip(item.relevance, 0.0, 1.0))

        
        item.salience = self.compute_salience(item.relevance, item.timestamp)

        self.items.append(item)

        if len(self.items) > self.capacity:
            
            now: float = time.time()
            for it in self.items:
                it.salience = self.compute_salience(it.relevance, it.timestamp)

            
            self.items.sort(
                key=lambda x: (x.salience, x.timestamp, len(x.text)),
                reverse=True,
            )
            self.items = self.items[: self.capacity]

        assert len(self.items) <= self.capacity, (
            f"[WMB.add] Capacity invariant violated: "
            f"len={len(self.items)} > capacity={self.capacity}."
        )

    def retrieve(self, query_emb: np.ndarray, top_k: int = 3) -> List[MemoryItem]:
        
        if not self.items:
            return []

        _validate_embedding(query_emb, dim=self.embed_dim, context="WMB.retrieve")

        k: int = min(top_k, len(self.items))
        if k < 1:
            return []

        
        emb_matrix: np.ndarray = np.stack(
            [it.embedding for it in self.items], axis=0
        )  
        scores: np.ndarray = emb_matrix @ query_emb  

        
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[MemoryItem] = []
        for idx in top_indices:
            it = self.items[int(idx)]
            it.access_count += 1
            results.append(it)

        return results

    def __len__(self) -> int:
        
        return len(self.items)

    def __repr__(self) -> str:
        return (
            f"WMB("
            f"capacity={self.capacity}, "
            f"size={len(self.items)}, "
            f"α={self.alpha:.3f}, "
            f"β={self.beta:.3f}, "
            f"τ={self.decay_half_life:.0f}s"
            f")"
        )


# ── 

def _validate_embedding(emb: np.ndarray, dim: int, context: str) -> None:

    if not isinstance(emb, np.ndarray):
        raise TypeError(
            f"[{context}] embedding must be np.ndarray, "
            f"got {type(emb).__name__}."
        )
    if emb.shape != (dim,):
        raise ValueError(
            f"[{context}] embedding.shape {emb.shape} ≠ ({dim},). "
            f"Dimension mismatch — check EMBED_DIM and MemoryEncoder model."
        )
    if emb.dtype != np.float32:
        raise ValueError(
            f"[{context}] embedding.dtype {emb.dtype} ≠ float32. "
            f"LanceDB and FAISS require float32; float64 causes silent corruption."
        )


_ENCODER_SENTINEL = object()   


def make_item(
    text:      str,
    salience:  float = 0.5,
    relevance: float = 0.5,
    encoder=_ENCODER_SENTINEL,
) -> MemoryItem:
    
    _enc = encoder
    if _enc is _ENCODER_SENTINEL:
        try:
            import __main__ as _main
            _enc = getattr(_main, "encoder", None)
        except ImportError:
            _enc = None

    if _enc is not None:
        emb: np.ndarray = _enc.encode_passage(text)
    else:
        
        raise RuntimeError(
            "[make_item] No encoder available. Either:\n"
            "  (a) Pass encoder= explicitly: make_item(text, encoder=my_encoder)\n"
            "  (b) Ensure 'encoder' is defined in __main__ scope before calling.\n"
            "  A zero-vector fallback silently corrupts all cosine similarity scores."
        )

    return MemoryItem(
        text         = text,
        embedding    = emb,
        salience     = float(np.clip(salience, 0.0, 1.0)),
        timestamp    = time.time(),
        relevance    = float(np.clip(relevance, 0.0, 1.0)),
        access_count = 0,
    )


