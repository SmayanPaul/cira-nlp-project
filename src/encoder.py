
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


try:
    from config import CIRAConfig as _cfg  

    _ENCODER_MODEL: str = _cfg.ENCODER_MODEL
    _EMBED_DIM: int = _cfg.EMBED_DIM
    _BGE_QUERY_PREFIX: str = _cfg.BGE_QUERY_PREFIX
except (ImportError, AttributeError):
    _ENCODER_MODEL = "BAAI/bge-large-en-v1.5"
    _EMBED_DIM = 1024
    _BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


_CACHE_MAX_SIZE: int = 512     
_ENCODE_BATCH_SIZE: int = 32   
_NORM_TOLERANCE: float = 1e-4  

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | encoder | %(message)s",
)
_log = logging.getLogger(__name__)




class MemoryEncoder:
    

    def __init__(self, model_name: str = _ENCODER_MODEL) -> None:
        
        self.model_name: str = model_name
        self.dim: int = _EMBED_DIM

        try:
            self.model: SentenceTransformer = SentenceTransformer(
                model_name,
                device=None,  
            )
        except (OSError, RuntimeError, ValueError, Exception) as exc:
            raise RuntimeError(
                f"[MemoryEncoder.__init__] Failed to load '{model_name}'. "
                f"Verify network access or pre-downloaded cache. "
                f"Original error → {type(exc).__name__}: {exc}"
            ) from exc

        
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_max: int = _CACHE_MAX_SIZE

        
        _probe: np.ndarray = self._encode_single("encoder warmup probe")
        assert _probe.shape == (self.dim,), (
            f"[MemoryEncoder.__init__] Model '{model_name}' produced shape "
            f"{_probe.shape}; expected ({self.dim},). "
            f"Wrong model loaded or EMBED_DIM misconfigured."
        )
        assert _probe.dtype == np.float32, (
            f"[MemoryEncoder.__init__] Output dtype {_probe.dtype} ≠ float32. "
            f"LanceDB / FAISS will reject float64 vectors."
        )

        _log.info(
            "MemoryEncoder ready | model=%s | dim=%d | cache_max=%d",
            model_name, self.dim, self._cache_max,
        )

    

    def encode_query(self, text: str) -> np.ndarray:
        
        _validate_text(text, context="encode_query")
        prefixed: str = _BGE_QUERY_PREFIX + text
        emb: np.ndarray = self._encode_single(prefixed)
        _assert_invariants(emb, dim=self.dim, context="encode_query")
        return emb

    def encode_passage(self, text: str) -> np.ndarray:
        
        _validate_text(text, context="encode_passage")

        if text in self._cache:
            self._cache.move_to_end(text)   
            return self._cache[text]

        emb: np.ndarray = self._encode_single(text)
        _assert_invariants(emb, dim=self.dim, context="encode_passage")
        self._lru_insert(text, emb)
        return emb

    def encode_passages_batch(self, texts: List[str]) -> np.ndarray:
        
        if not isinstance(texts, list):
            raise TypeError(
                f"[encode_passages_batch] Expected list[str], "
                f"got {type(texts).__name__}."
            )
        if len(texts) == 0:
            raise ValueError(
                "[encode_passages_batch] 'texts' must contain at least one element."
            )
        for idx, t in enumerate(texts):
            _validate_text(t, context=f"encode_passages_batch[{idx}]")

        
        missing: List[str] = list(
            dict.fromkeys(t for t in texts if t not in self._cache)
        )

        if missing:
            
            raw: np.ndarray = self.model.encode(
                missing,
                normalize_embeddings=True,
                batch_size=_ENCODE_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)

            assert raw.shape == (len(missing), self.dim), (
                f"[encode_passages_batch] Raw batch shape {raw.shape} ≠ "
                f"({len(missing)}, {self.dim})."
            )
            assert raw.dtype == np.float32, (
                f"[encode_passages_batch] Raw batch dtype {raw.dtype} ≠ float32."
            )

            
            norms: np.ndarray = np.linalg.norm(raw, axis=1, keepdims=True)
            raw = raw / np.where(norms > 0, norms, 1.0)

            for t, e in zip(missing, raw):
                _assert_invariants(e, dim=self.dim, context="encode_passages_batch[item]")
                self._lru_insert(t, e)

        
        result: np.ndarray = np.stack(
            [self._cache[t] for t in texts], axis=0
        )
        assert result.shape == (len(texts), self.dim), (
            f"[encode_passages_batch] Output shape {result.shape} ≠ "
            f"({len(texts)}, {self.dim})."
        )
        assert result.dtype == np.float32
        return result

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        
        assert a.shape == b.shape, (
            f"[cosine_sim] Shape mismatch: {a.shape} vs {b.shape}. "
            f"Both vectors must have identical shape."
        )
        return float(np.dot(a, b))

    

    def _encode_single(self, text: str) -> np.ndarray:
        
        emb: np.ndarray = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        
        norm: float = float(np.linalg.norm(emb))
        if norm > 0.0:
            emb = emb / norm

        return emb

    def _lru_insert(self, key: str, value: np.ndarray) -> None:
        
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return

        if len(self._cache) >= self._cache_max:
            evicted_key, _ = self._cache.popitem(last=False)  # LRU eviction
            _log.debug(
                "Cache eviction | key_prefix='%.40s…'", evicted_key
            )

        self._cache[key] = value

    

    @property
    def cache_size(self) -> int:
        
        return len(self._cache)

    def clear_cache(self) -> None:
        
        self._cache.clear()
        _log.info("MemoryEncoder cache cleared.")

    def __repr__(self) -> str:
        return (
            f"MemoryEncoder("
            f"model={self.model_name!r}, "
            f"dim={self.dim}, "
            f"cache={self.cache_size}/{self._cache_max}"
            f")"
        )



_orig_encoder_init = MemoryEncoder.__init__

def _encoder_init_cuda1(self, model_name):
    from collections import OrderedDict
    from sentence_transformers import SentenceTransformer

    
    self.model_name: str = model_name
    self.dim: int = _EMBED_DIM

    try:
        self.model = SentenceTransformer(
            model_name,
            device="cuda:1",
            model_kwargs={"torch_dtype": torch.float16},
        )
        print(f"[MemoryEncoder] cuda:1 | "
              f"{torch.cuda.memory_allocated(1)/1e9:.2f} GB used")
    except Exception as exc:
        raise RuntimeError(
            f"[MemoryEncoder.__init__] Failed to load '{model_name}'. "
            f"Original error → {exc}"
        ) from exc

    
    self._cache: OrderedDict = OrderedDict()
    self._cache_max: int = _CACHE_MAX_SIZE

    
    _probe = self._encode_single("encoder warmup probe")
    assert _probe.shape == (self.dim,), (
        f"[MemoryEncoder patch] shape {_probe.shape} != ({self.dim},). "
        f"Wrong model or EMBED_DIM mismatch."
    )
    assert _probe.dtype == np.float32, (
        f"[MemoryEncoder patch] dtype {_probe.dtype} != float32."
    )
    _log.info("MemoryEncoder patched → cuda:1 | dim=%d | cache_max=%d",
              self.dim, self._cache_max)

MemoryEncoder.__init__ = _encoder_init_cuda1
print("MemoryEncoder patched → cuda:1 ✅")




def _validate_text(text: str, context: str) -> None:
    
    if not isinstance(text, str):
        raise TypeError(
            f"[{context}] Expected str, got {type(text).__name__}: {text!r}"
        )
    if not text.strip():
        raise ValueError(
            f"[{context}] Input text is empty or whitespace-only. "
            f"Encode requires at least one non-whitespace character."
        )


def _assert_invariants(
    emb: np.ndarray,
    dim: int,
    context: str,
) -> None:
    
    assert emb.shape == (dim,), (
        f"[{context}] shape {emb.shape} ≠ ({dim},). "
        f"Dimension mismatch — check EMBED_DIM and model config."
    )
    assert emb.dtype == np.float32, (
        f"[{context}] dtype {emb.dtype} ≠ float32. "
        f"LanceDB and FAISS require float32; float64 causes silent corruption."
    )
    norm: float = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) <= _NORM_TOLERANCE, (
        f"[{context}] L2-norm {norm:.8f} deviates from 1.0 "
        f"by {abs(norm - 1.0):.2e}, tolerance is {_NORM_TOLERANCE}. "
        f"normalize_embeddings=True may have been bypassed."
    )

