

import lancedb
import pyarrow as pa
import pandas as pd
import numpy as np
import math
import time
import os
import logging

logger = logging.getLogger(__name__)


_EMBED_DIM            = getattr(cfg, "EMBED_DIM",            1024)    if "cfg" in dir() else 1024
_LTMS_PATH            = getattr(cfg, "LTMS_PATH",            "./cira_ltm") if "cfg" in dir() else "./cira_ltm"
_LTMS_DECAY_S         = getattr(cfg, "LTMS_DECAY_S",         86400.0) if "cfg" in dir() else 86400.0
_LTMS_DECAY_THRESHOLD = getattr(cfg, "LTMS_DECAY_THRESHOLD", 0.05)    if "cfg" in dir() else 0.05
_LTMS_CHECKPOINT_PATH = getattr(cfg, "LTMS_CHECKPOINT_PATH", "/kaggle/working/ltms_checkpoint.parquet") if "cfg" in dir() else "/kaggle/working/ltms_checkpoint.parquet"


_LTMS_SCHEMA = pa.schema([
    pa.field("text",         pa.string()),
    pa.field("vector",       pa.list_(pa.float32(), _EMBED_DIM)),
    pa.field("timestamp",    pa.float64()),
    pa.field("access_count", pa.int32()),
])


class LTMS:
    
    _OVERFETCH: int = 3

    def __init__(self, path: str = _LTMS_PATH) -> None:
        
        self._path  = path
        self._db    = lancedb.connect(path)
        self._init_table()
        logger.info(f"[LTMS] Connected to {path}. Rows: {self.count()}")

    

    def _init_table(self) -> None:
        
        try:
            self.table = self._db.open_table("memories")
            logger.debug("[LTMS] Opened existing table 'memories'.")
        except Exception:
            # Table does not exist yet — create it with the strict schema.
            self.table = self._db.create_table("memories", schema=_LTMS_SCHEMA)
            logger.debug("[LTMS] Created new table 'memories'.")

    

    def store(self, text: str, embedding: np.ndarray) -> None:
        
        if embedding.dtype != np.float32:
            raise TypeError(
                f"[LTMS.store] embedding.dtype must be float32, got {embedding.dtype}. "
                f"Cast with: embedding.astype(np.float32)"
            )
        if embedding.shape != (_EMBED_DIM,):
            raise ValueError(
                f"[LTMS.store] embedding.shape must be ({_EMBED_DIM},), got {embedding.shape}. "
                f"Check EMBED_DIM or encoder output."
            )
        if not np.isfinite(embedding).all():
            raise ValueError(
                "[LTMS.store] embedding contains NaN or Inf values. "
                "Upstream encoder produced a corrupted vector — drop this entry."
            )

        
        self.table.add([{
            "text":         text,
            "vector":       embedding.tolist(),
            "timestamp":    time.time(),
            "access_count": 0,
        }])

    

    def retrieve(self, query_emb: np.ndarray, top_k: int = 5) -> list[dict]:
        
        if query_emb.dtype != np.float32:
            raise TypeError(
                f"[LTMS.retrieve] query_emb.dtype must be float32, got {query_emb.dtype}."
            )

        
        n_rows = self.count()
        if n_rows == 0:
            return []

        
        fetch_k = min(top_k * self._OVERFETCH, n_rows)

        
        try:
            raw_results: list[dict] = (
                self.table
                    .search(query_emb.tolist())
                    .limit(fetch_k)
                    .to_list()
            )
        except Exception as exc:
            logger.error(f"[LTMS.retrieve] ANN search failed: {exc}")
            return []

        
        filtered = self._decay_filter(raw_results)
        if not filtered:
            return []

        
        for r in filtered:
            
            l2: float = float(r.get("_distance", 0.0))

            
            cosine_equiv: float = float(np.clip(1.0 - (l2 ** 2) / 2.0, 0.0, 1.0))

            
            r["_combined_score"] = cosine_equiv * r["retention"]

        
        filtered.sort(key=lambda x: x["_combined_score"], reverse=True)
        return filtered[:top_k]

    def _decay_filter(self, results: list[dict]) -> list[dict]:
        
        now     = time.time()
        output: list[dict] = []

        for r in results:
            
            raw_ts = r.get("timestamp")
            if raw_ts is None:
                logger.warning("[LTMS._decay_filter] Result missing 'timestamp' — discarding.")
                continue

            try:
                ts: float = float(raw_ts)
            except (TypeError, ValueError):
                logger.warning(f"[LTMS._decay_filter] Unparseable timestamp '{raw_ts}' — discarding.")
                continue

            age: float = max(now - ts, 0.0)  

            
            retention: float = math.exp(-age / _LTMS_DECAY_S)

            if retention >= _LTMS_DECAY_THRESHOLD:
                r["retention"] = retention
                output.append(r)

        return output

    

    def to_memory_items(self, results: list[dict]) -> list:
        
        try:
            _MemoryItem = MemoryItem  
        except NameError as exc:
            raise RuntimeError(
                "[LTMS.to_memory_items] MemoryItem is not in scope. "
                "Ensure Cell 05 (wmb.py) has been executed before calling this method."
            ) from exc

        items = []
        for r in results:
            retention: float = float(r.get("retention", 0.5))

            
            raw_vec = r.get("vector", [])
            try:
                emb = np.array(raw_vec, dtype=np.float32)
            except (ValueError, TypeError) as exc:
                logger.warning(f"[LTMS.to_memory_items] Could not parse vector: {exc} — skipping.")
                continue

            if emb.shape != (_EMBED_DIM,):
                logger.warning(
                    f"[LTMS.to_memory_items] Skipping result with wrong vector shape {emb.shape}. "
                    f"Expected ({_EMBED_DIM},)."
                )
                continue

            items.append(_MemoryItem(
                text         = str(r.get("text", "")),
                embedding    = emb,
                salience     = retention,   # freshness as salience proxy
                timestamp    = float(r.get("timestamp", time.time())),
                relevance    = retention,   # stored for WMB dynamic eviction recompute
                access_count = int(r.get("access_count", 0)),
            ))

        return items

    

    def save_checkpoint(self, path: str = _LTMS_CHECKPOINT_PATH) -> None:
        
        n_rows = self.count()
        if n_rows == 0:
            logger.info(f"[LTMS] Table is empty — nothing to checkpoint at {path}.")
            return

        df = self.table.to_pandas()

        
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        df.to_parquet(path, index=False)
        print(f"[LTMS] Checkpoint saved: {n_rows} rows → {path}")

    def load_checkpoint(self, path: str = _LTMS_CHECKPOINT_PATH) -> None:
        
        if not os.path.exists(path):
            print(f"[LTMS] No checkpoint found at {path} — starting with empty store.")
            return

        df = pd.read_parquet(path)

        if df.empty:
            logger.info(f"[LTMS] Checkpoint at {path} is empty — nothing to load.")
            return

        
        n_existing = self.count()
        if n_existing > 0:
            existing_texts = set(self.table.to_pandas()["text"].tolist())
        else:
            existing_texts = set()

        records: list[dict] = df.to_dict(orient="records")

        valid_records: list[dict] = []
        skipped_dupes: int = 0

        for r in records:
            
            if r.get("text") in existing_texts:
                skipped_dupes += 1
                continue

            raw_vec = r.get("vector")
            if raw_vec is None:
                logger.warning("[LTMS.load_checkpoint] Record missing 'vector' — skipping.")
                continue
            try:
                vec_f32 = np.array(raw_vec, dtype=np.float32)
            except (ValueError, TypeError) as exc:
                logger.warning(f"[LTMS.load_checkpoint] Bad vector in record: {exc} — skipping.")
                continue

            if vec_f32.shape != (_EMBED_DIM,):
                logger.warning(
                    f"[LTMS.load_checkpoint] Wrong vector shape {vec_f32.shape} — skipping."
                )
                continue

            if not np.isfinite(vec_f32).all():
                logger.warning("[LTMS.load_checkpoint] NaN/Inf in vector — skipping.")
                continue

            r["vector"] = vec_f32.tolist()  
            valid_records.append(r)
            existing_texts.add(r.get("text", ""))  

        if not valid_records:
            print(f"[LTMS] All {len(records)} checkpoint rows already in table — nothing added.")
            return

        self.table.add(valid_records)
        print(f"[LTMS] Checkpoint loaded: {len(valid_records)} new rows "
              f"(skipped {skipped_dupes} duplicates) ← {path}")

    

    def count(self) -> int:
        
        return self.table.count_rows()

    def __repr__(self) -> str:
        return f"LTMS(path={self._path!r}, rows={self.count()}, decay_s={_LTMS_DECAY_S})"



ltms = LTMS()
print(f"LTMS loaded. {ltms}")


