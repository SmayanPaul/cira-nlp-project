

import math
import time
import logging
from typing import Dict, List, Literal, Tuple

logger = logging.getLogger(__name__)


_DECAY_S: float = getattr(cfg, "LTMS_DECAY_S", 86400.0) if "cfg" in dir() else 86400.0


PolicyType    = Literal["recency", "confidence", "merge-with-flag"]
ConflictGraph = Dict[Tuple[int, int], float]

_VALID_POLICIES = ("recency", "confidence", "merge-with-flag")


_UNVERIFIED_PREFIX = "[UNVERIFIED — conflicts with newer memory]: "


_FLAG_SALIENCE_MULTIPLIER: float = 0.3




def _recency_score(m: "MemoryItem") -> float:  # noqa: F821
    
    return m.timestamp


def _confidence_score(m: "MemoryItem") -> Tuple[float, int]:  # noqa: F821
    
    age: float           = max(time.time() - m.timestamp, 0.0)   # clamp ≥ 0
    recency_decay: float = math.exp(-age / _DECAY_S)
    primary: float       = m.salience * recency_decay
    return (primary, m.access_count)




def resolve_conflicts(
    memories:       "List[MemoryItem]",  
    conflict_graph: ConflictGraph,
    policy:         PolicyType = "confidence",
) -> "List[MemoryItem]":                 
    
    if policy not in _VALID_POLICIES:
        raise ValueError(
            f"[resolve_conflicts] Unknown policy '{policy}'. "
            f"Must be one of: {_VALID_POLICIES}."
        )
    if not memories:
        raise ValueError(
            "[resolve_conflicts] memories list is empty. "
            "At least one MemoryItem is required."
        )

    
    if not conflict_graph:
        return memories

    
    if policy == "merge-with-flag":
        return _resolve_merge_with_flag(memories, conflict_graph)

    
    return _resolve_greedy(memories, conflict_graph, policy)




def _resolve_greedy(
    memories:       "List[MemoryItem]",  
    conflict_graph: ConflictGraph,
    policy:         PolicyType,
) -> "List[MemoryItem]":                  
    
    survivors: set = set(range(len(memories)))

    
    sorted_conflicts = sorted(
        conflict_graph.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )

    for (i, j), _score in sorted_conflicts:
        
        if i not in survivors or j not in survivors:
            continue

        
        if policy == "recency":
            
            loser: int = i if _recency_score(memories[i]) < _recency_score(memories[j]) else j
        else:  
            loser = i if _confidence_score(memories[i]) < _confidence_score(memories[j]) else j

        survivors.discard(loser)
        logger.debug(
            f"[resolve_conflicts] Conflict ({i},{j}) score={_score:.4f}: "
            f"loser={loser} eliminated by policy='{policy}'."
        )

    
    assert len(survivors) >= 1, (
        f"[resolve_conflicts] BUG: all {len(memories)} memories eliminated. "
        f"conflict_graph={conflict_graph}. This indicates a malformed "
        f"conflict_graph (e.g., self-loops or index out of range)."
    )

    
    return [memories[idx] for idx in sorted(survivors)]


def _resolve_merge_with_flag(
    memories:       "List[MemoryItem]",   
    conflict_graph: ConflictGraph,
) -> "List[MemoryItem]":                  
    
    result: list = list(memories)

    try:
        _MemoryItem = MemoryItem  
    except NameError as exc:
        raise RuntimeError(
            "[resolve_conflicts] MemoryItem is not in scope. "
            "Ensure Cell 05 (wmb.py) has been executed first."
        ) from exc

    for (i, j), _score in conflict_graph.items():
        
        ci = _confidence_score(memories[i])
        cj = _confidence_score(memories[j])
        loser_idx: int = i if ci < cj else j

        old = result[loser_idx]

        
        result[loser_idx] = _MemoryItem(
            text         = _UNVERIFIED_PREFIX + old.text,
            embedding    = old.embedding,               
            salience     = old.salience * _FLAG_SALIENCE_MULTIPLIER,
            timestamp    = old.timestamp,               
            relevance    = old.relevance,               
            access_count = old.access_count,            
        )

        logger.debug(
            f"[resolve_conflicts] merge-with-flag: conflict ({i},{j}) "
            f"score={_score:.4f}, loser={loser_idx} flagged."
        )

    return result


print("ConflictResolver loaded.")


