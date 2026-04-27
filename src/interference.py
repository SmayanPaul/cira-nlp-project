

import itertools
import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


_NLI_MODEL     = getattr(cfg, "NLI_MODEL",      "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli") if "cfg" in dir() else "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
_NLI_THRESHOLD = getattr(cfg, "NLI_THRESHOLD",  0.70)  if "cfg" in dir() else 0.70
_NLI_BATCH     = getattr(cfg, "NLI_BATCH_SIZE", 32)    if "cfg" in dir() else 32
_NLI_MAX_LEN   = getattr(cfg, "NLI_MAX_LEN",    512)   if "cfg" in dir() else 512
_DEVICE        = getattr(cfg, "NLI_DEVICE",     "cuda:1" if torch.cuda.is_available() else "cpu") if "cfg" in dir() else ("cuda:1" if torch.cuda.is_available() else "cpu")  # B13 FIX: was SLM_DEVICE (cuda:0); DeBERTa must be on cuda:1


_MNLI_CONTRADICTION_IDX: int = 2
_MNLI_PATTERN = re.compile(r"mnli|snli|fever|anli|ling|wanli", re.IGNORECASE)


class InterferenceDetector:
     

    def __init__(
        self,
        model_name: str  = _NLI_MODEL,
        threshold:  float = _NLI_THRESHOLD,
    ) -> None:
        
        self.threshold       = threshold
        self.model_name      = model_name
        self.device          = _DEVICE
        self._chunk_size     = _NLI_BATCH
        self._max_len        = _NLI_MAX_LEN

        
        self._cache: Dict[frozenset, float] = {}

        
        logger.info(f"[InterferenceDetector] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        
        logger.info(f"[InterferenceDetector] Loading model on {self.device} ...")
        try:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=dtype,
            )
            self.model.to(self.device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            
            logger.warning(
                f"[InterferenceDetector] GPU load failed ({exc}). "
                f"Falling back to CPU with float32."
            )
            self.device = "cpu"
            self.model  = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )
            self.model.to(self.device)

        self.model.eval()

        
        self.contradiction_idx: int = self._resolve_contradiction_index()

        
        assert self.contradiction_idx is not None, (
            "[InterferenceDetector] BUG: contradiction_idx is None after resolution."
        )
        num_labels = self.model.config.num_labels
        assert 0 <= self.contradiction_idx < num_labels, (
            f"[InterferenceDetector] contradiction_idx={self.contradiction_idx} "
            f"is out of bounds for num_labels={num_labels}."
        )

        print(
            f"[InterferenceDetector] Ready. "
            f"device={self.device}, "
            f"contradiction_idx={self.contradiction_idx}, "
            f"id2label={self.model.config.id2label}, "
            f"threshold={self.threshold}"
        )

    

    def _resolve_contradiction_index(self) -> int:
        
        id2label: Dict[int, str] = self.model.config.id2label

        # ── Tier 1: explicit semantic label search ────────────────
        for idx, label in id2label.items():
            normalised = label.strip().lower()
            if "contradiction" in normalised:
                logger.info(
                    f"[InterferenceDetector] Tier 1 resolved: "
                    f"idx={idx}, label='{label}'"
                )
                return int(idx)

        
        all_labels_generic = all(
            re.fullmatch(r"LABEL_\d+", lbl.strip(), re.IGNORECASE)
            for lbl in id2label.values()
        )

        if all_labels_generic and _MNLI_PATTERN.search(self.model_name):
            logger.warning(
                f"[InterferenceDetector] Tier 1 failed (generic labels: {id2label}). "
                f"Tier 2 matched MNLI family in model name '{self.model_name}'. "
                f"Applying standard MNLI contradiction index={_MNLI_CONTRADICTION_IDX}."
            )
            
            if 0 <= _MNLI_CONTRADICTION_IDX < self.model.config.num_labels:
                return _MNLI_CONTRADICTION_IDX

        
        raise RuntimeError(
            f"[InterferenceDetector] FATAL: Cannot resolve contradiction index.\n"
            f"  model_name : {self.model_name}\n"
            f"  id2label   : {id2label}\n"
            f"  Tier 1 found no 'contradiction' substring in any label.\n"
            f"  Tier 2 regex did not match known MNLI/SNLI family.\n"
            f"  Halting — proceeding with an unknown index guarantees silent "
            f"data corruption. Verify the model's id2label config manually."
        )

    

    def _batch_nli(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        
        assert len(pairs) > 0, "_batch_nli called with empty pairs list."

        
        model_max = getattr(self.model.config, "max_position_embeddings", 512)
        eff_max   = min(self._max_len, model_max)

        
        premises    = [p[0] for p in pairs]
        hypotheses  = [p[1] for p in pairs]

        encodings = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=eff_max,
            return_tensors="pt",
        )

        input_ids      = encodings["input_ids"]        
        attention_mask = encodings["attention_mask"]  
        token_type_ids = encodings.get("token_type_ids") 

        
        all_probs: List[np.ndarray] = []
        n_pairs = len(pairs)

        for chunk_start in range(0, n_pairs, self._chunk_size):
            chunk_end = min(chunk_start + self._chunk_size, n_pairs)

            
            chunk_input_ids = input_ids[chunk_start:chunk_end].to(self.device)
            chunk_attn_mask = attention_mask[chunk_start:chunk_end].to(self.device)

            chunk_kwargs = {
                "input_ids":      chunk_input_ids,
                "attention_mask": chunk_attn_mask,
            }
            if token_type_ids is not None:
                chunk_kwargs["token_type_ids"] = (
                    token_type_ids[chunk_start:chunk_end].to(self.device)
                )

            with torch.no_grad():
                logits = self.model(**chunk_kwargs).logits   

            
            probs = F.softmax(logits, dim=-1)                

            
            contradiction_probs = probs[:, self.contradiction_idx]

            
            all_probs.append(contradiction_probs.cpu().float().numpy())

        
        result = np.concatenate(all_probs, axis=0)  

        
        assert result.shape == (n_pairs,), (
            f"_batch_nli output shape mismatch: got {result.shape}, "
            f"expected ({n_pairs},)."
        )
        assert np.all((result >= 0.0) & (result <= 1.0)), (
            f"_batch_nli produced probabilities outside [0, 1]. "
            f"min={result.min():.6f}, max={result.max():.6f}. "
            f"Softmax may not have applied correctly."
        )

        return result.astype(np.float32)

    

    def build_conflict_graph(
        self,
        memories: List[str],
    ) -> Dict[Tuple[int, int], float]:
        
        n = len(memories)
        if n < 2:
            return {}

        
        unique_pairs: List[Tuple[int, int]] = list(itertools.combinations(range(n), 2))

        
        normalised: List[str] = [self._normalise(m) for m in memories]

        to_run:        List[Tuple[int, int]] = []   
        to_run_texts:  List[Tuple[str, str]]  = []   

        for (i, j) in unique_pairs:
            text_a, text_b = memories[i], memories[j]
            norm_a, norm_b = normalised[i], normalised[j]
            cache_key      = frozenset({text_a, text_b})

            
            if norm_a == norm_b:
                self._cache[cache_key] = 0.0
                continue

            
            if cache_key in self._cache:
                continue

            
            to_run.append((i, j))
            to_run_texts.append((text_a, text_b))

        
        if to_run:
            
            bidirectional_pairs: List[Tuple[str, str]] = []
            for (text_a, text_b) in to_run_texts:
                bidirectional_pairs.append((text_a, text_b))  
                bidirectional_pairs.append((text_b, text_a))  

            
            all_scores = self._batch_nli(bidirectional_pairs)

            
            for k, (i, j) in enumerate(to_run):
                text_a, text_b = memories[i], memories[j]
                cache_key      = frozenset({text_a, text_b})

                fwd_score: float = float(all_scores[2 * k])      
                rev_score: float = float(all_scores[2 * k + 1])  

                
                sym_score: float = float(np.mean([fwd_score, rev_score]))

                self._cache[cache_key] = sym_score

        
        graph: Dict[Tuple[int, int], float] = {}
        for (i, j) in unique_pairs:
            text_a, text_b = memories[i], memories[j]
            cache_key      = frozenset({text_a, text_b})
            sym_score      = self._cache.get(cache_key, 0.0)

            if sym_score > self.threshold:
                graph[(i, j)] = sym_score

        return graph

    

    @staticmethod
    def _normalise(text: str) -> str:
        
        return unicodedata.normalize("NFC", text).lower().strip()

    @property
    def cache_size(self) -> int:
        
        return len(self._cache)

    def clear_cache(self) -> None:
        
        self._cache.clear()
        logger.info("[InterferenceDetector] Cache cleared.")

    def __repr__(self) -> str:
        return (
            f"InterferenceDetector("
            f"model='{self.model_name}', "
            f"threshold={self.threshold}, "
            f"contradiction_idx={self.contradiction_idx}, "
            f"device={self.device}, "
            f"cache_size={self.cache_size})"
        )


def _detector_init_cuda1(self, model_name, threshold):
    from transformers import (AutoTokenizer,
                              AutoModelForSequenceClassification)
    
    self.model_name  = model_name
    self.threshold   = threshold
    self.device      = torch.device("cuda:1")

    
    self._cache: dict = {}          

    
    self._chunk_size = _NLI_BATCH   
    self._max_len    = _NLI_MAX_LEN 

    
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model     = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to("cuda:1").eval()

    
    self.contradiction_idx = self._resolve_contradiction_index()

    print(f"[InterferenceDetector] cuda:1 | "
          f"{torch.cuda.memory_allocated(1)/1e9:.2f} GB used")
    print(f"  contradiction_idx={self.contradiction_idx} | "
          f"threshold={self.threshold} | cache=0")

InterferenceDetector.__init__ = _detector_init_cuda1
print("InterferenceDetector patched → cuda:1 ✅")


print("InterferenceDetector class defined.")
print("Load with: detector = InterferenceDetector()")




_original_detector_init = InterferenceDetector.__init__

def _patched_detector_init(self, model_name, threshold):
    
    self.model_name  = model_name
    self.threshold   = threshold
    self.device      = torch.device("cuda:1")

    
    self._cache: dict = {}          

    
    self._chunk_size = _NLI_BATCH   
    self._max_len    = _NLI_MAX_LEN 

    
    self.model     = _nli_model
    self.tokenizer = _nli_tokenizer
    self.model.eval()

    
    self.contradiction_idx = self._resolve_contradiction_index()

    print(f"[InterferenceDetector] Reusing pre-loaded model on {self.device}")
    print(f"  contradiction_idx={self.contradiction_idx} | "
          f"threshold={self.threshold} | cache=0")

InterferenceDetector.__init__ = _patched_detector_init
print("InterferenceDetector patched → cuda:1 ✅")


