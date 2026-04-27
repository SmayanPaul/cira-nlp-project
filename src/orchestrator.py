

import time
import logging
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt = "%Y-%m-%dT%H:%M:%S",
)
_log = logging.getLogger("CognitiveSystem")



_MAX_PROMPT_TOKENS: int = 3800
_SYSTEM_PROMPT: str = (
    "You are a helpful assistant with access to a structured memory context. "
    "Use the provided context to answer accurately. "
    "If the context contains conflicting information it has already been flagged — "
    "prefer the fact without the [UNVERIFIED] prefix."
)


class CognitiveSystem:
    

    def __init__(self, config=None) -> None:
        self.cfg = config if config is not None else cfg

        _log.info(
            "CognitiveSystem.__init__ starting",
            extra={"slm_model": self.cfg.SLM_MODEL, "embed_dim": self.cfg.EMBED_DIM},
        )

        
        self.encoder = MemoryEncoder(self.cfg.ENCODER_MODEL)
        _startup_emb = self.encoder.encode_passage("warmup")
        assert _startup_emb.shape == (self.cfg.EMBED_DIM,), (
            f"[Unit 1] Encoder output shape {_startup_emb.shape} "
            f"!= expected ({self.cfg.EMBED_DIM},). Wrong model loaded."
        )
        assert _startup_emb.dtype == np.float32, (
            f"[Unit 1] Encoder dtype {_startup_emb.dtype} != float32."
        )
        _log.info("Unit 1 (MemoryEncoder) ready. dim=%d", self.cfg.EMBED_DIM)

        # ── Unit 2: Working Memory Buffer ────────────────────────────────────
        self.wmb = WMB(capacity=self.cfg.WMB_CAPACITY)
        _log.info("Unit 2 (WMB) ready. capacity=%d", self.cfg.WMB_CAPACITY)

        # ── Unit 3: Long-Term Memory Store ───────────────────────────────────
        self.ltms = LTMS(path=self.cfg.LTMS_PATH)
        _log.info("Unit 3 (LTMS) ready. rows=%d", self.ltms.count())

        # ── Unit 4: InterferenceDetector ─────────────────────────────────────
        self.detector = InterferenceDetector(
            model_name=self.cfg.NLI_MODEL,
            threshold=self.cfg.NLI_THRESHOLD,
        )
        _log.info(
            "Unit 4 (InterferenceDetector) ready. threshold=%.2f",
            self.cfg.NLI_THRESHOLD,
        )

        
        _log.info("Loading SLM tokenizer: %s", self.cfg.SLM_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.SLM_MODEL)

        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        _log.info("Loading SLM model: %s (fp16, device_map=cuda:0)", self.cfg.SLM_MODEL)
        self.slm = AutoModelForCausalLM.from_pretrained(
            self.cfg.SLM_MODEL,
            torch_dtype = self.cfg.SLM_DTYPE,
            device_map  = {"": "cuda:0"},       
        )
        self.slm.eval()
        _log.info("SLM loaded on cuda:0 | VRAM: %.2f GB",
                  torch.cuda.memory_allocated(0)/1e9)

        
        _get_slm_tokenizer()
        _log.info("Unit 6 reconstructor tokenizer primed.")

        _log.info("CognitiveSystem fully initialised.")

    

    def process(
        self,
        query  : str,
        policy : "PolicyType" = "confidence",
        verbose: bool         = False,
    ) -> str:
        
        t_start: float = time.time()
        assert policy in ("recency", "confidence", "merge-with-flag"), (
            f"Unknown policy: {policy!r}. "
            "Must be 'recency', 'confidence', or 'merge-with-flag'."
        )

        

        q_emb: np.ndarray | None = self._stage_encode(query, verbose)
        if q_emb is None:
            
            return (
                "I encountered an error processing your request. "
                "Please try again."
            )

        # ── Stage 2: Retrieve + deduplicate ──────────────────────────────────
        candidates: list = self._stage_retrieve(q_emb, verbose)
        # candidates is list[MemoryItem]. May be empty — handled in Stage 3.

        # ── Stage 3: Detect interference ─────────────────────────────────────
        conflict_graph: dict = self._stage_detect(candidates, verbose)
        # conflict_graph is dict[tuple[int,int], float]. May be empty.

        # ── Stage 4: Resolve conflicts ────────────────────────────────────────
        survivors: list = self._stage_resolve(candidates, conflict_graph, policy, verbose)
        # survivors is list[MemoryItem]. Falls back to candidates if resolution fails.

        # ── Stage 5: Reconstruct context ─────────────────────────────────────
        context: str = self._stage_reconstruct(survivors, q_emb, verbose)
        # context is a str. May be "" if no valid memories remained.

        # ── Stage 6: Assemble prompt + token-budget guard ────────────────────
        prompt_ids: torch.Tensor | None = self._stage_assemble_prompt(
            query, context, verbose
        )
        if prompt_ids is None:
            return (
                "I encountered an error assembling the prompt. "
                "Please try again."
            )

        # ── Stage 7: Generate response ───────────────────────────────────────
        response: str = self._stage_generate(prompt_ids, verbose)

        # ── Stage 8: Update memory (MUST be AFTER generate completes) ────────
        
        self._stage_update_memory(query, response, q_emb, verbose)

        elapsed: float = time.time() - t_start
        _log.info(
            "Turn completed. elapsed=%.2fs wmb_size=%d",
            elapsed,
            len(self.wmb),
        )
        if verbose:
            print(
                f"[CognitiveSystem] Turn done in {elapsed:.2f}s | "
                f"WMB={len(self.wmb)} | context_len={len(context)} chars"
            )

        return response

    

    def _stage_encode(
        self, query: str, verbose: bool
    ) -> "np.ndarray | None":
        
        if not isinstance(query, str):
            _log.warning("[Stage 1] query is not str (got %s). Coercing.", type(query))
            query = str(query)

        if not query.strip():
            
            _log.warning("[Stage 1] Empty query received — returning zero embedding.")
            return np.zeros(self.cfg.EMBED_DIM, dtype=np.float32)

        try:
            q_emb: np.ndarray = self.encoder.encode_query(query)

            
            if q_emb.shape != (self.cfg.EMBED_DIM,):
                _log.error(
                    "[Stage 1] FATAL: embedding shape %s != (%d,). "
                    "Wrong encoder model.",
                    q_emb.shape,
                    self.cfg.EMBED_DIM,
                )
                return None

            if q_emb.dtype != np.float32:
                _log.error(
                    "[Stage 1] FATAL: embedding dtype %s != float32.", q_emb.dtype
                )
                return None

            if verbose:
                norm = float(np.linalg.norm(q_emb))
                print(
                    f"[Stage 1] Encoded query. shape={q_emb.shape} "
                    f"norm={norm:.4f}"
                )

            _log.info("[Stage 1] Encode OK. dim=%d", self.cfg.EMBED_DIM)
            return q_emb

        except Exception as exc:
            _log.error("[Stage 1] Encoder raised: %s", exc, exc_info=True)
            return None

    def _stage_retrieve(
        self, q_emb: np.ndarray, verbose: bool
    ) -> list:
        
        wm_hits: list = []
        try:
            wm_hits = self.wmb.retrieve(q_emb, top_k=4)
            assert isinstance(wm_hits, list), (
                f"WMB.retrieve() must return list, got {type(wm_hits)}"
            )
            _log.info("[Stage 2] WMB retrieved %d items.", len(wm_hits))
        except Exception as exc:
            _log.error("[Stage 2] WMB.retrieve() failed: %s", exc, exc_info=True)
            wm_hits = []

        ltm_hits: list = []
        try:
            ltm_raw = self.ltms.retrieve(q_emb, top_k=6)
            assert isinstance(ltm_raw, list), (
                f"LTMS.retrieve() must return list, got {type(ltm_raw)}"
            )
            ltm_hits = self.ltms.to_memory_items(ltm_raw)
            assert isinstance(ltm_hits, list), (
                f"LTMS.to_memory_items() must return list, got {type(ltm_hits)}"
            )
            _log.info("[Stage 2] LTMS retrieved %d items.", len(ltm_hits))
        except Exception as exc:
            _log.error("[Stage 2] LTMS.retrieve() failed: %s", exc, exc_info=True)
            ltm_hits = []

        
        validated_wm: list = []
        for item in wm_hits:
            if self._validate_memory_item(item, stage="Stage 2 WMB"):
                validated_wm.append(item)

        validated_ltm: list = []
        for item in ltm_hits:
            if self._validate_memory_item(item, stage="Stage 2 LTMS"):
                validated_ltm.append(item)

        
        candidates: list = list(
            {m.text: m for m in validated_wm + validated_ltm}.values()
        )

        if verbose:
            print(
                f"[Stage 2] WM={len(validated_wm)} | "
                f"LTM={len(validated_ltm)} | "
                f"after dedup={len(candidates)}"
            )

        _log.info(
            "[Stage 2] Candidates after dedup: %d",
            len(candidates),
        )
        return candidates

    def _stage_detect(
        self, candidates: list, verbose: bool
    ) -> dict:
        
        if len(candidates) < 2:
            _log.info(
                "[Stage 3] Only %d candidate(s). Skipping NLI (need ≥2).",
                len(candidates),
            )
            if verbose:
                print(f"[Stage 3] Skipped NLI — only {len(candidates)} candidate(s).")
            return {}

        try:
            texts: list[str] = [m.text for m in candidates]
            conflict_graph: dict = self.detector.build_conflict_graph(texts)

            
            assert isinstance(conflict_graph, dict), (
                f"build_conflict_graph() must return dict, got {type(conflict_graph)}"
            )
            for key, val in conflict_graph.items():
                assert isinstance(key, tuple) and len(key) == 2, (
                    f"Conflict graph key must be tuple[int,int], got {key!r}"
                )
                assert isinstance(val, float), (
                    f"Conflict graph value must be float, got {type(val)}"
                )

            _log.info(
                "[Stage 3] Conflict graph: %d edge(s) detected.", len(conflict_graph)
            )
            if verbose and conflict_graph:
                print(
                    f"[Stage 3] Conflicts detected: "
                    f"{list(conflict_graph.keys())}"
                )
            elif verbose:
                print("[Stage 3] No conflicts detected.")

            return conflict_graph

        except Exception as exc:
            _log.error(
                "[Stage 3] InterferenceDetector raised: %s", exc, exc_info=True
            )
            
            return {}

    def _stage_resolve(
        self,
        candidates     : list,
        conflict_graph : dict,
        policy         : str,
        verbose        : bool,
    ) -> list:
        
        if not candidates:
            _log.info("[Stage 4] No candidates. Skipping conflict resolution.")
            if verbose:
                print("[Stage 4] Skipped — no candidates.")
            return []

        if not conflict_graph:
            _log.info(
                "[Stage 4] No conflict edges. Skipping resolution (fast path). "
                "Returning all %d candidates.",
                len(candidates),
            )
            if verbose:
                print(
                    f"[Stage 4] Fast path — no conflicts. "
                    f"All {len(candidates)} candidates survive."
                )
            return candidates

        try:
            survivors: list = resolve_conflicts(
                candidates, conflict_graph, policy=policy
            )

            
            assert isinstance(survivors, list), (
                f"resolve_conflicts() must return list, got {type(survivors)}"
            )
            assert len(survivors) >= 1, (
                "resolve_conflicts() returned an empty list — invariant violated."
            )
            for item in survivors:
                assert hasattr(item, "text"), (
                    "Survivor must be MemoryItem with .text field."
                )

            _log.info(
                "[Stage 4] Resolution complete. policy=%s | "
                "candidates=%d → survivors=%d",
                policy,
                len(candidates),
                len(survivors),
            )
            if verbose:
                print(
                    f"[Stage 4] Resolved. policy={policy!r} | "
                    f"{len(candidates)} → {len(survivors)} survivors"
                )

            return survivors

        except Exception as exc:
            _log.error(
                "[Stage 4] resolve_conflicts() raised: %s. "
                "Falling back to full candidate list.",
                exc,
                exc_info=True,
            )
            
            return candidates

    def _stage_reconstruct(
        self, survivors: list, q_emb: np.ndarray, verbose: bool
    ) -> str:
        
        if not survivors:
            _log.info("[Stage 5] No survivors. Returning empty context.")
            if verbose:
                print("[Stage 5] No survivors — empty context.")
            return ""

        try:
            triples: list[dict] = rank_survivors(survivors, q_emb)

            assert isinstance(triples, list), (
                f"rank_survivors() must return list, got {type(triples)}"
            )
            

            context: str = reconstruct_context(
                triples, max_tokens=self.cfg.MAX_CONTEXT_TOKENS
            )

            assert isinstance(context, str), (
                f"reconstruct_context() must return str, got {type(context)}"
            )

            _log.info(
                "[Stage 5] Context reconstructed. triples=%d | "
                "context_len=%d chars",
                len(triples),
                len(context),
            )
            if verbose:
                preview = context[:120].replace("\n", " ↵ ")
                print(
                    f"[Stage 5] Context ({len(context)} chars). "
                    f'Preview: "{preview}…"'
                )

            return context

        except Exception as exc:
            _log.error(
                "[Stage 5] Reconstruction raised: %s. Using empty context.",
                exc,
                exc_info=True,
            )
            return ""

    def _stage_assemble_prompt(
        self, query: str, context: str, verbose: bool
    ) -> "torch.Tensor | None":
        
        try:
            if context:
                user_content = f"Context:\n{context}\n\nQuestion: {query}"
            else:
                user_content = query

            messages: list[dict] = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ]

            prompt_ids: torch.Tensor = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt = True,
                return_tensors        = "pt",
            ).to(self.cfg.SLM_DEVICE)

            
            prompt_len: int = prompt_ids.shape[1]
            if prompt_len > _MAX_PROMPT_TOKENS:
                _log.warning(
                    "[Stage 6] Prompt too long (%d tokens). "
                    "Left-truncating to %d.",
                    prompt_len,
                    _MAX_PROMPT_TOKENS,
                )
                if verbose:
                    print(
                        f"[Stage 6] WARN: prompt={prompt_len} tokens → "
                        f"truncating to {_MAX_PROMPT_TOKENS}"
                    )
                prompt_ids = prompt_ids[:, -_MAX_PROMPT_TOKENS:]

            _log.info(
                "[Stage 6] Prompt assembled. tokens=%d context_empty=%s",
                prompt_ids.shape[1],
                not bool(context),
            )
            if verbose:
                print(f"[Stage 6] Prompt: {prompt_ids.shape[1]} tokens.")

            return prompt_ids

        except Exception as exc:
            _log.error(
                "[Stage 6] Prompt assembly raised: %s", exc, exc_info=True
            )
            return None

    def _stage_generate(
        self, prompt_ids: torch.Tensor, verbose: bool
    ) -> str:
        
        try:
            with torch.no_grad():
                out_ids: torch.Tensor = self.slm.generate(
                    prompt_ids,
                    max_new_tokens = self.cfg.SLM_MAX_NEW_TOKENS,
                    do_sample      = False,                         
                    pad_token_id   = self.tokenizer.eos_token_id,  
                )

            
            prompt_len: int = prompt_ids.shape[1]
            response: str = self.tokenizer.decode(
                out_ids[0][prompt_len:],
                skip_special_tokens = True,
            ).strip()

            _log.info(
                "[Stage 7] Generation complete. response_len=%d chars",
                len(response),
            )
            if verbose:
                preview = response[:80].replace("\n", " ")
                print(f'[Stage 7] Response ({len(response)} chars): "{preview}…"')

            return response

        except Exception as exc:
            _log.error(
                "[Stage 7] SLM.generate() raised: %s", exc, exc_info=True
            )
            return (
                "I was unable to generate a response due to an internal error. "
                "Please try again."
            )

    def _stage_update_memory(
        self,
        query    : str,
        response : str,
        q_emb    : np.ndarray,
        verbose  : bool,
    ) -> None:
        
        combined_text: str = f"Q: {query} A: {response}"

        try:
            combined_emb: np.ndarray = self.encoder.encode_passage(combined_text)
            relevance: float = float(np.dot(combined_emb, q_emb))   

            new_item = MemoryItem(
                text         = combined_text,
                embedding    = combined_emb,
                salience     = self.wmb.compute_salience(relevance, time.time()),
                timestamp    = time.time(),
                relevance    = relevance,    
                access_count = 0,
            )

            
            try:
                self.wmb.add(new_item)
                _log.info(
                    "[Stage 8] WMB updated. wmb_size=%d relevance=%.4f",
                    len(self.wmb),
                    relevance,
                )
            except Exception as exc:
                _log.error("[Stage 8] WMB.add() failed: %s", exc, exc_info=True)

            
            try:
                self.ltms.store(combined_text, combined_emb)
                _log.info("[Stage 8] LTMS updated. total_rows=%d", self.ltms.count())
            except Exception as exc:
                _log.error("[Stage 8] LTMS.store() failed: %s", exc, exc_info=True)

            if verbose:
                print(
                    f"[Stage 8] Memory updated. wmb={len(self.wmb)} | "
                    f"ltms={self.ltms.count()} | relevance={relevance:.4f}"
                )

        except Exception as exc:
            _log.error(
                "[Stage 8] Memory encoding failed: %s. "
                "This turn will not be stored in memory.",
                exc,
                exc_info=True,
            )

    

    def _validate_memory_item(self, item: object, stage: str) -> bool:
        
        try:
            # .text
            if not hasattr(item, "text") or not isinstance(item.text, str):
                _log.warning("[%s] Item has no valid .text field. Dropping.", stage)
                return False
            if not item.text.strip():
                _log.warning("[%s] Item has empty .text. Dropping.", stage)
                return False

            # .embedding
            if not hasattr(item, "embedding") or not isinstance(
                item.embedding, np.ndarray
            ):
                _log.warning(
                    "[%s] Item has no valid .embedding field. Dropping.", stage
                )
                return False
            if item.embedding.shape != (self.cfg.EMBED_DIM,):
                _log.warning(
                    "[%s] Item embedding shape %s != (%d,). Dropping.",
                    stage,
                    item.embedding.shape,
                    self.cfg.EMBED_DIM,
                )
                return False
            if item.embedding.dtype != np.float32:
                _log.warning(
                    "[%s] Item embedding dtype %s != float32. Dropping.",
                    stage,
                    item.embedding.dtype,
                )
                return False

            # .salience
            if not hasattr(item, "salience") or not isinstance(
                item.salience, (float, int)
            ):
                _log.warning(
                    "[%s] Item has no valid .salience field. Dropping.", stage
                )
                return False

            # .timestamp
            if not hasattr(item, "timestamp") or not isinstance(
                item.timestamp, (float, int)
            ):
                _log.warning(
                    "[%s] Item has no valid .timestamp field. Dropping.", stage
                )
                return False

            return True

        except Exception as exc:
            _log.error(
                "[%s] _validate_memory_item raised unexpectedly: %s. Dropping.",
                stage,
                exc,
            )
            return False

    

    def save_checkpoint(self, path: str | None = None) -> None:
        
        ckpt = path or self.cfg.LTMS_CHECKPOINT_PATH
        self.ltms.save_checkpoint(ckpt)
        _log.info("LTMS checkpoint saved → %s", ckpt)

    def load_checkpoint(self, path: str | None = None) -> None:
        
        ckpt = path or self.cfg.LTMS_CHECKPOINT_PATH
        self.ltms.load_checkpoint(ckpt)
        _log.info("LTMS checkpoint loaded ← %s (rows=%d)", ckpt, self.ltms.count())

    def __repr__(self) -> str:
        return (
            f"CognitiveSystem("
            f"wmb_size={len(self.wmb)}, "
            f"ltms_rows={self.ltms.count()}, "
            f"slm={self.cfg.SLM_MODEL!r}"
            f")"
        )


# ── 
CIRAOrchestrator = CognitiveSystem
CognitiveSystem.forward = CognitiveSystem.process

cira = CognitiveSystem()
print("CognitiveSystem ready.")
print(repr(cira))

print(f"\nFinal VRAM layout:")
print(f"  cuda:0 (Phi-3):       {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
print(f"  cuda:1 (BGE+DeBERTa): {torch.cuda.memory_allocated(1)/1e9:.2f} GB")

from transformers import AutoModelForCausalLM, AutoTokenizer as AT

_original_cs_init = CognitiveSystem.__init__

def _patched_cs_init(self, config=None):
    _original_cs_init(self, config)
    
    print(f"\nFinal VRAM layout:")
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i)/1e9
        reserv = torch.cuda.memory_reserved(i)/1e9
        print(f"  cuda:{i} → allocated={alloc:.2f} GB | reserved={reserv:.2f} GB")

CognitiveSystem.__init__ = _patched_cs_init

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer



_nli_tokenizer = AutoTokenizer.from_pretrained(cfg.NLI_MODEL)
_nli_model     = AutoModelForSequenceClassification.from_pretrained(
    cfg.NLI_MODEL,
    torch_dtype = torch.float16,  
).to("cuda:1").eval()

print(f"DeBERTa loaded on cuda:1")
print(f"  cuda:0 allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
print(f"  cuda:1 allocated: {torch.cuda.memory_allocated(1)/1e9:.2f} GB")

print("--- TRIGGERING THE FIREWALL ---")

ans = cira.forward("What mode did I just ask you to enter?", verbose=True)
print(f"\nFinal Answer: {ans}")

import json
import os

# Find the file
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        print(os.path.join(root, f))

with open("/kaggle/input/datasets/vanaparthisaikiran/cira-eval-500-json/cira_eval_500.json") as f:
    data = json.load(f)

CONTRADICTION_SCENARIOS = data["scenarios"]
print(f"Loaded {len(CONTRADICTION_SCENARIOS)} scenarios")

import sys, shutil
from pathlib import Path


EVAL_SRC = Path("/kaggle/input/datasets/vanaparthisaikiran/new-data")  # ← change to your dataset name
WORK     = Path("/kaggle/working")

for f in ["metrics.py", "baselines.py",
          "eval_c1_interference.py", "eval_c2_reconstructive.py",
          "eval_c3_dual_memory.py", "eval_stats.py"]:
    shutil.copy(EVAL_SRC / f, WORK / f)

sys.path.insert(0, str(WORK))
print("Eval files staged to /kaggle/working/ ✅")

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "rank_bm25", "bert_score", "evaluate", "scikit-learn"])


import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "datasets", "bert_score"])

from datasets import load_dataset
from bert_score import score as bert_score
import re, random, time
random.seed(42)


# ── 0a. Load SQuAD ───────────────────────────────────────────────
print("Loading SQuAD v1.1 validation split...")
squad = load_dataset("squad", split="validation")
print(f"  Total samples: {len(squad)}")


# ── 0b. Metric helpers ───────────────────────────────────────────

def normalize(text: str) -> str:
    
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def exact_match(pred: str, gold: str) -> int:
    
    p, g = normalize(pred), normalize(gold)
    if not g:
        return 0
    if p == g:
        return 1
    if len(g) > 2 and re.search(r"\b" + re.escape(g) + r"\b", p):
        return 1
    return 0

def run_bertscore(preds: list[str], golds: list[str]) -> float:
    
    if not preds:
        return 0.0
    _, _, F1 = bert_score(preds, golds,
                          lang="en",
                          model_type="distilbert-base-uncased",
                          verbose=False)
    return float(F1.mean())

def fresh_cira():
    
    cs = CognitiveSystem.__new__(CognitiveSystem)
    cs.cfg      = cfg
    cs.encoder  = cira.encoder      
    cs.wmb      = WMB(capacity=cfg.WMB_CAPACITY)
    cs.ltms     = LTMS(path=f"./cira_eval_ltms_{int(time.time()*1000)}")
    cs.detector = cira.detector     
    cs.tokenizer = cira.tokenizer   
    cs.slm       = cira.slm         
    _get_slm_tokenizer()            
    return cs

def seed(cs, text: str):
    
    emb = cs.encoder.encode_passage(text)
    item = MemoryItem(
        text=text, embedding=emb,
        salience=0.9, timestamp=time.time(),
        relevance=0.9, access_count=0
    )
    cs.wmb.add(item)
    cs.ltms.store(text, emb)

def query(cs, q: str) -> str:
    return cs.process(q, policy="confidence", verbose=False)




print("\n" + "═"*60)
print("TEST 1 — FAITHFULNESS  (n=30)")
print("═"*60)

N1 = 30

samples_t1 = random.sample(list(squad), N1)

em_cira_t1, em_base_t1 = [], []
preds_cira_t1, preds_base_t1, golds_t1 = [], [], []

for i, s in enumerate(samples_t1):
    context  = s["context"]
    question = s["question"]
    gold     = s["answers"]["text"][0]   

    
    cs = fresh_cira()
    
    chunks = [context[j:j+512] for j in range(0, len(context), 512)]
    for chunk in chunks:
        seed(cs, chunk)
    pred_cira = query(cs, question)

    
    cs_base = fresh_cira()             
    pred_base = query(cs_base, question)

    em_c = exact_match(pred_cira, gold)
    em_b = exact_match(pred_base, gold)
    em_cira_t1.append(em_c)
    em_base_t1.append(em_b)
    preds_cira_t1.append(pred_cira)
    preds_base_t1.append(pred_base)
    golds_t1.append(gold)

    print(f"  [{i+1:02d}] Q: {question[:60]}")
    print(f"        Gold : {gold}")
    print(f"        CIRA : {pred_cira[:80]}  → EM={em_c}")
    print(f"        Base : {pred_base[:80]}  → EM={em_b}")

bs_cira_t1 = run_bertscore(preds_cira_t1, golds_t1)
bs_base_t1 = run_bertscore(preds_base_t1, golds_t1)

print(f"\n── T1 RESULTS ──────────────────────────────")
print(f"  EM   CIRA={sum(em_cira_t1)/N1:.2%}  Baseline={sum(em_base_t1)/N1:.2%}  Δ=+{(sum(em_cira_t1)-sum(em_base_t1))/N1:.2%}")
print(f"  BS-F1 CIRA={bs_cira_t1:.3f}  Baseline={bs_base_t1:.3f}  Δ=+{bs_cira_t1-bs_base_t1:.3f}")




print("\n" + "═"*60)
print("TEST 2 — CONFLICT RESOLUTION  (n=20)")
print("═"*60)

N2 = 20
samples_t2 = random.sample(list(squad), N2)

em_cira_t2, em_base_t2 = [], []
preds_cira_t2, preds_base_t2, golds_t2 = [], [], []

for i, s in enumerate(samples_t2):
    question = s["question"]
    gold     = s["answers"]["text"][0]

    
    wrong_s  = random.choice(samples_t2)
    wrong    = wrong_s["answers"]["text"][0]
    if wrong == gold:                   
        wrong = "Unknown"

    correct_passage  = f"The answer to '{question}' is {gold}."
    incorrect_passage = f"The answer to '{question}' is {wrong}."

    cs = fresh_cira()

    
    emb_wrong = cs.encoder.encode_passage(incorrect_passage)
    old_item = MemoryItem(
        text=incorrect_passage, embedding=emb_wrong,
        salience=0.8, timestamp=time.time() - 3600,   
        relevance=0.8, access_count=0
    )
    cs.wmb.add(old_item)
    cs.ltms.store(incorrect_passage, emb_wrong)

    time.sleep(0.01)   

    
    emb_correct = cs.encoder.encode_passage(correct_passage)
    new_item = MemoryItem(
        text=correct_passage, embedding=emb_correct,
        salience=0.9, timestamp=time.time(),
        relevance=0.9, access_count=0
    )
    cs.wmb.add(new_item)
    cs.ltms.store(correct_passage, emb_correct)

    pred_cira = query(cs, question)

    
    cs_base = fresh_cira()
    seed(cs_base, incorrect_passage)
    seed(cs_base, correct_passage)
    pred_base = query(cs_base, question)   

    em_c = exact_match(pred_cira, gold)
    em_b = exact_match(pred_base, gold)
    em_cira_t2.append(em_c)
    em_base_t2.append(em_b)
    preds_cira_t2.append(pred_cira)
    preds_base_t2.append(pred_base)
    golds_t2.append(gold)

    print(f"  [{i+1:02d}] Q: {question[:55]}")
    print(f"        Correct  : {gold}")
    print(f"        Distractor: {wrong}")
    print(f"        CIRA : {pred_cira[:70]}  → EM={em_c}")
    print(f"        Base : {pred_base[:70]}  → EM={em_b}")

bs_cira_t2 = run_bertscore(preds_cira_t2, golds_t2)
bs_base_t2 = run_bertscore(preds_base_t2, golds_t2)

print(f"\n── T2 RESULTS ──────────────────────────────")
print(f"  EM   CIRA={sum(em_cira_t2)/N2:.2%}  Baseline={sum(em_base_t2)/N2:.2%}  Δ=+{(sum(em_cira_t2)-sum(em_base_t2))/N2:.2%}")
print(f"  BS-F1 CIRA={bs_cira_t2:.3f}  Baseline={bs_base_t2:.3f}  Δ=+{bs_cira_t2-bs_base_t2:.3f}")




print("\n" + "═"*60)
print("TEST 3 — SCALE ROBUSTNESS  (5 targets × 4 scale levels)")
print("═"*60)

SCALES  = [5, 10, 25, 50]
N_TGTS  = 5    
noise_pool = random.sample(list(squad), 200)   

results_t3 = {}   

for N in SCALES:
    print(f"\n  Scale N={N} noise memories")
    results_t3[N] = {"em": [], "bs_preds": [], "bs_golds": []}

    targets = random.sample(list(squad), N_TGTS)

    for j, tgt in enumerate(targets):
        question = tgt["question"]
        gold     = tgt["answers"]["text"][0]
        target_passage = f"The answer to '{question}' is {gold}."

        cs = fresh_cira()

        
        noise_samples = random.sample(noise_pool, N)
        for ns in noise_samples:
            noise_text = ns["context"][:256]   
            seed(cs, noise_text)

        
        seed(cs, target_passage)

        pred = query(cs, question)
        em = exact_match(pred, gold)
        results_t3[N]["em"].append(em)
        results_t3[N]["bs_preds"].append(pred)
        results_t3[N]["bs_golds"].append(gold)

        print(f"    [{j+1}] Q: {question[:55]}  Gold: {gold}  → EM={em}")

    bs = run_bertscore(results_t3[N]["bs_preds"], results_t3[N]["bs_golds"])
    results_t3[N]["bs"] = bs
    print(f"    EM@{N}={sum(results_t3[N]['em'])/N_TGTS:.2%}  BS-F1={bs:.3f}")




print("\n" + "═"*60)
print("FINAL SUMMARY")
print("═"*60)

print(f"""
TEST 1 — Faithfulness (Hallucination Reduction)
  CIRA  EM: {sum(em_cira_t1)/N1:.2%}   BERTScore: {bs_cira_t1:.3f}
  Base  EM: {sum(em_base_t1)/N1:.2%}   BERTScore: {bs_base_t1:.3f}
  Improvement: EM Δ={( sum(em_cira_t1)-sum(em_base_t1))/N1:+.2%}  BS Δ={bs_cira_t1-bs_base_t1:+.3f}

TEST 2 — Conflict Resolution (Reasoning Accuracy)
  CIRA  EM: {sum(em_cira_t2)/N2:.2%}   BERTScore: {bs_cira_t2:.3f}
  Base  EM: {sum(em_base_t2)/N2:.2%}   BERTScore: {bs_base_t2:.3f}
  Improvement: EM Δ={( sum(em_cira_t2)-sum(em_base_t2))/N2:+.2%}  BS Δ={bs_cira_t2-bs_base_t2:+.3f}

TEST 3 — Scale Robustness (Long-Context Handling)""")

for N in SCALES:
    em_n = sum(results_t3[N]["em"]) / N_TGTS
    print(f"  N={N:>3} noise → EM={em_n:.2%}  BS-F1={results_t3[N]['bs']:.3f}")

print("═"*60)


