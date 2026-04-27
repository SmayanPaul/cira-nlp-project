

import re
import unicodedata
import spacy
import numpy as np
from transformers import AutoTokenizer


try:
    
    from config import cfg
except ImportError:
    
    from __main__ import cfg


_nlp: spacy.language.Language | None = None
_slm_tokenizer: AutoTokenizer | None = None


def _get_nlp() -> spacy.language.Language:
    
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(cfg.SPACY_MODEL)
    return _nlp


def _get_slm_tokenizer() -> AutoTokenizer:
    
    global _slm_tokenizer
    if _slm_tokenizer is None:
        _slm_tokenizer = AutoTokenizer.from_pretrained(cfg.SLM_MODEL)
    return _slm_tokenizer




def count_tokens(text: str) -> int:
    
    if not text:
        return 0
    return len(_get_slm_tokenizer().encode(text, add_special_tokens=False))



_MAX_FACT_CHARS: int = 1_200


_REDACTED: str = "[REDACTED]"


_ZERO_WIDTH_RE: re.Pattern = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f"
    r"\u2060\u2061\u2062\u2063\u2064"
    r"\u206a\u206b\u206c\u206d\u206e\u206f"
    r"\ufeff\u00ad]"
)


_INJECTION_PATTERNS: list[re.Pattern] = [

    
    re.compile(
        r"\b(ignore|disregard|forget|bypass|override|dismiss)\s+"
        r"(all\s+)?(previous|prior|above|earlier|existing)\s+"
        r"(instructions?|rules?|directives?|prompts?|constraints?|guidelines?)\b",
        re.IGNORECASE,
    ),
    
    re.compile(
        r"\b(ignore|disregard|forget|bypass)\s+"
        r"(your|the|all|any)\s+"
        r"(instructions?|rules?|directives?|system\s+prompt|guidelines?)\b",
        re.IGNORECASE,
    ),

    
    re.compile(r"\bsystem\s*(prompt|override|instruction|message|command)\s*:", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+in\s+\w+\s+mode\b", re.IGNORECASE),
    re.compile(r"\benable\s+(unrestricted|unsafe|developer|admin)\s+mode\b", re.IGNORECASE),

    
    re.compile(r"\bact\s+as\s+(a\s+|an\s+)?\w[\w\s]{0,30}\b", re.IGNORECASE),
    re.compile(r"\bpretend\s+(to\s+be|you\s+are)\b", re.IGNORECASE),
    re.compile(r"\bbehave\s+as\s+(a\s+|an\s+)?\w[\w\s]{0,30}\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(?!(a\s+helpful|an\s+assistant))", re.IGNORECASE),
    re.compile(r"\bswitch\s+(to|into)\s+\w+\s+persona\b", re.IGNORECASE),

    
    re.compile(
        r"\b(print|reveal|show|output|display|repeat|dump|return|expose)\s+"
        r"(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?|rules?|directives?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\btell\s+(me\s+)?your\s+(system\s+)?(instructions?|prompt|rules?)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(are\s+)?your\s+(system\s+)?(instructions?|prompt|rules?)\b", re.IGNORECASE),

    
    re.compile(
        r"<\/?\s*(system|instructions?|prompt|override|admin|root|superuser|command)\s*>",
        re.IGNORECASE,
    ),

    
    re.compile(
        r"\b(do\s+not|don'?t|never)\s+(follow|obey|respect|adhere\s+to)\s+"
        r"(any|the|your)\s+(rules?|instructions?|guidelines?|policies?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bstop\s+(following|obeying|respecting)\s+(your|the|any)\s+"
        r"(rules?|instructions?|guidelines?)\b",
        re.IGNORECASE,
    ),

    
    re.compile(r"\b(su|super\s*user)\s+mode\b", re.IGNORECASE),
    re.compile(r"\badmin\s+(mode|access|override|privileges?)\b", re.IGNORECASE),
    re.compile(r"\broot\s+(access|mode|override)\b", re.IGNORECASE),

    
    re.compile(r"\b(aWdub3Jl|SWdub3Jl|c3lzdGVt|U3lzdGVt)\b"),
]


_MULTI_REDACTED_RE: re.Pattern = re.compile(r"(\[REDACTED\]\s*){2,}")


def _sanitize_text(text: str) -> str:
    
    if not isinstance(text, str):
        return ""

    # Step 1: Remove zero-width / invisible Unicode
    text = _ZERO_WIDTH_RE.sub("", text)

    # Step 2: Unicode NFC normalisation — prevents homoglyph obfuscation
    text = unicodedata.normalize("NFC", text)

    # Step 3: Adversarial pattern redaction
    for pattern in _INJECTION_PATTERNS:
        text = pattern.sub(_REDACTED, text)

    # Step 4: Collapse runs of [REDACTED] to a single marker
    text = _MULTI_REDACTED_RE.sub(_REDACTED + " ", text)

    # Step 5: Hard character-length truncation
    if len(text) > _MAX_FACT_CHARS:
        text = text[:_MAX_FACT_CHARS].rstrip() + "…"

    return text.strip()


def _is_valid_memory(text: str) -> bool:
    
    if not text or not text.strip():
        return False
    payload = text.replace(_REDACTED, "").strip()
    return len(payload) > 0




def extract_subject(text: str) -> str:
    
    nlp = _get_nlp()
    doc = nlp(text)
    chunks = list(doc.noun_chunks)   
    if not chunks:
        
        words = text.split()
        return " ".join(words[: min(3, len(words))]) or "unknown"
    
    return chunks[0].root.text




def rank_survivors(
    survivors : list,         
    query_emb : np.ndarray,    
) -> list[dict]:
    
    if not survivors:
        return []

    seen_facts: set[str] = set()    
    triples: list[dict] = []

    for m in survivors:
       
        raw_text = getattr(m, "text", None)
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue

        
        sanitized = _sanitize_text(raw_text)

        
        if not _is_valid_memory(sanitized):
            continue

        
        if sanitized in seen_facts:
            continue
        seen_facts.add(sanitized)

        
        query_relevance: float = float(np.dot(m.embedding, query_emb))
        intrinsic_salience: float = float(m.salience)
        combined_salience: float = float(np.clip(
            query_relevance * intrinsic_salience, 0.0, 1.0
        ))

        
        subject: str = extract_subject(sanitized)

        triples.append(
            {
                "subject":  subject,
                "fact":     sanitized,
                "salience": combined_salience,
            }
        )

    
    triples.sort(key=lambda x: x["salience"], reverse=True)
    return triples



_MAX_MEMORY_ENTRIES: int = 10


def reconstruct_context(
    triples   : list[dict],
    max_tokens: int = None,   
) -> str:
    
    if max_tokens is None:
        max_tokens = cfg.MAX_CONTEXT_TOKENS

    if not triples:
        return ""

    lines: list[str] = []
    used_tokens: int  = 0
    entries_added: int = 0

    for t in triples:
        # Entry-count cap — enforced before token check to fail fast
        if entries_added >= _MAX_MEMORY_ENTRIES:
            break

        subject  : str   = t.get("subject", "unknown")
        fact     : str   = t.get("fact", "")
        salience : float = float(t.get("salience", 0.0))

        # Drop entries that somehow arrived without a usable fact string
        if not fact or not fact.strip():
            continue

        # Assemble the line in the standard format
        line: str  = f"[{subject}]: {fact}"
        cost: int  = count_tokens(line)

        # Token budget enforcement — strict hard cut
        if used_tokens + cost > max_tokens:
            break

        lines.append(line)
        used_tokens  += cost
        entries_added += 1

    return "\n".join(lines)


print("MemoryReconstructor loaded.")


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

