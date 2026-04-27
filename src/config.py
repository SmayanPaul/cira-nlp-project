import os
import torch
import gc

class CIRAConfig:
    # Graceful fallback for devices
    _num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    _default_device = "cuda:0" if _num_gpus > 0 else "cpu"
    _secondary_device = "cuda:1" if _num_gpus > 1 else _default_device

    # Unit 1 - BGE encoder -> cuda:1 (or fallback)
    ENCODER_MODEL    = "BAAI/bge-large-en-v1.5"
    EMBED_DIM        = 1024
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    ENCODER_DEVICE   = _secondary_device

    # Unit 2
    WMB_CAPACITY          = 15   
    WMB_ALPHA             = 0.4
    WMB_BETA              = 0.6
    WMB_DECAY_HALF_LIFE_S = 3600.0

    # Unit 3
    LTMS_PATH              = "./cira_ltm"
    LTMS_DECAY_S           = 604800.0 
    LTMS_DECAY_THRESHOLD   = 0.05
    LTMS_CHECKPOINT_PATH   = "./ltms_checkpoint.parquet"

    # Unit 4 - DeBERTa NLI -> cuda:1 (or fallback)
    NLI_MODEL       = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    NLI_THRESHOLD   = 0.70
    NLI_BATCH_SIZE  = 16
    NLI_MAX_LEN     = 512
    NLI_DEVICE      = _secondary_device

    # Unit 6
    SPACY_MODEL        = "en_core_web_trf"
    MAX_CONTEXT_TOKENS = 3200

    # Unit 7 - Phi-3 SLM -> cuda:0 ALONE
    SLM_MODEL          = "microsoft/Phi-3-mini-4k-instruct"
    SLM_MAX_NEW_TOKENS = 256
    SLM_DTYPE          = torch.float16 if torch.cuda.is_available() else torch.float32
    SLM_DEVICE         = _default_device

cfg = CIRAConfig()

# Optional: Set PyTorch allocator config if using CUDA
if torch.cuda.is_available():
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
