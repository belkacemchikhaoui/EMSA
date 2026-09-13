from baselines.cosrec import CosRec
from baselines.emma import EMMA
from baselines.emorec import EmoRec
from baselines.mm_dialog import MMDialog
from baselines.redial_bert import ReDialBert
from baselines.unimind import UniMIND
from baselines.visualbert_crs import VisualBertCRS

BASELINE_REGISTRY = {
    "visualbert_crs": VisualBertCRS,
    "mm_dialog": MMDialog,
    "emorec": EmoRec,
    "cosrec": CosRec,
    "redial_bert": ReDialBert,
    "unimind": UniMIND,
    "emma": EMMA,
}


def build_baseline(name: str, hidden_size: int, text_encoder_name: str, dropout: float,
                    num_emotion_categories: int = 6, vocab_size: int = 30522,
                    num_layers: int = 4, num_heads: int = 8):
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(BASELINE_REGISTRY)}")
    cls = BASELINE_REGISTRY[name]
    kwargs = dict(hidden_size=hidden_size, dropout=dropout)
    if name in ("visualbert_crs", "mm_dialog", "redial_bert", "unimind", "emma"):
        kwargs.update(text_encoder_name=text_encoder_name, num_layers=num_layers,
                       num_heads=num_heads, vocab_size=vocab_size)
    if name == "emorec":
        kwargs.update(text_encoder_name=text_encoder_name, num_emotion_categories=num_emotion_categories)
    if name == "emma":
        kwargs.update(num_emotion_categories=num_emotion_categories)
    return cls(**kwargs)
