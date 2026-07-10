"""Quality axis -- generative perplexity (and optional MAUVE).

Kept deliberately thin: quality is the control axis. The point of the pilot is
to show TSF can move quality (PPL down / MAUVE up) while the faithfulness axis
stays flat -- the "looks like progress but isn't" pattern the survey warns about.
"""
from __future__ import annotations

from typing import Optional, Sequence


def generative_ppl(texts: Sequence[str], scorer: str = "gpt2",
                   device: str = "cpu", batch_size: int = 8) -> float:
    """Mean perplexity of `texts` under an external causal LM scorer."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return float("nan")

    tok = AutoTokenizer.from_pretrained(scorer)
    model = AutoModelForCausalLM.from_pretrained(scorer).to(device).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    import math
    total_nll = total_tok = 0.0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = [t for t in texts[i:i + batch_size] if t.strip()]
            if not chunk:
                continue
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(device)
            out = model(**enc, labels=enc["input_ids"])
            ntok = enc["attention_mask"].sum().item()
            total_nll += out.loss.item() * ntok
            total_tok += ntok
    if total_tok == 0:
        return float("nan")
    return float(math.exp(total_nll / total_tok))


def mauve_score(generations: Sequence[str], references: Sequence[str]) -> Optional[float]:
    """Optional distributional quality via the `mauve-text` package."""
    try:
        import mauve
    except Exception:
        return None
    out = mauve.compute_mauve(p_text=list(generations), q_text=list(references),
                              verbose=False)
    return float(out.mauve)


def mock_quality(model, tsf: bool) -> dict:
    """Simulated PPL for the mock backbone."""
    return {"gen_ppl": model.ppl[tsf], "mauve": None}
