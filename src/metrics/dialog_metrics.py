"""Episode-level metrics for the dialog policy-planning runners.

These work on a list of *episode records* — one per simulated dialog — each a dict with:

    {
      "did":          str,            # optional dialog/scenario id
      "task":         str,            # optional: "p4g" | "esc" | "cb"
      "success":      bool,           # did the system reach its goal (donate / solved / deal)?
      "num_turns":    int,            # number of system<->user turns in the dialog
      # CraigslistBargain only (used for SL):
      "deal_price":   float | None,   # agreed price if a deal was reached, else None
      "buyer_price":  float,          # the buyer's target price
      "seller_price": float,          # the seller's list price
    }

Metrics implemented:
  * SR  -- Success Rate: fraction of episodes that reached the goal within a turn limit.
  * AT  -- Average Turn: mean number of turns; PPDPP-style, failed/over-limit episodes
           count as the turn limit (set ``failures_as_max=False`` for "mean over successes").
  * SL  -- Sale-to-List ratio (CB only), from the buyer's perspective; higher == better
           deal, failed negotiations get SL = 0.
           SL = (deal_price - seller_list_price) / (buyer_target_price - seller_list_price)
"""
import math


def success_rate(episodes, max_turns=None):
    """SR: proportion of episodes that reached the goal within ``max_turns`` (None = no extra limit)."""
    if not episodes:
        return 0.0
    n_ok = 0
    for ep in episodes:
        if not ep.get("success"):
            continue
        if max_turns is not None and ep.get("num_turns", 0) > max_turns:
            continue
        n_ok += 1
    return n_ok / len(episodes)


def average_turn(episodes, max_turns=None, failures_as_max=True):
    """AT: mean number of turns to reach the goal.

    If ``failures_as_max`` and ``max_turns`` is set, episodes that fail or exceed the limit
    contribute ``max_turns`` (this is the PPDPP / common convention so AT is comparable across
    methods). Otherwise AT is averaged over successful (within-limit) episodes only.
    """
    use_cap = failures_as_max and max_turns is not None
    turns = []
    for ep in episodes:
        n = ep.get("num_turns", 0)
        reached = ep.get("success") and (max_turns is None or n <= max_turns)
        if reached:
            turns.append(min(n, max_turns) if max_turns is not None else n)
        elif use_cap:
            turns.append(max_turns)
    if not turns:
        return float("nan")
    return sum(turns) / len(turns)


def sale_to_list_ratio(episode, clip=(0.0, 1.0)):
    """SL for one CraigslistBargain episode (buyer's perspective; higher == better deal).

    Failed negotiations (no deal) return 0.0. Otherwise:
        SL = (deal_price - seller_list_price) / (buyer_target_price - seller_list_price)
    which is 0 when the buyer pays the full list price and 1 when the buyer gets its target.
    ``clip`` (default [0, 1]) bounds the result; pass ``clip=None`` for the raw ratio.
    """
    if not episode.get("success"):
        return 0.0
    deal = episode.get("deal_price")
    seller = episode.get("seller_price")
    buyer = episode.get("buyer_price")
    if deal is None or seller is None or buyer is None or buyer == seller:
        return 0.0
    sl = (deal - seller) / (buyer - seller)
    if clip is not None:
        lo, hi = clip
        sl = max(lo, min(hi, sl))
    return sl


def mean_sale_to_list_ratio(episodes, clip=(0.0, 1.0)):
    """Average SL over all CB episodes (failures contribute 0.0)."""
    if not episodes:
        return 0.0
    return sum(sale_to_list_ratio(ep, clip=clip) for ep in episodes) / len(episodes)


def _looks_like_cb(episodes):
    return any(("deal_price" in ep) or ("seller_price" in ep) or (ep.get("task") == "cb") for ep in episodes)


def compute_metrics(episodes, task=None, max_turns=None, failures_as_max=True, sl_clip=(0.0, 1.0)):
    """Compute the standard metric bundle for a list of episode records.

    Returns ``{"n", "SR", "AT", "max_turns", ["SL" if the episodes look like CraigslistBargain]}``.
    ``task`` (optional) forces whether SL is reported.
    """
    episodes = list(episodes)
    out = {
        "n": len(episodes),
        "max_turns": max_turns,
        "SR": success_rate(episodes, max_turns=max_turns),
        "AT": average_turn(episodes, max_turns=max_turns, failures_as_max=failures_as_max),
    }
    if task == "cb" or (task is None and _looks_like_cb(episodes)):
        out["SL"] = mean_sale_to_list_ratio(episodes, clip=sl_clip)
    return out


def format_metrics(metrics):
    """Pretty one-line-per-metric string for printing."""
    parts = [f"n={metrics.get('n', 0)}"]
    if metrics.get("max_turns") is not None:
        parts.append(f"max_turns={metrics['max_turns']}")
    if "SR" in metrics:
        parts.append(f"SR={metrics['SR']:.4f}")
    if "AT" in metrics:
        at = metrics["AT"]
        parts.append("AT=nan" if (isinstance(at, float) and math.isnan(at)) else f"AT={at:.3f}")
    if "SL" in metrics:
        parts.append(f"SL={metrics['SL']:.4f}")
    return "  ".join(parts)
