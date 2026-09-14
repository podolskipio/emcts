"""Response parsing and history windowing shared by the p4g / esc / cb players.

Everything here is a pure function over a generation-model response or a DialogSession.
The prompts themselves stay in each task's own module, where they can be read next to the
game they belong to.
"""


def parse_das(data, valid_das) -> list:
    """The dialog acts the model wrote as ``[<act>]``, one per generated sequence.

    Sequences with no bracketed act, or an act outside ``valid_das``, are dropped: callers
    histogram what is left, so an unparseable sample is a sample that does not vote.
    """
    das = []
    for resp in data:
        text = resp["generated_text"].strip()
        start, end = text.find("["), text.find("]")
        if start == -1 or end == -1:
            continue
        da = text[start + 1:end].strip()
        if da in valid_das:
            das.append(da)
    return das


def split_da(utt: str, valid_das, default: str) -> "tuple[str, str]":
    """Split a ``[<act>] utterance`` reply into ``(act, utterance)``.

    The tag is stripped whenever it is present, even when it names an act outside
    ``valid_das`` -- otherwise a misspelt act would leak into the stored history and back
    into every later prompt. An untagged reply is returned unchanged, tagged ``default``.
    """
    start, end = utt.find("["), utt.find("]")
    if start == -1 or end == -1:
        return default, utt
    da = utt[start + 1:end]
    utt = utt.replace(f"[{da}]", "", 1).strip()
    return (da if da in valid_das else default), utt


def recent_turns(exp, max_hist_num_turns: int = -1):
    """Yield ``(index, turn)`` over the last ``max_hist_num_turns`` turns of ``exp``.

    A turn is a system/user pair, so the window is applied on ``index // 2``. The index is
    the position in the full history, which callers rely on to look ahead to the next turn.
    ``max_hist_num_turns <= 0`` keeps the whole history.
    """
    skip = max(0, len(exp) - max_hist_num_turns) if max_hist_num_turns > 0 else 0
    for i, turn in enumerate(exp):
        if (i // 2) >= skip:
            yield i, turn
