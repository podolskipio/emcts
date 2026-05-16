"""Hugging Face Hub dataset loaders for the offline runners.

Triggered by ``--data hf:<repo>[:<config_or_split>[:<split>]]`` (see ``runners/_common``).
Requires ``pip install datasets``. Each loader returns the same normalized list shape as
the local-file readers in ``runners/_common``:

    [{"id": str, "scenario": tuple, "turns": [{"sys_da","sys_utt","usr_da","usr_utt"}, ...]}, ...]

Field-name parsing is defensive: HF column schemas drift between versions, so we look up
columns by a small set of plausible names rather than hard-coding one. If a loader can't
make sense of the rows it raises ``ValueError`` with the columns it actually saw.

Recommended dataset paths (per the README):
  * P4G: ``hf:spawn99/PersuasionForGood:FullDialog``
  * ESC: ``hf:thu-coai/esconv:validation``
  * CB:  ``hf:stanfordnlp/craigslist_bargains:validation``
"""
import json
import logging

from games import PersuasionGame, EmotionalSupportGame, CBGame

logger = logging.getLogger(__name__)

HF_PREFIX = "hf:"

# kept in sync with runners/_common._P4G_USER_DA_MAP (duplicated here to avoid a circular import)
_P4G_USER_DA_MAP = {
	"disagree-donation": PersuasionGame.U_NoDonation,
	"negative-reaction-to-donation": PersuasionGame.U_NegativeReaction,
	"positive-reaction-to-donation": PersuasionGame.U_PositiveReaction,
	"agree-donation": PersuasionGame.U_Donate,
}
_CB_DEAL_STRATEGIES = {"agree", "affirm", "accept"}


# ---------------------------------------------------------------------------
# path parsing + dataset loading
# ---------------------------------------------------------------------------
def _parse_hf(path):
	"""Parse ``hf:<repo>[:<a>[:<b>]]`` into ``(repo, a, b)`` with missing tokens as ``None``."""
	parts = path[len(HF_PREFIX):].split(":")
	if len(parts) == 1:
		return parts[0], None, None
	if len(parts) == 2:
		return parts[0], parts[1], None
	return parts[0], parts[1], ":".join(parts[2:])


def _load_hf(path):
	"""Load the requested split, accepting either ``hf:repo:split`` or ``hf:repo:config:split``.

	When the middle token is ambiguous (config vs split), tries it as a split on the default
	config first, then falls back to treating it as a config name (taking the first split).
	"""
	try:
		from datasets import load_dataset
	except ImportError as e:
		raise ImportError(
			"hf:<...> data paths require the `datasets` package — install with `pip install datasets`."
		) from e

	repo, second, third = _parse_hf(path)
	if third is not None:
		return load_dataset(repo, second, split=third)
	if second is None:
		ds = load_dataset(repo)
		return ds[next(iter(ds.keys()))]
	# disambiguate split vs config: try split on the default config first
	try:
		return load_dataset(repo, split=second)
	except Exception:
		ds = load_dataset(repo, second)
		return ds[next(iter(ds.keys()))]


def _get(row, *names, default=None):
	"""Return the first present-and-non-None field from ``names`` (case-insensitive)."""
	lower = {k.lower(): k for k in row.keys()}
	for n in names:
		k = lower.get(n.lower())
		if k is not None and row[k] is not None:
			return row[k]
	return default


# ---------------------------------------------------------------------------
# tiny segment helpers (mirror of _common._collapse_segments / _pair_segments,
# inlined to keep hf_loaders importable from _common without a circular import)
# ---------------------------------------------------------------------------
def _collapse(raw_turns):
	segs = []
	for t in raw_turns:
		speaker = t.get("speaker")
		text = (t.get("text") or "").strip()
		da = t.get("strategy")
		if not text:
			continue
		if segs and segs[-1][0] == speaker:
			ps, pt, pd = segs[-1]
			segs[-1] = (ps, f"{pt} {text}".strip(), da or pd)
		else:
			segs.append((speaker, text, da))
	return segs


def _pair(segs, sys_key="sys", usr_key="usr"):
	out = []
	i = 0
	while i + 1 < len(segs):
		if segs[i][0] != sys_key:
			i += 1
			continue
		if segs[i + 1][0] != usr_key:
			i += 1
			continue
		out.append((segs[i], segs[i + 1]))
		i += 2
	return out


# ---------------------------------------------------------------------------
# P4G (spawn99/PersuasionForGood:FullDialog)
# ---------------------------------------------------------------------------
def read_p4g_hf(path, system_dialog_acts):
	"""Load a P4G-style HF dataset.

	Handles two row shapes:
	  * one row per dialog with a ``dialog`` list of ``{speaker,text,strategy?}``;
	  * one row per turn keyed by a dialog id (``Dialogue_ID`` / ``Unit_id`` / ``dialog_id``).
	Strategy labels are usually absent in the HF dump — we default ``sys_da`` to ``"other"``
	and ``usr_da`` to ``U_Neutral``.
	"""
	ds = _load_hf(path)

	# shape 1: each row is already a full dialog
	cols = set(ds.column_names)
	if "dialog" in {c.lower() for c in cols}:
		out = []
		for i, row in enumerate(ds):
			did = _get(row, "id", "Dialogue_ID", "Unit_id", "dialog_id", default=f"p4g-hf-{i}")
			segs = _collapse(_get(row, "dialog") or [])
			turns = []
			for (_, ts, das), (_, tu, dau) in _pair(segs):
				sys_da = das if (das and das in system_dialog_acts) else "other"
				usr_da = _P4G_USER_DA_MAP.get(dau or "", PersuasionGame.U_Neutral)
				turns.append({"sys_da": sys_da, "sys_utt": ts, "usr_da": usr_da, "usr_utt": tu})
			if turns:
				out.append({"id": str(did), "scenario": (), "turns": turns})
		return out

	# shape 2: row-per-turn — group by dialog id, alternate speakers based on a role/speaker column
	did_keys = ("Dialogue_ID", "B2", "Unit_id", "dialog_id", "did")
	utt_keys = ("Utterance", "utterance", "B4", "text", "Sentence")
	role_keys = ("Speaker", "speaker", "role", "Role", "agent")

	groups = {}
	for row in ds:
		did = _get(row, *did_keys)
		utt = _get(row, *utt_keys)
		if did is None or not utt:
			continue
		groups.setdefault(str(did), []).append(row)

	if not groups:
		raise ValueError(
			f"read_p4g_hf: could not find dialog-id / utterance columns in {path!r}; "
			f"saw columns {sorted(ds.column_names)}"
		)

	out = []
	for did, rows in groups.items():
		raw = []
		for r in rows:
			role_raw = (_get(r, *role_keys) or "").strip().lower()
			speaker = "sys" if role_raw.startswith(("er", "persuader", "sys", "system", "agent_0", "0")) else "usr"
			raw.append({"speaker": speaker, "text": _get(r, *utt_keys) or "", "strategy": None})
		segs = _collapse(raw)
		turns = []
		for (_, ts, _), (_, tu, _) in _pair(segs):
			turns.append({
				"sys_da": "other", "sys_utt": ts,
				"usr_da": PersuasionGame.U_Neutral, "usr_utt": tu,
			})
		if turns:
			out.append({"id": did, "scenario": (), "turns": turns})
	return out


# ---------------------------------------------------------------------------
# ESConv (thu-coai/esconv)
# ---------------------------------------------------------------------------
def read_esc_hf(path, system_dialog_acts):
	"""Load thu-coai/esconv-style HF rows.

	Each row commonly has a single ``text`` column with a JSON-encoded session
	(``{emotion_type, problem_type, situation, dialog:[{speaker,content/text,strategy?}]}``).
	Falls back to columns named ``dialog`` / ``conversations`` / ``emotion_type`` directly.
	"""
	ds = _load_hf(path)
	out = []
	for i, row in enumerate(ds):
		# decode the JSON-string row format (most common on the Hub for ESConv)
		obj = None
		text_col = _get(row, "text")
		if isinstance(text_col, str) and text_col.strip().startswith("{"):
			try:
				obj = json.loads(text_col)
			except json.JSONDecodeError:
				obj = None
		if obj is None:
			obj = row

		dialog = _get(obj, "dialog", "conversations", "utterances") or []
		emotion = _get(obj, "emotion_type", "emotion") or "anxiety"
		problem = _get(obj, "problem_type", "problem") or "ongoing stress"

		raw = []
		for t in dialog:
			speaker = (t.get("speaker") or t.get("role") or "").lower()
			# ESConv role names: supporter / seeker
			if speaker in ("sys", "supporter", "system"):
				sp = "sys"
			elif speaker in ("usr", "seeker", "user"):
				sp = "usr"
			else:
				sp = speaker
			raw.append({
				"speaker": sp,
				"text": t.get("text") or t.get("content") or "",
				"strategy": t.get("strategy") or t.get("annotation"),
			})

		segs = _collapse(raw)
		turns = []
		for (_, ts, das), (_, tu, _) in _pair(segs):
			sys_da = das if das in system_dialog_acts else EmotionalSupportGame.S_Others
			turns.append({
				"sys_da": sys_da, "sys_utt": ts,
				"usr_da": EmotionalSupportGame.U_FeelTheSame, "usr_utt": tu,
			})
		if not turns:
			continue
		did = _get(obj, "id", "dialog_id", default=f"esc-hf-{i}")
		out.append({"id": str(did), "scenario": (emotion, problem), "turns": turns})
	return out


# ---------------------------------------------------------------------------
# CraigslistBargain (stanfordnlp/craigslist_bargains)
# ---------------------------------------------------------------------------
def _cb_speaker(role_raw, agent_idx):
	"""Map a CB role/agent index to ``sys`` (buyer) / ``usr`` (seller)."""
	if isinstance(role_raw, str):
		low = role_raw.lower()
		if "buy" in low:
			return "sys"
		if "sell" in low:
			return "usr"
	# fall back to agent index — agent 0 == buyer in the HF dump
	return "sys" if int(agent_idx) == 0 else "usr"


def _cb_strategy(act):
	"""Pull a strategy/intent label out of one CB dialogue act."""
	if act is None:
		return None
	if isinstance(act, str):
		return act
	if isinstance(act, dict):
		return act.get("intent") or act.get("strategy") or act.get("action")
	return None


def read_cb_hf(path, system_dialog_acts):
	"""Load stanfordnlp/craigslist_bargains-style HF rows.

	Each row contains parallel lists ``utterance`` / ``dialogue_acts`` / ``agent`` plus an
	``agent_info`` dict (with per-agent ``Role`` and ``Target`` prices) and ``items`` (with
	``Title`` and a list ``Price``). Falls back to row-level ``dialog`` if the parallel
	layout isn't present.
	"""
	ds = _load_hf(path)
	out = []
	for i, row in enumerate(ds):
		dialog = _get(row, "dialog")
		raw = []
		if dialog:
			for t in dialog:
				raw.append({
					"speaker": t.get("speaker"),
					"text": t.get("text") or "",
					"strategy": t.get("strategy"),
				})
		else:
			utts = _get(row, "utterance", "utterances") or []
			acts = _get(row, "dialogue_acts", "acts") or []
			agents = _get(row, "agent", "agents") or []
			for k, utt in enumerate(utts):
				role_raw = None
				ag_idx = agents[k] if k < len(agents) else (k % 2)
				agent_info = _get(row, "agent_info") or {}
				roles = agent_info.get("Role") if isinstance(agent_info, dict) else None
				if isinstance(roles, list) and ag_idx < len(roles):
					role_raw = roles[ag_idx]
				speaker = _cb_speaker(role_raw, ag_idx)
				strategy = _cb_strategy(acts[k] if k < len(acts) else None)
				raw.append({"speaker": speaker, "text": utt or "", "strategy": strategy})

		segs = _collapse(raw)
		turns = []
		for (_, ts, das), (_, tu, dau) in _pair(segs):
			sys_da = das if (das and das in system_dialog_acts) else CBGame.S_Inquire
			usr_da = CBGame.U_Deal if (dau in _CB_DEAL_STRATEGIES) else CBGame.U_No_deal
			turns.append({"sys_da": sys_da, "sys_utt": ts, "usr_da": usr_da, "usr_utt": tu})
		if not turns:
			continue

		# scenario: (item_name, buyer_desc, buyer_price, seller_desc, seller_price)
		items = _get(row, "items") or {}
		agent_info = _get(row, "agent_info") or {}
		title = ""
		desc = ""
		prices = []
		if isinstance(items, dict):
			title = (items.get("Title") or [""])[0] if isinstance(items.get("Title"), list) else (items.get("Title") or "")
			desc_v = items.get("Description")
			desc = (desc_v[0] if isinstance(desc_v, list) and desc_v else (desc_v or "")) if desc_v else ""
			prices_v = items.get("Price")
			if isinstance(prices_v, list):
				prices = prices_v
		buyer_target = seller_list = None
		if isinstance(agent_info, dict):
			targets = agent_info.get("Target")
			if isinstance(targets, list) and len(targets) >= 2:
				buyer_target, seller_list = targets[0], targets[1]
		if seller_list is None and prices:
			seller_list = prices[0]
		scenario = (title, desc, buyer_target, desc, seller_list)
		did = _get(row, "id", "dialog_id", default=f"cb-hf-{i}")
		out.append({"id": str(did), "scenario": scenario, "turns": turns})
	return out
