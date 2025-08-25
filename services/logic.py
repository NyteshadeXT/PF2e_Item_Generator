# services/logic.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json, random, re, sqlite3
import pandas as pd
from pathlib import Path
from services.utils import to_gp, normalize_str_columns, apply_adjustments_probabilistic, apply_materials_probabilistic, bump_rarity, add_price, parse_potency_rank, within_range
from services.db import load_items


print(">>> USING services.logic from", __file__)

CONFIG = json.loads(Path(__file__).resolve().parent.parent.joinpath("config.json").read_text(encoding="utf-8"))
GROUPS = CONFIG.get("source_table_groups", {})

def _format_price(gp_value: float | None) -> str:
    """Format a gp float into PF2e denominations (gp/sp/cp)."""
    if gp_value is None:
        return ""
    cp_total = int(round(float(gp_value) * 100))  # 1 gp = 100 cp
    gp, rem = divmod(cp_total, 100)
    sp, cp = divmod(rem, 10)
    parts = []
    if gp: parts.append(f"{gp} gp")
    if sp: parts.append(f"{sp} sp")
    if cp: parts.append(f"{cp} cp")
    return " ".join(parts) if parts else "0 gp"


def _group(key: str, default: list[str]) -> list[str]:
    vals = GROUPS.get(key)
    return vals if isinstance(vals, list) and vals else default


@dataclass
class PickCounts:
    mundane: int
    armor: int
    weapons: int
    magic: int

# -------- helpers --------

def _level_window(party_level: int) -> Tuple[int, int]:
    caps   = CONFIG.get("level_caps", {"min": 1, "max": 20})
    spread = CONFIG.get("level_spread", {"min_offset": -3, "max_offset": 1})
    lo = max(caps["min"], min(caps["max"], party_level + spread["min_offset"]))
    hi = max(caps["min"], min(caps["max"], party_level + spread["max_offset"]))
    if hi < lo: hi = lo
    return lo, hi

def _normalize_pair(v, default=(0, 0)) -> tuple[int, int]:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        lo, hi = int(v[0]), int(v[1])
    else:
        lo, hi = default
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi

def _counts_block(shop_type: str, shop_size: str) -> dict:
    """
    Return the band block for this shop_type+size, falling back to size-only CONFIG['counts'].
    """
    st = (shop_type or "").strip().lower()
    sz = (shop_size or "medium").strip().lower()
    by_shop = (CONFIG.get("counts_by_shop") or {})
    # exact shop_type + size
    block = (by_shop.get(st) or {}).get(sz)
    if not block:
        # fallback: size-only counts
        block = (CONFIG.get("counts") or {}).get(sz, {})
    return block or {}

def _counts_for_size(shop_type: str, shop_size: str) -> PickCounts:
    block = _counts_block(shop_type, shop_size)
    r = random.Random()
    m_lo, m_hi = _normalize_pair(block.get("mundane"))
    a_lo, a_hi = _normalize_pair(block.get("armor"))
    w_lo, w_hi = _normalize_pair(block.get("weapons"))
    g_lo, g_hi = _normalize_pair(block.get("magic"))
    return PickCounts(
        mundane=r.randint(m_lo, m_hi),
        armor=r.randint(a_lo, a_hi),
        weapons=r.randint(w_lo, w_hi),
        magic=r.randint(g_lo, g_hi),
    )

def _counts_for_size_type(shop_type: str, shop_size: str, item_type: str) -> int:
    block = _counts_block(shop_type, shop_size)  # uses counts_by_shop if present, else counts
    band = block.get((item_type or "").strip().lower(), [0, 0])
    lo, hi = _normalize_pair(band)
    return random.randint(lo, hi)

def _counts_for_specific_magic(shop_type_or_size: str | None = None,
                               maybe_shop_size: str | None = None) -> int:
    import random

    if maybe_shop_size is None:
        # legacy call: only size provided
        sz = (shop_type_or_size or "medium").strip().lower()
        band = (CONFIG.get("specific_magic_counts") or {}).get(sz, [0, 0])
    else:
        st = (shop_type_or_size or "").strip().lower()
        sz = (maybe_shop_size or "medium").strip().lower()
        by_shop = (CONFIG.get("specific_magic_counts_by_shop") or {})
        band = (by_shop.get(st) or {}).get(sz)
        if band is None:
            band = (CONFIG.get("specific_magic_counts") or {}).get(sz, [0, 0])

    lo, hi = _normalize_pair(band)
    return random.randint(lo, hi)

def _filter_source_tables(df: pd.DataFrame, source_tables) -> pd.DataFrame:
    # If no source_table column, don't filter on it
    if "source_table" not in df.columns:
        return df

    # Normalize requested tables
    if isinstance(source_tables, str):
        source_tables = [source_tables]

    def _norm(s: str) -> str:
        s = str(s or "").lower().strip()
        return "".join(ch for ch in s if ch.isalnum())

    wanted = {_norm(s) for s in source_tables if str(s).strip()}
    col_norm = df["source_table"].astype(str).map(_norm)

    # 1) exact normalized match
    mask = col_norm.isin(wanted)

    # 2) substring fallbacks per family
    if not mask.any():
        want_weapon = any(k.startswith("weapon") for k in wanted) or any("weapon" in k for k in wanted)
        want_armor  = any(k.startswith("armor")  for k in wanted) or any("armor"  in k for k in wanted)
        want_shield = any(k.startswith("shield") for k in wanted) or any("shield" in k for k in wanted)

        sub_masks = []
        if want_weapon:
            sub_masks.append(col_norm.str.contains("weapon", na=False))
        if want_armor:
            sub_masks.append(col_norm.str.contains("armor", na=False))
        if want_shield:
            sub_masks.append(col_norm.str.contains("shield", na=False))

        if sub_masks:
            m = sub_masks[0]
            for sm in sub_masks[1:]:
                m = m | sm
            mask = m

    # 3) category fallback if source_table is messy/blank
    if not mask.any() and "category" in df.columns:
        cat_norm = df["category"].astype(str).str.strip().str.lower()
        want_weapon = any(k.startswith("weapon") for k in wanted)
        want_armor  = any(k.startswith("armor")  for k in wanted)
        want_shield = any(k.startswith("shield") for k in wanted)

        m = None
        if want_weapon:
            m = (cat_norm == "weapon")
        if want_armor:
            m = (m | (cat_norm == "armor")) if m is not None else (cat_norm == "armor")
        if want_shield:
            m = (m | (cat_norm == "shield")) if m is not None else (cat_norm == "shield")
        if m is not None:
            mask = m

    return df[mask] if mask.any() else df

# --- shop type matching (exact + fuzzy) ---

def _normalize_shop(s: str) -> str:
    s = str(s or "").lower().strip()
    return "".join(ch for ch in s if ch.isalnum())

def _apply_shop_type(pool: pd.DataFrame, shop_type: str, strict: bool = False) -> pd.DataFrame:
    """
    Filter by shop_type with graceful fallbacks:
      1) exact (normalized) match
      2) alias map from CONFIG['shop_type_aliases']
      3) substring / startswith matches
      4) fuzzy (difflib) with configurable threshold
    Set strict=True to force exact-only (previous behavior).
    """
    if not shop_type or "shop_type" not in pool.columns or pool.empty:
        return pool

    # prepare
    from difflib import SequenceMatcher
    threshold = float(CONFIG.get("shop_type_fuzzy_threshold", 0.84))
    aliases   = CONFIG.get("shop_type_aliases", {})

    target_raw = str(shop_type).strip()
    target_norm = _normalize_shop(target_raw)

    col = pool["shop_type"].dropna().astype(str).map(str.strip)
    vals = col.unique().tolist()
    vals_norm = {_normalize_shop(v): v for v in vals}

    # 1) exact normalized match
    if target_norm in vals_norm:
        chosen = vals_norm[target_norm]
        return pool[col.str.lower() == chosen.lower()]

    if strict:
        # In strict mode, no exact match means no items for this shop.
        # This prevents unrelated categories from leaking into specialized shops
        # (e.g., Tattooist showing mundane/weapons/armor).
        return pool.iloc[0:0]

    # 2) alias map
    alias_hit = None
    if aliases:
        aliases_norm = {_normalize_shop(k): v for k, v in aliases.items() if v}
        if target_norm in aliases_norm:
            alias_hit = aliases_norm[target_norm]
            alias_norm = _normalize_shop(alias_hit)
            if alias_norm in vals_norm:
                chosen = vals_norm[alias_norm]
                return pool[col.str.lower() == chosen.lower()]

    # 3) substring / startswith
    starts = [v for v in vals if _normalize_shop(v).startswith(target_norm)]
    if len(starts) == 1:
        return pool[col.str.lower() == starts[0].lower()]
    contains = [v for v in vals if target_norm in _normalize_shop(v)]
    if len(contains) == 1:
        return pool[col.str.lower() == contains[0].lower()]

    # 4) fuzzy
    def score(a, b): return SequenceMatcher(None, a, b).ratio()
    best = None
    best_sc = 0.0
    for v in vals:
        sc = score(target_norm, _normalize_shop(v))
        if sc > best_sc:
            best_sc, best = sc, v

    if best and best_sc >= threshold:
        return pool[col.str.lower() == best.lower()]

    return pool

def _apply_shop_type_exact(pool: pd.DataFrame, shop_type: str) -> pd.DataFrame:
    return _apply_shop_type(pool, shop_type, strict=True)


def _level_bounds_for(item_type: str, party_level: int) -> tuple[int, int]:
    it = (item_type or "").lower()
    if it in ("mundane", "weapons", "armor", "materials"):
        return (0, party_level + 1)

    caps   = CONFIG.get("level_caps", {"min": 1, "max": 20})
    spread = CONFIG.get("level_spread", {"min_offset": -3, "max_offset": 1})
    lo = max(caps["min"], min(caps["max"], party_level + spread["min_offset"]))
    hi = max(caps["min"], min(caps["max"], party_level + spread["max_offset"]))
    if hi < lo: hi = lo
    return (lo, hi)


def _aggregate_items(rows_base, rows_crit, disposition: str) -> list[dict]:
    """
    Combine base + critical rows, collapse duplicates by (name, critical),
    sum quantities, and apply disposition to per-unit price. Robust to bad inputs.
    """
    # --- defensive normalization ---
    def _normalize_rows(rows):
        try:
            import pandas as _pd
        except Exception:
            _pd = None

        if isinstance(rows, dict):
            return [rows]
        if _pd is not None and isinstance(rows, _pd.DataFrame):
            return rows.to_dict(orient="records")
        if _pd is not None and isinstance(rows, _pd.Series):
            return [rows.to_dict()]
        if isinstance(rows, list):
            out = []
            for r in rows:
                if isinstance(r, dict):
                    out.append(r)
                elif _pd is not None and isinstance(r, _pd.Series):
                    out.append(r.to_dict())
            return out
        return []

    rows_base = _normalize_rows(rows_base)
    rows_crit = _normalize_rows(rows_crit)

    from collections import defaultdict
    bucket: dict[tuple[str, bool], dict] = {}
    qtys = defaultdict(int)
    crit_flags = defaultdict(bool)

    def _unit_price_text(r: dict) -> str:
        gp = to_gp(r.get("price_text", ""))
        if gp is None:
            return str(r.get("price_text", "")).strip()
        adj = _apply_disposition(gp, disposition)
        return _format_price(adj)

    def _add(r: dict, is_crit: bool):
        name = (r.get("name") or "").strip()
        if not name:
            return

        # Carry Source and detect 3PP on a per-row basis
        src = (r.get("Source") or r.get("source") or "").strip()
        pub = (r.get("Publisher_Source") or r.get("publisher_source") or "").strip().lower()
        is_3pp = pub in ("3rd party", "3rd-party", "third party", "3pp")

        # Use custom dedupe key if present
        key_name = str(r.get("_dedupe_key") or name).strip()
        key = (key_name, bool(is_crit))
        qtys[key] += 1
        crit_flags[key] = crit_flags[key] or bool(is_crit)

        if key not in bucket:
            bucket[key] = dict(r)  # keep originals as a base
            bucket[key].update({
                "name": name,
                "level": int(r.get("level", 0) or 0),
                "rarity": str(r.get("rarity", "")).strip().title(),
                "price": _unit_price_text(r),
                "quantity": 0,
                "category": r.get("category", ""),
                "critical": bool(is_crit),
            })
            # Persist Source and 3PP flag on the display dict
            if src:
                bucket[key]["Source"] = src
            if is_3pp:
                bucket[key]["is_3pp"] = True
                
    for r in rows_base:
        _add(r, False)
    for r in rows_crit:
        _add(r, True)

    items: list[dict] = []
    for key, it in bucket.items():
        it["quantity"] = qtys[key]
        it["critical"] = crit_flags[key]
        items.append(it)
    return items

def _ritual_display_name(base_name: str, level: int) -> str:
    """
    Format ritual names like a scroll: 'Ritual - X Level (Ritual Name)'.
    Avoid double-wrapping if it's already formatted.
    """
    bn = (base_name or "").strip()
    if bn.lower().startswith("ritual -"):
        return bn
    return f"Ritual - {int(level)} Level ({bn})"

def _boost_quantities(items: list[dict], shop_size: str, item_type: str) -> list[dict]:
    rules_all = CONFIG.get("quantity_boost", {}) or {}
    key = (item_type or "").lower()
    r = rules_all.get(key)
    if r is None and key == "armor":
        r = rules_all.get("weapons")
    if r:
        r = r.get((shop_size or "medium").strip().lower(), r.get("medium", {}))
    if not r or not items:
        return items

    import random
    p            = float(r.get("p", 0.55))
    add_min      = int(r.get("add_min", 0))
    add_max      = int(r.get("add_max", 2))
    crit_add_min = int(r.get("crit_add_min", add_min))
    crit_add_max = int(r.get("crit_add_max", add_max))
    max_per_item = int(r.get("max_per_item", 5))

    rng = random.Random()
    out = []
    for it in items:
        q = int(it.get("quantity", 1) or 1)

        # 🚫 Skip boosting uncommon/rare
        rarity = (it.get("rarity") or "Common").strip().title()
        if rarity in ("Uncommon", "Rare"):
            new_it = dict(it); new_it["quantity"] = q
            out.append(new_it)
            continue

        if rng.random() < p:
            if it.get("critical", False):
                q += rng.randint(crit_add_min, crit_add_max)
            else:
                q += rng.randint(add_min, add_max)

        q = max(1, min(q, max_per_item))
        new_it = dict(it); new_it["quantity"] = q
        out.append(new_it)

    return out


# ---------- Name composition helpers (NEW) ----------

_MAT_LABEL_RX = re.compile(r"\(([^)]+)\)\s*$")

def _extract_material_label_from_name(name: str) -> str | None:
    """
    Best-effort: if a previous step templated materials as 'Base (Material [Grade])',
    recover the label from trailing parentheses.
    """
    m = _MAT_LABEL_RX.search((name or "").strip())
    if not m:
        return None
    return m.group(1).strip()

def _compose_weapon_name(it: dict) -> str:
    """
    Final name order:
      Fundamental Rune → Property Rune(s) → Material → Adjustment(s) → Base
    """
    base = (it.get("_base_name") or it.get("base_name") or it.get("name") or "").strip()

    parts: list[str] = []

    # Fundamental rune
    rf = (it.get("_rune_fund_label") or "").strip()
    if rf:
        parts.append(rf)

    # Property runes (use pick order; de-dupe)
    rp = it.get("_rune_prop_labels") or []
    rp = [r.strip() for r in rp if r and isinstance(r, str)]
    if rp:
        seen = set()
        rp_u = []
        for r in rp:
            k = r.lower()
            if k not in seen:
                seen.add(k)
                rp_u.append(r)
        parts.extend(rp_u)

    # Material (e.g., "Cold Iron (Low-Grade)")
    ml = str(it.get("_mat_label") or "").strip()
    if ml:
        parts.append(ml)

    # Adjustments: prefer explicit labels; otherwise parse existing tags (adjustment:<name>)
    adjs = list(it.get("_adj_labels") or [])
    if not adjs:
        tags = str(it.get("tags", ""))
        for tok in tags.split(","):
            tok = tok.strip()
            if tok.lower().startswith("adjustment:"):
                adjs.append(tok.split(":", 1)[1].strip())
    parts.extend([a for a in adjs if a])

    # Build → de-dupe whitespace
    import re as _re
    return _re.sub(r"\s+", " ", " ".join([*parts, base]).strip())

def _compose_armor_name(it: dict) -> str:
    """Rune (fundamental + properties) → Material → Adjustment(s) → Base."""
    base = (it.get("_base_name") or it.get("name") or "").strip()
    parts: list[str] = []
    rf = (it.get("_rune_fund_label") or "").strip()
    if rf: parts.append(rf)
    rp = it.get("_rune_prop_labels") or []
    rp = [r.strip() for r in rp if r and isinstance(r, str)]
    if rp:
        seen=set(); rp_u=[]
        for r in rp:
            k=r.lower()
            if k not in seen:
                seen.add(k); rp_u.append(r)
        parts.extend(rp_u)
    ml = (it.get("_mat_label") or "").strip()
    if ml: parts.append(ml)
    adjs = it.get("_adj_labels") or []
    if not adjs:
        tags = str(it.get("tags",""))
        for tok in tags.split(","):
            tok = tok.strip()
            if tok.lower().startswith("adjustment:"):
                adjs.append(tok.split(":",1)[1].strip())
    parts.extend([a for a in adjs if a])
    import re as _re
    return _re.sub(r"\s+"," "," ".join([*parts, base]).strip())


def _apply_adjustments_to_items(
    items: list[dict],
    adjustments_df: pd.DataFrame,
    item_type: str,
    party_level: int,
    disposition: str,
    rng: random.Random,
    respect_level_window: bool = True,
    skip_specific_magic: bool = True,
) -> list[dict]:
    """
    Legacy internal adjustment handler retained for compatibility with older flows.
    Not used in the main pipeline anymore; external services.utils.apply_adjustments_probabilistic is preferred.
    """
    cfg = CONFIG.get("adjustments", {}) or {}
    # Per-type rates (fallback by item_type if subtype-specific not present)
    base_rate = float(cfg.get("apply_rate", {}).get(item_type.lower(), 0.0))
    rar_w     = cfg.get("rarity_weights", CONFIG.get("rarity_weights", {"Common":90,"Uncommon":9,"Rare":1}))
    name_tpls = cfg.get("name_template", {})
    name_tpl  = name_tpls.get(item_type.lower(), "{base} ({adj})")

    if adjustments_df is None or adjustments_df.empty or not items:
        return items

    A = adjustments_df.copy()
    for c in ("name","subtype","rarity","price_text","level"):
        if c in A.columns:
            A[c] = A[c].astype(str).str.strip() if c != "level" else A[c]
    # Use magic-style window for adjustments themselves
    lo_adj, hi_adj = _level_bounds_for("magic", party_level)
    if "level" in A.columns:
        A = A[(A["level"].astype(int) >= lo_adj) & (A["level"].astype(int) <= hi_adj)]
    if A.empty:
        return items

    # For optional window check on the fused result
    lo_out, hi_out = _level_bounds_for(item_type, party_level)

    def _subtype_of(it: dict) -> str | None:
        c = str(it.get("category","")).lower()
        st = str(it.get("source_table","")).lower()
        nm = str(it.get("name","")).lower()
        if "shield" in (c + st + nm): return "Shield"
        if "weapon" in (c + st):      return "Weapon"
        if "armor"  in (c + st):      return "Armor"
        return None

    def _rarity_max(rb: str, ra: str) -> str:
        order = {"Common":0,"Uncommon":1,"Rare":2,"Unique":3}
        RB, RA = (rb or "Common").title(), (ra or "Common").title()
        return RB if order.get(RB,0) >= order.get(RA,0) else RA

    def _pick_with_weights(df_pool: pd.DataFrame) -> pd.Series:
        wmap = {k.title(): float(v) for k, v in (rar_w or {}).items()}
        common_w = float(wmap.get("Common", 1.0))
        rar = df_pool["rarity"].astype(str).str.strip().str.title()
        ws  = rar.map(lambda x: wmap.get(x, common_w)).fillna(common_w).clip(lower=0.0)
        if not (ws > 0).any():
            ws[:] = 1.0
        return df_pool.sample(n=1, replace=True, weights=ws, random_state=rng.randint(0,10**9)).iloc[0]

    out: list[dict] = []
    for it in items:
        # Optional: don’t stack on specific-magic items
        if skip_specific_magic and str(it.get("category","")).lower().startswith("specific"):
            out.append(it); continue

        subtype = _subtype_of(it)
        if subtype not in ("Armor","Shield","Weapon"):
            out.append(it); continue

        # Allow subtype-specific rate overrides
        rate = float(cfg.get("apply_rate", {}).get(subtype.lower(), base_rate))
        if rng.random() >= rate:
            out.append(it); continue

        pool = A[A["subtype"].str.title().eq(subtype)]
        if pool.empty:
            out.append(it); continue

        pick = _pick_with_weights(pool)

        # ----- PRICE: base (already disposition-adjusted) + adjustment (apply same disposition) -----
        base_gp = to_gp(it.get("price", ""))
        adj_raw = to_gp(pick.get("price_text",""))
        # Note: base 'price' is already disposition-adjusted; we just add the adj gp.
        new_gp  = (base_gp or 0.0) + (adj_raw or 0.0)
        new_price_text = _format_price(new_gp)

        # ----- RARITY: take the rarer -----
        new_rarity = _rarity_max(it.get("rarity","Common"), pick.get("rarity","Common"))

        # ----- LEVEL: max(base, adjustment), optionally enforce window -----
        base_lvl = int(it.get("level", 0) or 0)
        adj_lvl  = int(pick.get("level", 0) or 0)
        fused_lv = max(base_lvl, adj_lvl)
        if respect_level_window and fused_lv > hi_out:
            out.append(it); continue  # skip applying this adj; keep base item

        # Compose the fused row
        fused = dict(it)
        # Important: don't finalize name here; add label only.
        fused.setdefault("_adj_labels", []).append(str(pick.get("name","")).strip())
        fused["rarity"] = new_rarity
        fused["level"]  = fused_lv
        fused["price"]  = new_price_text
        fused["tags"]   = ", ".join(x for x in [it.get("tags",""), f"adjustment:{pick.get('name','')}"] if x).strip()
        out.append(fused)

    return out


# ---------- NEW: Scroll enrichment ----------

_SCROLL_RE = re.compile(r"^Spell scroll \((\d+)(?:st|nd|rd|th) level\)$", re.IGNORECASE)

def _parse_scroll_level(name: str) -> int | None:
    m = _SCROLL_RE.match((name or "").strip())
    return int(m.group(1)) if m else None

def _rarity_multiplier_map() -> dict[str, float]:
    # Configurable; safe defaults if not present
    return {
        **{"Uncommon": 1.25, "Rare": 1.50},  # defaults
        **(CONFIG.get("rarity_price_multipliers", {}) or {})
    }

def _enrich_spell_scrolls(items: list[dict]) -> list[dict]:
    """
    For each picked item, if it's a Spell scroll (Nth level), choose a random spell
    from Spells where Rank = N. Append the spell name to the item's display name and
    bump price for Uncommon/Rare spells using config multipliers.
    """
    if not items:
        return items

    # Load Spells once from SQLite
    spells_df = pd.DataFrame()
    try:
        con = sqlite3.connect(CONFIG["sqlite_db_path"])
        spells_df = pd.read_sql_query('SELECT Name, Rank, Rarity FROM Spells;', con)
    except Exception as e:
        print(">>> WARN: could not load Spells table:", e)
    finally:
        try:
            con.close()
        except Exception:
            pass

    if spells_df.empty:
        return items

    # Normalize spell rarity
    if "Rarity" in spells_df.columns:
        spells_df["Rarity"] = spells_df["Rarity"].astype(str).str.strip().str.title()
    mults = _rarity_multiplier_map()

    rng = random.Random()
    out: list[dict] = []
    for it in items:
        name = str(it.get("name", "")).strip()
        lvl = _parse_scroll_level(name)
        if lvl is None:
            out.append(it); continue

        pool = spells_df[spells_df["Rank"].astype(int) == int(lvl)]
        if pool.empty:
            out.append(it); continue

        pick = pool.sample(n=1, replace=True, random_state=rng.randint(0, 10**9)).iloc[0]
        spell_name = str(pick.get("Name", "")).strip()
        spell_rar  = str(pick.get("Rarity", "Common")).title()

        # Bump price using rarity multiplier (applied to the already disposition-adjusted 'price' text)
        base_gp = to_gp(it.get("price", ""))
        if base_gp is None:
            # try original price_text -> disposition was applied earlier; fallback to raw text if needed
            base_gp = to_gp(it.get("price_text", ""))
        new_gp = base_gp or 0.0
        mult   = float(mults.get(spell_rar, 1.0))
        new_gp = new_gp * mult

        fused = dict(it)
        fused["name"]  = f"{name} - {spell_name}"
        fused["price"] = _format_price(new_gp)
        fused["spell"] = {"name": spell_name, "rarity": spell_rar, "rank": int(lvl)}
        fused["aon_target"] = spell_name   # <-- add this line
        out.append(fused)

    return out

def _is_shield(item: dict) -> bool:
    sub = (item.get("subtype") or item.get("Subtype") or "").strip().lower()
    cat = (item.get("category") or "").strip().lower()
    return ("shield" in sub) or ("shield" in cat)

def _is_shield_property(row: dict) -> bool:
    # Tolerant detection of shield property runes
    t   = (row.get("Type") or row.get("type") or "").strip().lower()
    st  = (row.get("Subtype") or row.get("subtype") or "").strip().lower()
    n   = (row.get("name") or "").strip().lower()
    cat = (row.get("category") or "").strip().lower()

    # Exact-ish label seen in many datasets
    if "shield property" in t and "rune" in t:
        return True
    if "shield property" in st and "rune" in st:
        return True

    # Fallback heuristics
    if ("shield" in cat or "shield" in t or "shield" in st or "shield" in n) and "rune" in (t + n):
        # exclude armor/weapon fundamentals by name
        if not any(k in n for k in ("potency", "striking", "resilient")):
            return True
    return False

def _shield_property_candidates(all_runes: list[dict], party_level: int) -> list[dict]:
    hi = int(party_level) + 1
    out = []
    for r in all_runes:
        if not _is_shield_property(r):
            continue
        rl = int(r.get("level") or 0)
        if rl <= hi:
            out.append(r)
    return out

# ---------- Armor rune helpers ----------

def _is_armor_fundamental(row: dict) -> bool:
    t = (row.get("Type") or row.get("type") or "").strip().lower()
    st = (row.get("Subtype") or row.get("subtype") or "").strip().lower()
    n = (row.get("name") or "").strip().lower()
    # tolerant: any fundamental+armor labeling or classic names (potency/resilient)
    if "fundamental" in t and "armor" in t:
        return True
    if "fundamental" in st and "armor" in st:
        return True
    if ("armor" in t or "armor" in st or "armor" in n) and ("potency" in n or "resilient" in n):
        return True
    return False

def _is_armor_property(row: dict) -> bool:
    t   = (row.get("Type") or row.get("type") or "").strip().lower()
    st  = (row.get("Subtype") or row.get("subtype") or "").strip().lower()
    cat = (row.get("category") or "").strip().lower()
    n   = (row.get("name") or "").strip().lower()
    # explicit armor property
    if ("property" in t and "armor" in t) or ("property" in st and "armor" in st):
        return True
    # category + rune-ish (exclude fundamentals)
    if ("armor" in cat and ("rune" in t or "rune" in n)) and "fundamental" not in t:
        return True
    # heuristic fallback: rune-ish name/type, not weapon, not fundamentals
    if (("rune" in t or "rune" in n)
        and "weapon" not in t
        and not any(k in n for k in ("potency", "striking"))):
        return True
    return False

def _is_shield(item: dict) -> bool:
    sub = (item.get("subtype") or item.get("Subtype") or "").strip().lower()
    cat = (item.get("category") or "").strip().lower()
    # be tolerant of schema naming
    return ("shield" in sub) or ("shield" in cat)

def _potency_cap_for_armor_level(pl: int) -> int:
    pl = int(pl or 0)
    if pl < 5:    return 0
    if pl < 11:   return 1   # 5..10
    if pl < 18:   return 2   # 11..17
    return 3                  # 18+

def _fundamental_candidates_armor(all_runes, armor_level, party_level):
    cap = _potency_cap_for_armor_level(party_level)
    if cap <= 0:
        return []
    lvl_hi = int(party_level)
    out = []
    for r in all_runes:
        if not _is_armor_fundamental(r):
            continue
        pr = parse_potency_rank(r.get("name"))
        if pr < 1 or pr > cap:
            continue
        if int(r.get("level") or 0) <= lvl_hi:
            out.append(r)
    return out

def _property_candidates_armor(all_runes: list[dict], party_level: int) -> list[dict]:
    lo, hi = party_level - 3, party_level + 1
    out = []
    for r in all_runes:
        if not _is_armor_property(r):
            continue
        rl = int(r.get("level") or 0)
        if lo <= rl <= hi:
            out.append(r)
    return out

def _potency_cap_for_weapon_level(pl: int) -> int:
    pl = int(pl or 0)
    if pl < 2:   return 0
    if pl < 10:  return 1   # 2..9
    if pl < 16:  return 2   # 10..15
    return 3                 # 16+

def _is_fundamental(row: dict) -> bool:
    t = (row.get("Type") or "").strip().lower()
    n = (row.get("name") or "").strip().lower()
    # tolerate pluralization and schema variance
    # e.g., "Weapon Fundamental Runes", "Weapon Fundamentals", etc.
    return (("fundamental" in t and "weapon" in t) or
            ("potency" in n and "weapon" in n))

def _is_property(row: dict) -> bool:
    # Be tolerant about schema/casing/fields
    t   = (row.get("Type")    or row.get("type")    or "").strip().lower()
    st  = (row.get("Subtype") or row.get("subtype") or "").strip().lower()
    cat = (row.get("category") or "").strip().lower()
    n   = (row.get("name")     or "").strip().lower()

    # 1) Explicit weapon property labeling in Type/Subtype
    if ("property" in t and "weapon" in t) or ("property" in st and "weapon" in st):
        return True

    # 2) Weapon category + rune-ish, but not fundamentals
    if ("weapon" in cat and ("rune" in t or "rune" in n)) and "fundamental" not in t:
        return True

    # 3) Heuristic fallback: looks like a property rune by name/type; exclude common fundamentals/armor
    if (("rune" in t or "rune" in n)
        and not any(k in n for k in ("potency", "striking", "resilient"))
        and "armor" not in t):
        return True

    return False

def _fundamental_candidates(all_runes, weapon_level, party_level):
    cap = _potency_cap_for_weapon_level(party_level)
    if cap <= 0:
        return []
    lvl_hi = int(party_level)
    out = []
    for r in all_runes:
        if not _is_fundamental(r):
            continue
        pr = parse_potency_rank(r.get("name"))
        if pr < 1 or pr > cap:
            continue
        if int(r.get("level") or 0) <= lvl_hi:
            out.append(r)
    return out

def _weighted_pick_fundamental(cands: list[dict], rng: random.Random, cfg: dict | None) -> dict | None:
    if not cands:
        return None
    fcfg = (cfg or {}).get("fundamental", {}) if cfg else {}
    pot_w = fcfg.get("potency_weights", {"1": 1, "2": 3, "3": 6})  # ← prefer higher potency

    weights = []
    for r in cands:
        pr = parse_potency_rank(r.get("name"))
        w = float(pot_w.get(str(pr), 1.0))
        weights.append(max(w, 0.0001))

    # simple roulette-wheel using rng
    total = sum(weights)
    pick_point = rng.random() * total
    acc = 0.0
    for r, w in zip(cands, weights):
        acc += w
        if pick_point <= acc:
            return r
    return cands[-1]


def _property_candidates(all_runes: list[dict], party_level: int) -> list[dict]:
    lo, hi = party_level - 3, party_level + 1
    out = []
    for r in all_runes:
        if not _is_property(r):
            continue
        rl = int(r.get("level") or 0)
        if lo <= rl <= hi:
            out.append(r)
    out.sort(key=lambda x: int(x.get("level") or 0), reverse=True)
    return out

def _load_runes_df() -> pd.DataFrame:
    df = load_items()
    if df is None or df.empty:
        return pd.DataFrame()
    R = df.copy()
    for c in ("source_table","Type","name","rarity","price_text"):
        if c in R.columns:
            R[c] = R[c].astype(str).str.strip()
    if "level" in R.columns:
        R["level"] = pd.to_numeric(R["level"], errors="coerce").fillna(0).astype(int)
    return R[R.get("source_table","").str.lower().eq("runes")]


def _select_items_core(
    df: pd.DataFrame,
    source_tables,
    item_type: str,
    shop_type: str,
    party_level: int,
    shop_size: str,
    disposition: str,
    include_crit: bool = True,
    count_override: int | None = None,   # NEW
):
    if df is None or df.empty:
        return [], [], {"base_count": 0, "critical_added": 0, "window": (0, 0)}

    d = normalize_str_columns(df, [
        "category", "source_table", "name", "rarity", "price_text",
        "tags", "shop_type", "Bulk", "Source", "subtype", "Publisher_Source"
    ])
    d = _filter_source_tables(d, source_tables)
    d = _apply_shop_type_exact(d, shop_type)

    # Exclude Unique items from all results
    if "rarity" in d.columns:
        d = d[~d["rarity"].str.strip().str.lower().eq("unique")]

    lo, hi = _level_bounds_for(item_type, party_level)
    if "level" in d.columns:
        if item_type.lower() in ("mundane", "weapons", "armor"):
            d = d[(d["level"] <= hi)]
        else:
            d = d[(d["level"] >= lo) & (d["level"] <= hi)]

    if d.empty:
        return [], [], {"base_count": 0, "critical_added": 0, "window": (lo, hi)}

    # --- base count: allow override (for specific magic) ---
    base_n = (
        int(count_override)
        if count_override is not None
        else _counts_for_size_type(shop_type, shop_size, item_type.lower())
    )

    crit_pool = d[(d.get("stock_flag", 0) == 2)]
    norm_pool = d[(d.get("stock_flag", 0) != 2)]

    rng = random.Random()

    # --- rarity-weighted sampling for BASE picks ---
    def _rarity_weight_series(df_in: pd.DataFrame):
        rw = CONFIG.get("rarity_weights", {"Common": 90, "Uncommon": 9, "Rare": 1})

        r = df_in.get("rarity")
        if r is None:
            w = pd.Series([rw.get("Common", 1.0)] * len(df_in), index=df_in.index, dtype=float)
        else:
            r_norm = r.astype(str).str.strip().str.title()
            w = r_norm.map(lambda x: float(rw.get(x, rw.get("Common", 1.0))))
            common_w = float(rw.get("Common", 1.0))
            w = w.fillna(common_w).clip(lower=0.0)

        if not (w > 0).any():
            w = pd.Series([1.0] * len(df_in), index=df_in.index, dtype=float)
        return w

    # Base (norm_pool) sampling
    base_rows = pd.DataFrame()
    if base_n > 0 and not norm_pool.empty:
        if item_type.lower() == "magic" and isinstance(source_tables, (list, tuple, set)):
            # --- Uniform-per-source sampling so each source_table has equal chance ---
            import re as _re
            npool = norm_pool.copy()

            # Normalize source_table tokens on BOTH sides to ensure matches like "held item" vs "held_items"
            def _st_norm(s: str) -> str:
                t = _re.sub(r"[\s\-]+", "_", str(s or "").strip().lower())
                # small alias map for common variants
                if t == "held_items": t = "held_item"
                if t == "held item":  t = "held_item"
                return t

            want = [_st_norm(s) for s in list(source_tables)]
            npool["_st_norm"] = npool["source_table"].astype(str).map(_st_norm)

            # groups only for requested sources (ignore extras)
            groups = {st: npool[npool["_st_norm"] == st] for st in set(want)}
            k = len(want) if len(want) > 0 else 1

            # even quotas across sources, remainder distributed randomly
            order = list(want)
            rng.shuffle(order)
            q, r = divmod(base_n, k)
            quotas = {st: q + (1 if i < r else 0) for i, st in enumerate(order)}

            picks = []
            deficit = 0
            for st, take_n in quotas.items():
                g = groups.get(st)
                if take_n <= 0 or g is None or g.empty:
                    deficit += take_n
                    continue
                w = _rarity_weight_series(g)
                replace_needed = len(g) < take_n
                picks.append(g.sample(n=take_n, replace=replace_needed, weights=w,
                                      random_state=rng.randint(0, 10**9)))

            base_rows = pd.concat(picks, ignore_index=False) if picks else pd.DataFrame()

            # If some source_tables were empty, fill the deficit from the whole pool
            deficit += base_n - len(base_rows)
            if deficit > 0:
                w_all = _rarity_weight_series(npool)
                extra = npool.sample(n=deficit, replace=(len(npool) < deficit), weights=w_all,
                                     random_state=rng.randint(0, 10**9))
                base_rows = pd.concat([base_rows, extra], ignore_index=False)

            # cleanup temp column
            try:
                base_rows = base_rows.drop(columns=["_st_norm"])
            except Exception:
                pass

        else:
            # --- original behavior for non-magic types ---
            weights = _rarity_weight_series(norm_pool)
            base_rows = norm_pool.sample(
                n=base_n,
                replace=True,
                weights=weights,
                random_state=rng.randint(0, 10**9)
            )

    # --- critical pool unchanged ---
    crit_rows = []
    critical_added = 0
    if include_crit and not crit_pool.empty:
        rate = CONFIG.get("critical_bonus_rate", 0.25)
        target = max(1 if not crit_pool.empty else 0, int(round(base_n * rate)))
        if target > 0:
            take = min(target, max(len(crit_pool), target))
            crit_rows = crit_pool.sample(
                n=take,
                replace=True,
                random_state=rng.randint(0, 10**9)
            ).to_dict(orient="records")
            critical_added = len(crit_rows)

    items_pre = _aggregate_items(base_rows, crit_rows, disposition)

    # --- Capture base name once (do NOT compose here) ---
    if item_type.lower() == "weapons":
        for it in items_pre:
            it.setdefault("_base_name", (it.get("name", "") or "").strip())

    # ---------- Enrich spell scrolls ----------
    if item_type.lower() == "magic":
        items_pre = _enrich_spell_scrolls(items_pre)

    # --- adjustments for armor & weapons ---
    if item_type.lower() in ("armor", "weapons"):
        from services.db import load_adjustments
        adj_df = load_adjustments()
        if adj_df is not None and not adj_df.empty:
            adj_cfg   = CONFIG.get("adjustments", {}) or {}
            apply_map = adj_cfg.get("apply_rate", {}) or {}
            rar_w     = adj_cfg.get("rarity_weights", CONFIG.get("rarity_weights", {"Common":90,"Uncommon":9,"Rare":1}))
            name_tpl  = adj_cfg.get("name_template", "{adj} {base}")
            items_pre = apply_adjustments_probabilistic(
                items=items_pre,
                adjustments_df=adj_df,
                apply_rate_map=apply_map,
                rarity_weights=rar_w,
                name_template=name_tpl,
                rng=rng,
            )

    # --- materials for armor & weapons ---
    if item_type.lower() in ("armor", "weapons"):
        from services.db import load_materials
        material_types = ["weapon"] if item_type.lower() == "weapons" else ["armor", "shield"]
        materials_df = load_materials(material_types)
        if materials_df is not None and not materials_df.empty:
            mat_cfg = CONFIG.get("materials", {})
            apply_rate = float(mat_cfg.get("apply_rate", 0.05))
            name_tpl = mat_cfg.get("name_template", "{base} ({material})")
            items_pre = apply_materials_probabilistic(
                items=items_pre,
                materials_df=materials_df,
                apply_rate=apply_rate,
                party_level=party_level,
                name_template=name_tpl,
                rng=rng,
            )

    # --- runes: weapons ---
    if item_type.lower() == "weapons":
        runes_df = _load_runes_df()
        rune_cfg = (CONFIG.get("weapon_runes") or CONFIG.get("runes") or {})
        items_pre = [
            apply_weapon_runes(
                w, player_level=party_level, runes_df=runes_df, rng=rng, rune_cfg=rune_cfg
            )
            for w in items_pre
        ]
        # Authoritative debug AFTER runes applied (scoped to weapons only)
        fund_ct  = sum(1 for w in items_pre if w.get("_rune_fund_label"))
        props_ct = sum(1 for w in items_pre if w.get("_rune_prop_labels"))
        print(f">>> DEBUG[weapons][after-runes] fundamentals={fund_ct}, with_props={props_ct}, items={len(items_pre)}")

        # Compose final weapon names once all systems have annotated
        for it in items_pre:
            it["name"] = _compose_weapon_name(it)

    # --- runes: armor & shields ---
    if item_type.lower() == "armor":
        runes_df = _load_runes_df()
        # Prefer armor_runes / shield_runes configs when provided
        armor_cfg  = (CONFIG.get("armor_runes")  or CONFIG.get("runes") or {})
        shield_cfg = (CONFIG.get("shield_runes") or CONFIG.get("armor_runes") or CONFIG.get("runes") or {})

        new_list = []
        for a in items_pre:
            if _is_shield(a):
                new_list.append(
                    apply_shield_runes(
                        a, player_level=party_level, runes_df=runes_df, rng=rng, rune_cfg=shield_cfg
                    )
                )
            else:
                new_list.append(
                    apply_armor_runes(
                        a, player_level=party_level, runes_df=runes_df, rng=rng, rune_cfg=armor_cfg
                    )
                )
        items_pre = new_list

        # (Optional) compose armor (and shield) names to show rune label prefixes
        for it in items_pre:
            it["name"] = _compose_armor_name(it)

    # --- AoN scroll target cleanup (safe even if none present) ---
    _scroll_with_spell = re.compile(r"^Spell scroll \(\d+(?:st|nd|rd|th) level\)\s*-\s*(.+)$", re.IGNORECASE)
    for it in items_pre:
        nm = str(it.get("name", ""))
        m = _scroll_with_spell.match(nm)
        if m:
            it["aon_target"] = m.group(1).strip()

    # --- boost quantities and return triple ---
    items_post = _boost_quantities(items_pre, shop_size, item_type)

    return items_pre, items_post, {
        "base_count": base_n,
        "critical_added": critical_added,
        "window": (lo, hi),
        "pool_counts": {"norm": int(len(norm_pool)), "crit": int(len(crit_pool))},
    }


def apply_weapon_runes(
    weapon: dict,
    *,
    player_level: int,
    runes_df: pd.DataFrame,
    rng: random.Random,
    rune_cfg: dict | None = None
) -> dict:
    """
    Apply fundamentals/properties with probability knobs and constraints.
    Store labels for name composition; do not mutate the weapon name here.
    """
    fused = dict(weapon)
    fused.setdefault("_base_name", (weapon.get("name") or "").strip())

    # Early out if no runes table
    R = pd.DataFrame(runes_df).copy() if runes_df is not None else pd.DataFrame()
    if R.empty:
        fused["runes"] = []
        return fused

    # Normalize columns we rely on
    for c in ("Type", "name", "rarity", "price_text", "source_table", "level"):
        if c in R.columns:
            if c == "level":
                R[c] = pd.to_numeric(R[c], errors="coerce").fillna(0).astype(int)
            else:
                R[c] = R[c].astype(str).str.strip()

    # Keep only rune rows if source_table exists
    if "source_table" in R.columns:
        R = R[R["source_table"].str.lower().eq("runes")]

    all_runes = R.to_dict(orient="records")
    if not all_runes:
        fused["runes"] = []
        return fused

    # Config
    rcfg = (rune_cfg or {}) if rune_cfg is not None else (CONFIG.get("runes", {}) or {})
    fund_rate = float((rcfg.get("fundamental") or {}).get("apply_rate", 1.0))   # default: try fund always
    prop_rate = float((rcfg.get("property")    or {}).get("apply_rate", 0.6))
    per_slot  = float((rcfg.get("property")    or {}).get("per_slot_rate", 0.7))

    # Base state from current weapon row
    weapon_level = int(fused.get("level") or 0)
    base_rarity  = (fused.get("rarity") or "Common").strip().title()
    base_gp      = to_gp(fused.get("price")) or to_gp(fused.get("price_text")) or 0.0

    new_gp = base_gp
    new_rarity = base_rarity
    chosen: list[dict] = []

    # --- FUNDAMENTAL (probabilistic gate) ---
    chosen_fund = None
    if rng.random() < fund_rate:
        fund_cands  = _fundamental_candidates(all_runes, weapon_level, player_level)
        chosen_fund = _weighted_pick_fundamental(fund_cands, rng, rcfg)
        if chosen_fund:
            chosen.append(chosen_fund)
            new_gp += (to_gp(chosen_fund.get("price_text")) or 0.0)
            new_rarity = bump_rarity(new_rarity, (chosen_fund.get("rarity") or "Common"))

            # Label for final name composer
            fused["_rune_fund_label"] = str(chosen_fund.get("name", "")).strip()

            potency = parse_potency_rank(chosen_fund.get("name"))  # 1..3

            # --- PROPERTIES (probabilistic gates per rules) ---
            prop_labels: list[str] = []
            if potency > 0 and rng.random() < prop_rate:
                prop_cands = _property_candidates(all_runes, player_level)

                picked_names = set()
                slots_taken = 0
                for _ in range(potency):
                    if rng.random() >= per_slot:
                        continue
                    pool = [r for r in prop_cands if r.get("name") not in picked_names]
                    if not pool:
                        break
                    r = pool[rng.randint(0, len(pool) - 1)]
                    picked_names.add(r.get("name"))
                    chosen.append(r)
                    new_gp += (to_gp(r.get("price_text")) or 0.0)
                    new_rarity = bump_rarity(new_rarity, (r.get("rarity") or "Common"))
                    prop_labels.append(str(r.get("name", "")).strip())
                    slots_taken += 1
                    if slots_taken >= potency:
                        break

            # Store property labels ONCE, after the loop finishes
            if prop_labels:
                fused["_rune_prop_labels"] = prop_labels

    # LEVEL = max(base, any chosen rune levels)
    try:
        base_lvl = int(fused.get("level") or 0)
    except Exception:
        base_lvl = 0
    try:
        rune_levels = [int(r.get("level") or 0) for r in (chosen or [])]
    except Exception:
        rune_levels = []
    if rune_levels:
        fused["level"] = max([base_lvl, *rune_levels])
    else:
        fused["level"] = base_lvl

    if chosen:  # at least one rune (fundamental or property)
        fused["category"] = "Runed Weapon"
        fused["is_magic_countable"] = True  # used by the summary counts
    else:
        fused["is_magic_countable"] = False

    # Save back
    fused["runes"]  = chosen
    fused["rarity"] = new_rarity
    fused["price"]  = _format_price(new_gp)
    return fused


def apply_armor_runes(
    armor: dict,
    *,
    player_level: int,
    runes_df: pd.DataFrame,
    rng: random.Random,
    rune_cfg: dict | None = None
) -> dict:
    """
    Armor version of rune application. Mirrors weapons:
      - choose fundamental (potency/resilient) with weights / caps
      - optionally choose property runes
      - bump price/rarity
      - bump level to max(base, runes)
      - set category = 'Runed Armor' and is_magic_countable=True when any rune lands
      - stash labels for final display (if you later want to compose armor names)
    """
    fused = dict(armor)
    fused.setdefault("_base_name", (armor.get("name") or "").strip())

    # NEW: do not apply armor runes to shields
    if _is_shield(fused):
        fused["runes"] = []
        fused["is_magic_countable"] = False
        return fused

    R = pd.DataFrame(runes_df).copy() if runes_df is not None else pd.DataFrame()
    if R.empty:
        fused["runes"] = []
        fused["is_magic_countable"] = False
        return fused

    # normalize
    for c in ("Type", "name", "rarity", "price_text", "source_table", "level"):
        if c in R.columns:
            if c == "level":
                R[c] = pd.to_numeric(R[c], errors="coerce").fillna(0).astype(int)
            else:
                R[c] = R[c].astype(str).str.strip()
    if "source_table" in R.columns:
        R = R[R["source_table"].str.lower().eq("runes")]

    all_runes = R.to_dict(orient="records")
    if not all_runes:
        fused["runes"] = []
        fused["is_magic_countable"] = False
        return fused

    rcfg = (rune_cfg or {}) if rune_cfg is not None else (CONFIG.get("runes", {}) or {})
    fund_rate = float((rcfg.get("fundamental") or {}).get("apply_rate", 1.0))
    prop_rate = float((rcfg.get("property")    or {}).get("apply_rate", 0.6))
    per_slot  = float((rcfg.get("property")    or {}).get("per_slot_rate", 0.7))

    armor_level = int(fused.get("level") or 0)
    base_rarity = (fused.get("rarity") or "Common").strip().title()
    base_gp     = to_gp(fused.get("price")) or to_gp(fused.get("price_text")) or 0.0

    new_gp = base_gp
    new_rarity = base_rarity
    chosen: list[dict] = []

    # FUNDAMENTAL
    chosen_fund = None
    if rng.random() < fund_rate:
        fund_cands  = _fundamental_candidates_armor(all_runes, armor_level, player_level)
        chosen_fund = _weighted_pick_fundamental(fund_cands, rng, rcfg)
        if chosen_fund:
            chosen.append(chosen_fund)
            new_gp     += (to_gp(chosen_fund.get("price_text")) or 0.0)
            new_rarity  = bump_rarity(new_rarity, (chosen_fund.get("rarity") or "Common"))
            fused["_rune_fund_label"] = str(chosen_fund.get("name", "")).strip()

            potency = parse_potency_rank(chosen_fund.get("name"))  # 1..3

            # PROPERTIES
            prop_labels: list[str] = []
            if potency > 0 and rng.random() < prop_rate:
                prop_cands = _property_candidates_armor(all_runes, player_level)
                picked_names = set()
                slots_taken = 0
                for _ in range(potency):
                    if rng.random() >= per_slot:
                        continue
                    pool = [r for r in prop_cands if r.get("name") not in picked_names]
                    if not pool:
                        break
                    r = pool[rng.randint(0, len(pool) - 1)]
                    picked_names.add(r.get("name"))
                    chosen.append(r)
                    new_gp     += (to_gp(r.get("price_text")) or 0.0)
                    new_rarity  = bump_rarity(new_rarity, (r.get("rarity") or "Common"))
                    prop_labels.append(str(r.get("name", "")).strip())
                    slots_taken += 1
                    if slots_taken >= potency:
                        break
            if prop_labels:
                fused["_rune_prop_labels"] = prop_labels

    # LEVEL bump
    try:
        base_lvl = int(fused.get("level") or 0)
    except Exception:
        base_lvl = 0
    try:
        rune_levels = [int(r.get("level") or 0) for r in (chosen or [])]
    except Exception:
        rune_levels = []
    fused["level"] = max([base_lvl, *rune_levels]) if rune_levels else base_lvl

    # CATEGORY / flags
    if chosen:
        fused["category"] = "Runed Armor"
        fused["is_magic_countable"] = True
        prev = (str(fused.get("tags", "")) or "").strip()
        fused["tags"] = ", ".join(t for t in [prev, "Runed"] if t).strip(", ")
    else:
        fused["is_magic_countable"] = False

    # Save back
    fused["runes"]  = chosen
    fused["rarity"] = new_rarity
    fused["price"]  = _format_price(new_gp)
    return fused

def apply_shield_runes(
    armor_row: dict,
    *,
    player_level: int,
    runes_df: pd.DataFrame,
    rng: random.Random,
    rune_cfg: dict | None = None
) -> dict:
    """
    Shields only get a single property rune (no fundamentals).
    - Candidate level <= player_level + 1
    - Update price/rarity/level
    - Category = 'Runed Shield'
    - is_magic_countable = True
    """
    fused = dict(armor_row)
    fused.setdefault("_base_name", (armor_row.get("name") or "").strip())

    # Only handle shields here; non-shields pass through unchanged
    if not _is_shield(fused):
        return fused

    # Load/normalize runes table
    R = pd.DataFrame(runes_df).copy() if runes_df is not None else pd.DataFrame()
    if R.empty:
        fused["runes"] = []
        fused["is_magic_countable"] = False
        return fused

    for c in ("Type", "name", "rarity", "price_text", "source_table", "level"):
        if c in R.columns:
            if c == "level":
                R[c] = pd.to_numeric(R[c], errors="coerce").fillna(0).astype(int)
            else:
                R[c] = R[c].astype(str).str.strip()
    if "source_table" in R.columns:
        R = R[R["source_table"].str.lower().eq("runes")]

    all_runes = R.to_dict(orient="records")
    if not all_runes:
        fused["runes"] = []
        fused["is_magic_countable"] = False
        return fused

    # Config: prefer shield_runes > armor_runes > runes
    rcfg = rune_cfg or CONFIG.get("shield_runes") or CONFIG.get("armor_runes") or CONFIG.get("runes") or {}
    prop_rate = float((rcfg.get("property") or {}).get("apply_rate", 0.50))

    # Base state
    base_rarity = (fused.get("rarity") or "Common").strip().title()
    base_gp     = to_gp(fused.get("price")) or to_gp(fused.get("price_text")) or 0.0
    new_gp      = base_gp
    new_rarity  = base_rarity

    # Single property rune gate
    chosen: list[dict] = []
    if rng.random() < prop_rate:
        pool = _shield_property_candidates(all_runes, player_level)
        if pool:
            r = pool[rng.randint(0, len(pool) - 1)]
            chosen.append(r)
            new_gp     += (to_gp(r.get("price_text")) or 0.0)
            new_rarity  = bump_rarity(new_rarity, (r.get("rarity") or "Common"))
            # label for composer
            fused["_rune_prop_labels"] = [str(r.get("name", "")).strip()]

            # level bump = max(base, rune level)
            try:
                base_lvl = int(fused.get("level") or 0)
            except Exception:
                base_lvl = 0
            rune_lvl = int(r.get("level") or 0)
            fused["level"] = max(base_lvl, rune_lvl)

            # category / flag / tags
            fused["category"] = "Runed Shield"
            fused["is_magic_countable"] = True
            prev = (str(fused.get("tags", "")) or "").strip()
            fused["tags"] = ", ".join(t for t in [prev, "Runed"] if t).strip(", ")
        else:
            fused["is_magic_countable"] = False
    else:
        fused["is_magic_countable"] = False

    # Finalize
    fused["runes"]  = chosen
    fused["rarity"] = new_rarity
    fused["price"]  = _format_price(new_gp)
    return fused


def _apply_disposition(gp: float, disposition: str) -> float:
    mults = CONFIG.get("disposition_multipliers", {"greedy": 0.9, "fair": 1.0, "generous": 1.15})
    m = mults.get((disposition or "fair").lower(), 1.0)
    return gp * m


def select_items_by_source(
    df: pd.DataFrame,
    source_tables,
    item_type: str,
    shop_type: str,
    party_level: int,
    shop_size: str,
    disposition: str,
    include_crit: bool = True,
    count_override: int | None = None,   # NEW
) -> dict:
    items_pre, items_post, meta = _select_items_core(
        df=df,
        source_tables=source_tables,
        item_type=item_type,
        shop_type=shop_type,
        party_level=party_level,
        shop_size=shop_size,
        disposition=disposition,
        include_crit=include_crit,
        count_override=count_override,    # NEW
    )
    
    # --- Ritual display-name tweak: 'Ritual (1st level) - <Name>'
    import re

    def _ordinal(n: int) -> str:
        n = int(n)
        if 10 <= (n % 100) <= 13:
            suf = "th"
        else:
            suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"

    def _ritual_display_name(n: str, lvl: int) -> str:
        n = (n or "").strip()
        # If already in the desired shape, leave it
        if re.match(r"^\s*Ritual\s*\(\d+(st|nd|rd|th)\s+level\)\s*-\s*", n, re.I):
            return n
        # Convert any older 'Ritual - X Level (Name)' to the new shape, otherwise build from base
        m = re.match(r"^\s*Ritual\s*-\s*(\d+)\s*Level\s*\((.+)\)\s*$", n, re.I)
        if m:
            lvl = int(m.group(1))
            base = m.group(2).strip()
            return f"Ritual ({_ordinal(lvl)} level) - {base}"
        return f"Ritual ({_ordinal(lvl)} level) - {n}"

    for it in items_post:
        cat  = str(it.get("category") or "").lower()
        st   = str(it.get("source_table") or "").lower()
        tg_v = it.get("tags")
        tags = " ".join(tg_v) if isinstance(tg_v, list) else str(tg_v or "")
        if ("ritual" in cat) or ("ritual" in st) or ("ritual" in tags.lower()):
            lvl = int(it.get("level") or 0)
            base = str(it.get("name") or "").strip()
            it["name"] = _ritual_display_name(base, lvl)
            # keep link targeting the underlying ritual page if you carry a base target
            it["aon_target"] = it.get("aon_target") or base

    return {
        "items": items_post,
        "base_count": meta["base_count"],
        "critical_added": meta["critical_added"],
        "window": meta["window"],
    }


def select_mundane_items(df, shop_type, party_level, shop_size, disposition):
    return select_items_by_source(
        df=df,
        source_tables=_group("mundane", ["mundane"]),
        item_type="mundane",
        shop_type=shop_type, party_level=party_level,
        shop_size=shop_size, disposition=disposition, include_crit=True
    )

def select_weapons_items(df, shop_type, party_level, shop_size, disposition):
    return select_items_by_source(
        df=df,
        source_tables=_group("weapons", ["weapon_basic", "Weapon_Basic", "weapon", "weapons"]),
        item_type="weapons",
        shop_type=shop_type, party_level=party_level,
        shop_size=shop_size, disposition=disposition, include_crit=True
    )

def select_armor_items(df, shop_type, party_level, shop_size, disposition):
    return select_items_by_source(
        df=df,
        source_tables=_group("armor", ["armor_basic", "armors", "armor", "shield_basic", "shields", "shield"]),
        item_type="armor",
        shop_type=shop_type, party_level=party_level,
        shop_size=shop_size, disposition=disposition, include_crit=True
    )

def select_specific_magic_armor(df, shop_type, party_level, shop_size, disposition):
    return select_items_by_source(
        df=df,
        source_tables=_group("specific_magic_armor", ["specific_magic_armor", "specific_magic_shield"]),
        item_type="magic",
        shop_type=shop_type, party_level=party_level,
        shop_size=shop_size, disposition=disposition,
        include_crit=True,
        count_override=_counts_for_specific_magic(shop_type, shop_size),  # CHANGED
    )

def select_specific_magic_weapons(df, shop_type, party_level, shop_size, disposition):
    return select_items_by_source(
        df=df,
        source_tables=_group("specific_magic_weapons", ["specific_magic_weapons"]),
        item_type="magic",
        shop_type=shop_type, party_level=party_level,
        shop_size=shop_size, disposition=disposition,
        include_crit=True,
        count_override=_counts_for_specific_magic(shop_type, shop_size),  # CHANGED
    )

def select_magic_items(df, shop_type, party_level, shop_size, disposition):
    return select_items_by_source(
        df=df,
        source_tables=_group("magic", [
        "alchemical_items", "cc_structure", "consumables", "grimoire",
        "held_item", "rune", "snares", "spellhearts", "staff_wand", "worn_items"
        ]),
        item_type="magic",
        shop_type=shop_type, party_level=party_level,
        shop_size=shop_size, disposition=disposition, include_crit=True
    )
    
def select_materials(df, shop_type, party_level, shop_size, disposition):
    return select_items_by_source(
        df=df,
        source_tables=_group("materials", ["materials"]),
        item_type="materials", 
        shop_type=shop_type, party_level=party_level,
        shop_size=shop_size, disposition=disposition, include_crit=True
    )
    
def select_formula_items(df, shop_type, party_level, shop_size, disposition):
    return select_formulas(df, shop_type, party_level, shop_size, disposition)

    
# ---------- FORMULAS (new) ----------

def _formula_cost_table_default() -> dict[int, int]:
    """
    Fallback mapping of formula level -> gp cost, based on the provided table.
    """
    return {
        1: 1, 2: 2, 3: 3, 4: 5, 5: 8, 6: 13, 7: 18, 8: 25, 9: 35,
        10: 50, 11: 70, 12: 100, 13: 150, 14: 225, 15: 325, 16: 500,
        17: 750, 18: 1200, 19: 2000, 20: 3500,
    }

def _load_formula_costs_from_sqlite() -> dict[int, int]:
    """
    Load 'Formula' level->gp mapping from SQLite table 'Formula'.
    Tolerates schemas with (Level|ItemLevel) and (Price|PriceText).
    Falls back to the default table on any failure.
    """
    try:
        con = sqlite3.connect(CONFIG["sqlite_db_path"])
        try:
            df = pd.read_sql_query('SELECT * FROM "Formula";', con)
        finally:
            con.close()
        if df is None or df.empty:
            return _formula_cost_table_default()

        # find reasonable columns
        cols = {c.lower(): c for c in df.columns}
        lvl_col = cols.get("level") or cols.get("itemlevel")
        price_col = cols.get("price") or cols.get("pricetext")

        if not lvl_col:
            # try to derive from Name like "Formula - 7"
            if "Name" in df.columns:
                df["__level_guess"] = df["Name"].astype(str).str.extract(r"Formula\s*-\s*(\d+)", expand=False)
                df["__level_guess"] = pd.to_numeric(df["__level_guess"], errors="coerce").astype("Int64")
                lvl_col = "__level_guess"
            else:
                return _formula_cost_table_default()

        def _gp_parse(v):
            s = str(v or "").strip().lower()
            if not s:
                return None
            try:
                return int(float(s))
            except Exception:
                if s.endswith("gp"):
                    try: return int(float(s[:-2].strip()))
                    except Exception: return None
                return None

        df = df.copy()
        df[lvl_col] = pd.to_numeric(df[lvl_col], errors="coerce").astype("Int64")
        if price_col:
            df["__gp"] = df[price_col].map(_gp_parse)
        else:
            # heuristic last resort: any column hinting price/cost/gp
            num_cols = [c for c in df.columns if re.search(r"price|cost|gp", c, re.I)]
            df["__gp"] = df[num_cols[0]].map(_gp_parse) if num_cols else None

        out = {}
        for _, row in df.iterrows():
            lv = row.get(lvl_col)
            gp = row.get("__gp")
            if pd.notna(lv) and int(lv) > 0:
                val = None
                try:
                    if gp is not None and not pd.isna(gp):
                        val = int(gp)
                except Exception:
                    val = None
                out[int(lv)] = val if val is not None else _formula_cost_table_default().get(int(lv))

        base = _formula_cost_table_default()
        base.update(out)   # prefer db values
        return base
    except Exception as e:
        print(">>> WARN: failed to load Formula table:", e)
        return _formula_cost_table_default()

def _counts_for_formulas(shop_type: str, shop_size: str) -> int:
    """
    Prefer counts_by_shop[shop_type][shop_size]['formulas'] if present.
    Else fall back to counts[shop_size]['formulas'].
    If still missing, fall back to ...['materials'].
    Finally, default to [0,0].
    """
    st = (shop_type or "").strip().lower()
    sz = (shop_size or "").strip().lower()

    # Try per-shop override first
    shop_block = (CONFIG.get("counts_by_shop", {}).get(st, {}) or {}).get(sz, {}) or {}
    global_block = (CONFIG.get("counts", {}) or {}).get(sz, {}) or {}

    band = (
        shop_block.get("formulas")
        or global_block.get("formulas")
        or shop_block.get("materials")
        or global_block.get("materials")
        or [0, 0]
    )

    lo, hi = _normalize_pair(band, default=(0, 0))
    n = random.randint(lo, hi) if hi >= lo else 0

    # Debug so you can verify which band it used
    print(f'>>> DEBUG[formulas-counts] shop="{st}" size="{sz}" band={band} -> n={n}')
    return n

def select_formulas(df: pd.DataFrame, shop_type: str, party_level: int, shop_size: str, disposition: str):
    """
    Build a list of formula entries derived from eligible items up to (party_level + 1).
    - Eligible sources: alchemical_items, cc_structure, consumables, grimoire, held_items,
                        runes, snares, spellhearts, staff_wand, worn_items,
                        specific_magic_armor, specific_magic_shield, specific_magic_weapons
    - Exclude Unique items
    - Formula Level = item level; Formula Rarity = item rarity
    - Formula Price = Formula table by Level (gp)
    - Name = 'Formula - <Level> (<Item Name>)'
    - Category = 'Formula'
    - No item_boosted (keep quantity=1)
    """
    if df is None or df.empty:
        return {"items": [], "base_count": 0, "critical_added": 0, "window": (0, 0)}

    eligible_sources = [
        "alchemical_items", "cc_structure", "consumables", "grimoire",
        "held_items", "runes", "snares", "spellhearts", "staff_wand",
        "worn_items", "specific_magic_armor", "specific_magic_shield",
        "specific_magic_weapons",
    ]

    # Normalize columns we might use
    d = normalize_str_columns(
        df,
        ["category","source_table","name","rarity","price_text","tags","shop_type","Bulk","Source","subtype"]
    )

    # --- IMPORTANT: normalize source_table names before filtering ---
    import re
    def _st_norm(s: str) -> str:
        # collapse to snake-ish: "Specific Magic Weapons" -> "specific_magic_weapons"
        return re.sub(r'[^a-z0-9]+', '_', str(s or '').lower()).strip('_')

    eligible_norm = { _st_norm(x) for x in eligible_sources }
    if "source_table" in d.columns:
        d["__st"] = d["source_table"].apply(_st_norm)
        d = d[d["__st"].isin(eligible_norm)]
    else:
        d = d.iloc[0:0]  # no source table info -> nothing eligible

    # Shop filter
    d = _apply_shop_type_exact(d, shop_type)

    # Exclude Unique
    if "rarity" in d.columns:
        d = d[~d["rarity"].str.strip().str.lower().eq("unique")]

    # Level window (<= party_level + 1, and >= 1)
    hi = int(party_level) + 1
    if "level" in d.columns:
        d["level"] = pd.to_numeric(d["level"], errors="coerce").fillna(0).astype(int)
        d = d[(d["level"] >= 1) & (d["level"] <= hi)]

    # Count to pick
    base_n = _counts_for_formulas(shop_type, shop_size)

    # Helpful debug
    print(f">>> DEBUG[formulas] pool={len(d)}, base_n={base_n}, PL+1 max={hi}")

    if d.empty or base_n <= 0:
        return {"items": [], "base_count": 0, "critical_added": 0, "window": (1, hi)}

    # Rarity-weighted sampling
    def _rarity_w_series(df_in: pd.DataFrame):
        rw = CONFIG.get("rarity_weights", {"Common": 80, "Uncommon": 16, "Rare": 4})
        r = df_in.get("rarity")
        if r is None:
            return pd.Series([rw.get("Common", 1.0)] * len(df_in), index=df_in.index, dtype=float)
        rr = r.astype(str).str.strip().str.title()
        w = rr.map(lambda x: float(rw.get(x, rw.get("Common", 1.0)))).fillna(float(rw.get("Common", 1.0)))
        if not (w > 0).any():
            w[:] = 1.0
        return w

    rng = random.Random()
    replace_needed = len(d) < base_n
    picks = d.sample(
        n=base_n,
        replace=replace_needed,
        weights=_rarity_w_series(d),
        random_state=rng.randint(0, 10**9),
    )

    # Price map (DB table if present, else fallback)
    lvl_cost = _load_formula_costs_from_sqlite()

    items = []
    for _, row in picks.iterrows():
        lvl = int(row.get("level") or 0)
        rarity = str(row.get("rarity") or "Common").strip().title()
        base_name = str(row.get("name") or "").strip()
        gp = lvl_cost.get(lvl, _formula_cost_table_default().get(lvl, 0))

        name = f"Formula - {lvl} ({base_name})"
        price_text = f"{gp} gp" if gp else "0 gp"

        items.append({
            "name": name,
            "level": lvl,
            "rarity": rarity,
            # keep raw and display (display goes through disposition as you had it)
            "price_text": price_text,
            "price": _format_price(_apply_disposition(gp, disposition)),
            "quantity": 1,
            "category": "Formula",
            "aon_target": base_name,
            "critical": False,
        })

    return {"items": items, "base_count": base_n, "critical_added": 0, "window": (1, hi)}
