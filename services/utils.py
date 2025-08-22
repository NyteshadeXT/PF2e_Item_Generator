# services/utils.py
import pandas as pd
import re, random

from decimal import Decimal, ROUND_DOWN

def to_gp(price_text: str | float | int | None) -> float | None:
    if price_text is None:
        return None
    s = str(price_text).strip().lower()
    if not s:
        return None
    try:
        return float(s)  # already numeric gp
    except ValueError:
        pass
    if s.endswith("gp"):
        try: return float(s[:-2].strip())
        except ValueError: return None
    if s.endswith("sp"):
        try: return float(s[:-2].strip()) / 10.0
        except ValueError: return None
    if s.endswith("cp"):
        try: return float(s[:-2].strip()) / 100.0
        except ValueError: return None
    return None


def format_price(gp_value: float | None) -> str:
    """
    Format a gp float into PF2e denominations:
      1 gp = 10 sp = 100 cp
    """
    if gp_value is None:
        return ""
    cp_total = int(round(float(gp_value) * 100))  # convert gp → cp
    gp, rem = divmod(cp_total, 100)
    sp, cp = divmod(rem, 10)
    parts = []
    if gp: parts.append(f"{gp} gp")
    if sp: parts.append(f"{sp} sp")
    if cp: parts.append(f"{cp} cp")
    return " ".join(parts) if parts else "0 gp"


def normalize_str_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip()
    return d

def aon_spell_url(spell_name: str) -> str:
    """
    Build an Archives of Nethys search URL for a spell.
    """
    if not spell_name:
        return "#"
    import urllib.parse
    q = urllib.parse.quote(spell_name.strip())
    return f"https://2e.aonprd.com/Search.aspx?query={q}&type=spell"


def canonical_rarity(s: str | None) -> str:
    s = (s or "Common").strip().title()
    return s if s in ("Common", "Uncommon", "Rare", "Unique") else "Common"


def rarity_counts(items: list[dict]) -> dict:
    counts = {"common": 0, "uncommon": 0, "rare": 0, "unique": 0}
    for it in items or []:
        r = canonical_rarity(it.get("rarity")).lower()
        counts[r] = counts.get(r, 0) + 1
    return counts


def normalize_token(s: str | None) -> str:
    s = str(s or "").lower().strip()
    return "".join(ch for ch in s if ch.isalnum())


def parse_int(val, default: int, lo: int | None = None, hi: int | None = None, warn=None) -> int:
    try:
        x = int(val)
    except Exception:
        if warn: warn(f"Invalid int '{val}', defaulting to {default}")
        x = default
    if lo is not None: x = max(lo, x)
    if hi is not None: x = min(hi, x)
    return x


def group_from_config(groups: dict, key: str, default: list[str]) -> list[str]:
    vals = groups.get(key, [])
    return vals if isinstance(vals, list) and vals else default


from urllib.parse import quote_plus

def aon_url(name: str | None) -> str:
    """
    Build a link to Archives of Nethys search for the given item name.
    Falls back to empty string if no name is provided.
    """
    if not name:
        return ""
    q = quote_plus(str(name).strip())
    return f"https://2e.aonprd.com/Search.aspx?query={q}&display=all"


# --- Adjustment helpers (utils) ---
def adj_detect_subtype(it: dict) -> str | None:
    """
    Classify a picked item as 'Armor', 'Shield', or 'Weapon' based on category/source_table/name.
    """
    c  = str(it.get("category", "")).lower()
    st = str(it.get("source_table", "")).lower()
    nm = str(it.get("name", "")).lower()
    if "shield" in (c + st + nm): return "Shield"
    if "weapon" in (c + st):      return "Weapon"
    if "armor"  in (c + st):      return "Armor"
    return None

def adj_rarer_rarity(base: str | None, adj: str | None) -> str:
    order = {"Common": 0, "Uncommon": 1, "Rare": 2, "Unique": 3}
    b = (base or "Common").title()
    a = (adj  or "Common").title()
    return b if order.get(b, 0) >= order.get(a, 0) else a

def adj_parse_price_to_gp(text: str | None) -> float | None:
    """
    Parse strings like '250 gp', '5 sp', '3 cp' (or bare numbers) to gold pieces as float.
    """
    if not text: return None
    s = str(text).strip().lower()
    if s in ("0", "0 gp", "0.0 gp"): return 0.0
    total_cp = Decimal(0)
    for val, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(gp|sp|cp)", s):
        d = Decimal(val)
        total_cp += d * 100 if unit == "gp" else d * 10 if unit == "sp" else d
    if total_cp == 0:
        try:
            return float(s)
        except Exception:
            return None
    return float(total_cp / Decimal(100))

def _fmt_dec_no_sci(d: Decimal, places: int = 2) -> str:
    q = Decimal(10) ** -places
    d = d.quantize(q, rounding=ROUND_DOWN)
    s = format(d, "f")          # ← never uses scientific notation
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"

def adj_format_price_text(gp: float | None) -> str:
    if gp is None:
        return ""
    gp_dec = Decimal(str(gp))
    if gp_dec >= 1:
        return f"{_fmt_dec_no_sci(gp_dec, 2)} gp"
    sp_dec = gp_dec * 10
    if sp_dec >= 1:
        return f"{_fmt_dec_no_sci(sp_dec, 0)} sp"
    cp_dec = gp_dec * 100
    return f"{_fmt_dec_no_sci(cp_dec, 0)} cp"


def adj_rarity_weights_series(df_in: pd.DataFrame, weights_cfg: dict | None) -> pd.Series:
    if not isinstance(weights_cfg, dict) or df_in.empty:
        return pd.Series([1.0] * len(df_in), index=df_in.index, dtype=float)
    wmap = {str(k).title(): float(v) for k, v in weights_cfg.items()}
    common_w = wmap.get("Common", 1.0)
    rar = df_in.get("rarity")
    if rar is None:
        return pd.Series([common_w] * len(df_in), index=df_in.index, dtype=float)
    s = rar.astype(str).str.strip().str.title().map(lambda x: wmap.get(x, common_w)).fillna(common_w)
    if not (s > 0).any(): s[:] = 1.0
    return s

def apply_adjustments_probabilistic(
    items: list[dict],
    adjustments_df: pd.DataFrame,
    apply_rate_map: dict,
    rarity_weights: dict | None = None,
    name_template: str = "{adj} {base}",
    rng: random.Random | None = None,
    level_window: tuple[int | None, int | None] | None = None,
) -> list[dict]:
    """
    For each picked item (armor/shield/weapon), roll a per-subtype chance.
    On success, pick a matching adjustment, then:
      - name  = name_template.format(adj=<adj name>, base=<base name>)  [PREFIX]
      - price = base_price + adjustment_price
      - rarity = rarer(base.rarity, adj.rarity)
      - tags += 'Adjusted, adjustment:<adj name>'
    """
    if not items or adjustments_df is None or adjustments_df.empty:
        return items
    if rng is None:
        rng = random.Random()

    A = adjustments_df.copy()
    # normalize minimal columns
    for c in ("name", "subtype", "rarity", "price_text"):
        if c in A.columns:
            A[c] = A[c].astype(str).str.strip()
    if "level" in A.columns:
        A["level"] = pd.to_numeric(A["level"], errors="coerce").fillna(0).astype(int)
        if level_window and all(v is not None for v in level_window):
            lo, hi = level_window
            A = A[(A["level"] >= int(lo)) & (A["level"] <= int(hi))]

    out: list[dict] = []
    for it in items:
        subtype = adj_detect_subtype(it)
        if subtype not in ("Armor", "Shield", "Weapon"):
            out.append(it)
            continue

        # Per-subtype probability (0 if missing)
        p = float((apply_rate_map or {}).get(subtype.lower(), 0.0))
        if rng.random() >= p:
            out.append(it)
            continue

        # Pull compatible adjustments
        pool = A[A["subtype"].str.title().eq(subtype)]
        if pool.empty:
            out.append(it)
            continue

        # Pick one (rarity-weighted if provided)
        ws = adj_rarity_weights_series(pool, rarity_weights or {})
        pick = pool.sample(n=1, replace=True, weights=ws, random_state=rng.randint(0, 10**9)).iloc[0]

        # PRICE = base + adjustment (both parsed to gp)
        base_gp = adj_parse_price_to_gp(it.get("price") or it.get("price_text"))
        adj_gp  = adj_parse_price_to_gp(pick.get("price_text"))
        new_gp  = (base_gp or 0.0) + (adj_gp or 0.0)
        new_price_text = adj_format_price_text(new_gp)

        # RARITY = rarer of the two
        new_rarity = adj_rarer_rarity(it.get("rarity"), pick.get("rarity"))

        # LEVEL = max(base, adjustment)
        try:
            base_lvl = int(it.get("level") or 0)
        except Exception:
            base_lvl = 0
        try:
            adj_lvl = int(pick.get("level") or pick.get("Level") or 0)
        except Exception:
            adj_lvl = 0
        new_level = max(base_lvl, adj_lvl)

        # NAME = prefix with adjustment
        new_name = name_template.format(adj=pick.get("name", ""), base=it.get("name", ""))

        # ---- build fused row ----
        fused = dict(it)  # <-- create fused BEFORE writing to it
        fused["name"]        = new_name
        fused.setdefault("_base_name", (it.get("name", "") or "").strip())
        # (optional) if some code expects 'base_name':
        # fused.setdefault("base_name", fused["_base_name"])
        fused["price"]       = new_price_text
        fused["rarity"]      = new_rarity
        fused["level"] = new_level
        fused["is_adjusted"] = True

        # Label for the final composer (ADJUSTMENT only; materials handled elsewhere)
        fused.setdefault("_adj_labels", []).append(str(pick.get("name", "")).strip())

        # tags: ensure a STRING, append both a human tag and a machine-parsable tag
        prev_tags = (str(it.get("tags", "")) or "").strip()
        adj_tag   = f"adjustment:{pick.get('name','')}"
        fused["tags"] = ", ".join([t for t in [prev_tags, "Adjusted", adj_tag] if t]).strip(", ")

        out.append(fused)

    return out
    
_SCROLL_RE = re.compile(r"^Spell scroll \((\d+)(?:st|nd|rd|th) level\)$", re.IGNORECASE)

def parse_scroll_level(item_name: str) -> int | None:
    """
    'Spell scroll (5th level)' -> 5
    Returns None if not a spell scroll name.
    """
    m = _SCROLL_RE.match(item_name.strip())
    return int(m.group(1)) if m else None

def pick_one(seq, rnd: random.Random | None = None):
    if not seq:
        return None
    r = rnd or random
    return r.choice(seq)

def apply_rarity_markup(base_price: int | float, rarity: str, multipliers: dict[str, float]) -> int:
    mult = multipliers.get(rarity.title(), 1.0)
    # round to nearest gp (or keep your currency rounding rules)
    return int(round(base_price * mult))

# --- Materials helpers (utils) ---
def apply_materials_probabilistic(
    items: list[dict],
    materials_df: pd.DataFrame,
    apply_rate: float,
    party_level: int,
    name_template: str = "{base} ({material})",
    rng: random.Random | None = None,
) -> list[dict]:
    """
    For each picked item (armor/shield/weapon), roll a chance to apply a special material.
    On success, pick a material compatible with the item's subtype and level, then:
      - name  = name_template.format(base=<base name>, material=<material name>)
      - price = base_price + material_price (+ per-bulk charge)
      - rarity = rarer(base.rarity, material.rarity)
      - level  = max(base.level, material.level)
      - labels: _mat_label for final name composer
    """
    if not items or materials_df is None or materials_df.empty or apply_rate <= 0:
        return items
    if rng is None:
        rng = random.Random()

    M = materials_df.copy()

    # Normalize columns we rely on (be tolerant of casing)
    for c in ("name", "prerequisite", "rarity"):
        if c in M.columns:
            M[c] = M[c].astype(str).str.strip()
    # level/price cols to numeric
    if "level" in M.columns:
        M["level"] = pd.to_numeric(M["level"], errors="coerce").fillna(0).astype(int)
    for c in ("price_add", "price_add_per_bulk"):
        if c in M.columns:
            M[c] = pd.to_numeric(M[c], errors="coerce").fillna(0.0)

    def _bulk_to_float(val) -> float:
        s = str(val or "").strip().lower()
        if not s or s in {"—", "-", "none"}:
            return 0.0
        if s == "l":  # treat Light bulk as a tenth (adjust if your pricing expects 0)
            return 0.1
        try:
            return float(s)
        except Exception:
            return 0.0

    out: list[dict] = []
    for it in items:
        # Roll chance to apply a material
        if rng.random() >= apply_rate:
            out.append(it)
            continue

        # Item subtype and bulk (needed for per-bulk pricing)
        item_subtype = str(it.get("subtype", "")).strip().lower()
        item_bulk = _bulk_to_float(it.get("Bulk"))

        if not item_subtype or "prerequisite" not in M.columns or "level" not in M.columns:
            out.append(it)
            continue

        # Compatible materials at/under party level
        pool = M[(M["prerequisite"].str.lower() == item_subtype) & (M["level"] <= int(party_level))]
        if pool.empty:
            out.append(it)
            continue

        # Pick one material
        pick = pool.sample(n=1, replace=True, random_state=rng.randint(0, 10**9)).iloc[0]

        # PRICE = base + flat add + per-bulk add
        base_gp = adj_parse_price_to_gp(it.get("price") or it.get("price_text"))
        add_gp = float(pick.get("price_add", 0.0))
        per_bulk_gp = float(pick.get("price_add_per_bulk", 0.0))
        new_gp = (base_gp or 0.0) + add_gp + (per_bulk_gp * item_bulk)
        if new_gp < 0:
            new_gp = 0.0
        new_price_text = adj_format_price_text(new_gp)

        # RARITY = rarer of the two
        new_rarity = adj_rarer_rarity(it.get("rarity"), pick.get("rarity"))

        # LEVEL = max(base, material)
        try:
            base_lvl = int(it.get("level") or 0)
        except Exception:
            base_lvl = 0
        try:
            mat_lvl = int(pick.get("level") or pick.get("Level") or 0)
        except Exception:
            mat_lvl = 0
        new_level = max(base_lvl, mat_lvl)

        # NAME (composer will overwrite for weapons, but this is fine)
        mat_name = str(pick.get("name", "")).strip()
        new_name = name_template.format(base=str(it.get("name", "")), material=mat_name)

        # ---- build fused row ----
        fused = dict(it)  # make sure fused exists before writing
        fused["name"]        = new_name
        fused.setdefault("_base_name", (it.get("name", "") or "").strip())
        fused["_mat_label"]  = mat_name
        fused["price"]       = new_price_text
        fused["rarity"]      = new_rarity
        fused["level"]       = new_level
        fused["is_material"] = True

        # tags
        prev_tags = (str(it.get("tags", "")) or "").strip()
        mat_tag   = f"material:{mat_name}"
        fused["tags"] = ", ".join(t for t in [prev_tags, "Material", mat_tag] if t).strip(", ")

        out.append(fused)

    return out

RARITY_ORDER = {"common": 0, "uncommon": 1, "rare": 2, "unique": 3}

def bump_rarity(base_rarity: str, add_rarity: str) -> str:
    """Return the higher rarity between base_rarity and add_rarity."""
    br = (base_rarity or "common").lower()
    ar = (add_rarity or "common").lower()
    # Handle unknowns gracefully
    bi = RARITY_ORDER.get(br, 0)
    ai = RARITY_ORDER.get(ar, 0)
    for k, v in RARITY_ORDER.items():
        if v == max(bi, ai):
            return k
    return br

def within_range(value: int, low: int, high: int) -> bool:
    return low <= value <= high

def add_price(current_price: int, addend: int) -> int:
    return (current_price or 0) + (addend or 0)

def parse_potency_rank(name: str) -> int:
    # e.g., "Weapon Potency +1" -> 1
    if not name:
        return 0
    name = name.strip().lower()
    for n in (3, 2, 1):
        if f"+{n}" in name:
            return n
    return 0
