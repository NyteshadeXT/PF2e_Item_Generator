# services/debug.py
from flask import Blueprint, request, Response, current_app
import json
from services.db import load_items
from services.logic import (
    select_mundane_items, select_weapons_items, _level_window, CONFIG as LOGIC_CONFIG,
    _filter_source_tables, _apply_shop_type_exact, _select_items_core
)

bp = Blueprint("debug", __name__)

# Optional: read once for convenience
GROUPS = LOGIC_CONFIG.get("source_table_groups", {})

def filter_items(df, source_tables, shop_type, level_range, level_mode="between"):
    """
    Returns (filtered_df, crit_pool, norm_pool)
    """
    if df is None or df.empty:
        empty = df.iloc[0:0] if hasattr(df, "iloc") else df
        return empty, empty, empty

    d = df.copy()
    for c in ["category","source_table","name","rarity","price_text","tags","shop_type","Bulk","Source","level"]:
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip() if c != "level" else d[c]

    # 1) source_table(s)
    d = _filter_source_tables(d, source_tables)
    # 2) shop_type exact if present
    d = _apply_shop_type_exact(d, shop_type)
    # 3) level window
    lo, hi = level_range
    if "level" in d.columns:
        if level_mode == "upper":
            d = d[(d["level"] <= hi)]
        else:
            d = d[(d["level"] >= lo) & (d["level"] <= hi)]
    # 4) split critical vs normal pools
    crit_pool = d[(d.get("stock_flag", 0) == 2)] if "stock_flag" in d.columns else d.iloc[0:0]
    norm_pool = d[(d.get("stock_flag", 0) != 2)] if "stock_flag" in d.columns else d
    return d, crit_pool, norm_pool


@bp.route("/health")
def health():
    df = load_items()

    def uniques(col):
        if col in df.columns:
            return sorted(df[col].dropna().astype(str).str.strip().unique().tolist())[:50]
        return []

    stats = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "shop_type_uniques": uniques("shop_type"),
        "category_uniques": uniques("category"),
        "source_table_uniques": uniques("source_table"),
        "level_min": int(df["level"].min()) if "level" in df.columns and len(df) else None,
        "level_max": int(df["level"].max()) if "level" in df.columns and len(df) else None,
        "sample": df.head(5).to_dict(orient="records"),
    }
    return Response(json.dumps(stats, indent=2, default=str), mimetype="application/json")


@bp.route("/mundane")
def debug_mundane():
    df = load_items()
    shop_type   = (request.args.get("shop_type") or "General").strip()
    shop_size   = (request.args.get("shop_size") or "medium").strip()
    disposition = (request.args.get("disposition") or "fair").strip()
    try:
        party_level = int(request.args.get("party_level") or 5)
    except ValueError:
        current_app.logger.warning("Invalid party_level '%s' → default 5", request.args.get("party_level"))
        party_level = 5

    lo, hi = _level_window(party_level)

    d3, crit_pool, norm_pool = filter_items(
        df=df,
        source_tables=GROUPS.get("mundane", ["mundane"]),
        shop_type=shop_type,
        level_range=(lo, hi),
        level_mode="between"
    )

    result = select_mundane_items(df, shop_type, party_level, shop_size, disposition)

    payload = {
        "inputs": {"shop_type": shop_type, "shop_size": shop_size, "disposition": disposition,
                   "party_level": party_level, "level_window": [lo, hi]},
        "counts_per_stage": {
            "after_filters": int(len(d3)),
            "critical_pool": int(len(crit_pool)),
            "normal_pool": int(len(norm_pool))
        },
        "picked_summary": {
            "base_count_requested": result.get("base_count"),
            "critical_added": result.get("critical_added"),
            "items_returned": len(result.get("items", []))
        },
        "first_five_after_filters": d3.head(5).to_dict(orient="records"),
        "first_five_criticals": crit_pool.head(5).to_dict(orient="records"),
        "first_items_returned": result.get("items", [])[:5]
    }
    return Response(json.dumps(payload, indent=2, default=str), mimetype="application/json")


@bp.route("/weapons")
def debug_weapons():
    df = load_items()
    shop_type   = (request.args.get("shop_type") or "Blacksmith").strip()
    shop_size   = (request.args.get("shop_size") or "medium").strip()
    disposition = (request.args.get("disposition") or "fair").strip()
    try:
        party_level = int(request.args.get("party_level") or 5)
    except ValueError:
        current_app.logger.warning("Invalid party_level '%s' → default 5", request.args.get("party_level"))
        party_level = 5

    lo, hi = _level_window(party_level)

    d3, crit_pool, norm_pool = filter_items(
        df=df,
        source_tables=GROUPS.get("weapons", ["weapon_basic","weapons_basic","weapon","weapons"]),
        shop_type=shop_type,
        level_range=(lo, hi),
        level_mode="between"
    )

    result = select_weapons_items(df, shop_type, party_level, shop_size, disposition)

    payload = {
        "inputs": {"shop_type": shop_type, "shop_size": shop_size, "disposition": disposition,
                   "party_level": party_level, "level_window": [lo, hi]},
        "counts_per_stage": {
            "after_filters": int(len(d3)),
            "critical_pool": int(len(crit_pool)),
            "normal_pool": int(len(norm_pool))
        },
        "picked_summary": {
            "base_count_requested": result.get("base_count"),
            "critical_added": result.get("critical_added"),
            "items_returned": len(result.get("items", []))
        },
        "first_five_after_filters": d3.head(5).to_dict(orient="records"),
        "first_five_criticals": crit_pool.head(5).to_dict(orient="records"),
        "first_items_returned": result.get("items", [])[:5]
    }
    return Response(json.dumps(payload, indent=2, default=str), mimetype="application/json")


@bp.route("/quantities")
def debug_quantities():
    df = load_items()
    item_type   = (request.args.get("type") or "weapons").strip().lower()
    shop_type   = (request.args.get("shop_type") or "Blacksmith").strip()
    shop_size   = (request.args.get("shop_size") or "medium").strip()
    disposition = (request.args.get("disposition") or "fair").strip()
    try:
        party_level = int(request.args.get("party_level") or 5)
    except ValueError:
        current_app.logger.warning("Invalid party_level '%s' → default 5", request.args.get("party_level"))
        party_level = 5

    # pull from config groups; keep safe default
    st = GROUPS.get(item_type, [])
    if not st:
        defaults = {
            "mundane": ["mundane"],
            "weapons": ["weapon_basic","weapons_basic","weapon","weapons"],
            "armor":   ["armor_basic","armors","armor","shield_basic","shields","shield"],
            "magic":   ["staff_wand","spellhearts","worn_items","scrolls","grimoire","tattoos"],
        }
        st = defaults.get(item_type, [])

    pre, post, meta = _select_items_core(
        df=df,
        source_tables=st,
        item_type=item_type,
        shop_type=shop_type,
        party_level=party_level,
        shop_size=shop_size,
        disposition=disposition,
        include_crit=True,
    )

    def _q(name, coll):
        for r in coll:
            if r.get("name") == name:
                return r.get("quantity")
        return None

    changed = []
    for r in post:
        name = r.get("name")
        if not name:
            continue
        pre_q = _q(name, pre)
        if pre_q != r.get("quantity"):
            changed.append({
                "name": name,
                "pre_qty": pre_q,
                "post_qty": r.get("quantity"),
                "critical": bool(r.get("critical"))
            })

    payload = {
        "inputs": {"type": item_type, "shop_type": shop_type, "shop_size": shop_size, "party_level": party_level},
        "config_rule_used": LOGIC_CONFIG.get("quantity_boost", {}).get(item_type, {}).get(shop_size, {}),
        "pool_counts": meta.get("pool_counts", {}),
        "pre_counts": len(pre),
        "post_counts": len(post),
        "changed_items": changed[:25],
        "first_pre": pre[:5],
        "first_post": post[:5],
    }
    return Response(json.dumps(payload, indent=2, default=str), mimetype="application/json")
