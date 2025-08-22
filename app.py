# app.py

from flask import Flask, render_template, request, send_file, redirect, abort, Response, stream_with_context
import io
import csv
import pandas as pd
import json
import queue, time, uuid

from services.db import load_items
from services.logic import (
    select_mundane_items, select_weapons_items, select_armor_items,
    select_specific_magic_armor, select_specific_magic_weapons,
    select_magic_items, select_materials, CONFIG as LOGIC_CONFIG,
    select_formulas,)
from services.utils import rarity_counts, aon_url
from services.spellbooks import select_spellbooks
from services.debug import bp as debug_bp

app = Flask(__name__)
app.register_blueprint(debug_bp, url_prefix="/debug")

# ---- Real-time broadcaster (SSE) ----
_subscribers: dict[str, list[queue.Queue]] = {}
_latest_roll_id: dict[str, str] = {}  # channel -> last id

def _subscribe(channel: str) -> queue.Queue:
    q = queue.Queue(maxsize=10)
    _subscribers.setdefault(channel, []).append(q)
    return q

def _publish(channel: str, roll_id: str):
    _latest_roll_id[channel] = roll_id
    for q in _subscribers.get(channel, [])[:]:
        try:
            q.put_nowait(roll_id)
        except Exception:
            try: _subscribers[channel].remove(q)
            except ValueError: pass

def _current_roll_id(channel: str) -> str:
    return _latest_roll_id.get(channel, "")

# ---- Real-time broadcaster (SSE) ----
_subscribers: dict[str, list[queue.Queue]] = {}
_latest_roll_id: dict[str, str] = {}  # channel -> last id

def _subscribe(channel: str) -> queue.Queue:
    q = queue.Queue(maxsize=10)
    _subscribers.setdefault(channel, []).append(q)
    return q

def _publish(channel: str, roll_id: str):
    _latest_roll_id[channel] = roll_id
    for q in _subscribers.get(channel, [])[:]:
        try:
            q.put_nowait(roll_id)
        except Exception:
            try: _subscribers[channel].remove(q)
            except ValueError: pass

def _current_roll_id(channel: str) -> str:
    return _latest_roll_id.get(channel, "")

# Optional: preserve the existing /health link in index.html
@app.route("/health")
def health_redirect():
    return redirect("/debug/health", code=302)


def get_shop_types(df: pd.DataFrame):
    if "shop_type" in df.columns and df["shop_type"].dropna().size:
        return sorted(x for x in df["shop_type"].dropna().unique())
    return LOGIC_CONFIG.get("default_shop_types", [])


def _count_crit(items):
    return sum(1 for it in (items or []) if it.get("critical"))


# make AoN helper available in templates
app.jinja_env.globals['aon_url'] = aon_url

def _common_inputs():
    shop_type   = (request.form.get("shop_type") or "General").strip()
    shop_size   = (request.form.get("shop_size") or "medium").strip()
    disposition = (request.form.get("disposition") or "fair").strip()
    try:
        party_level = int(request.form.get("party_level") or 5)
    except Exception:
        party_level = 5
    return shop_type, shop_size, disposition, party_level

def _build_payload(df, shop_type, shop_size, disposition, party_level):
    # mirror your existing results-building logic
    mnd = select_mundane_items(df, shop_type, party_level, shop_size, disposition)
    mat = select_materials(df, shop_type, party_level, shop_size, disposition)
    arm = select_armor_items(df, shop_type, party_level, shop_size, disposition)
    wep = select_weapons_items(df, shop_type, party_level, shop_size, disposition)
    mag = select_magic_items(df, shop_type, party_level, shop_size, disposition)
    # optional: specifics & formulas
    smA = select_specific_magic_armor(df, shop_type, party_level, shop_size, disposition)
    smW = select_specific_magic_weapons(df, shop_type, party_level, shop_size, disposition)
    frm = select_formulas(df, shop_type, party_level, shop_size, disposition)

    # choose a single “window” to display (any of the returned windows is fine—use magic’s)
    window = mag.get("window") or wep.get("window") or arm.get("window") or (party_level, party_level)

    return {
        "mundane_items":   mnd["items"],
        "materials_items": mat["items"],
        "armor_items":     arm["items"],
        "weapons_items":   wep["items"],
        "magic_items":     mag["items"],
        # Optional sections:
        "formulas_items":  frm["items"],
        # Specific magic is often mixed into Magic; include if you want a separate section:
        # "specific_armor_items": smA["items"],
        # "specific_weapon_items": smW["items"],
        "window": window
    }

@app.post("/player-view")
def player_view():
    # Prefer the exact items that were shown to the GM (snapshot). Fallback to recompute.
    raw = request.form.get("snapshot")
    lists = {}
    meta  = {}
    if raw:
        try:
            snap = json.loads(raw)
            lists = (snap or {}).get("lists") or {}
            meta  = (snap or {}).get("shop") or {}
        except Exception:
            lists, meta = {}, {}

    # If snapshot missing or empty, recompute to avoid a blank player view
    if not lists:
        df = load_items()
        shop_type, shop_size, disposition, party_level = _common_inputs()
        payload = _build_payload(df, shop_type, shop_size, disposition, party_level)
        lists = {
            "mundane_items":   payload.get("mundane_items", []),
            "material_items":  payload.get("materials_items", []),
            "armor_items":     payload.get("armor_items", []),
            "weapon_items":    payload.get("weapons_items", []),
            "magic_items":     payload.get("magic_items", []),
            "formula_items":   payload.get("formulas_items", []),
        }
        meta = {
            "shop_type": shop_type,
            "shop_size": shop_size,
            "disposition": disposition,
            "party_level": party_level,
            "window": payload.get("window"),
        }

    # Minimal header: “Player View — <Shop Type>”
    page_title = f'Player View — { (meta.get("shop_type") or "Shop").title() }'

    return render_template(
        "results_player.html",
        page_title=page_title,
        shop_type=meta.get("shop_type"),
        # Only pass what the player view needs:
        mundane_items=lists.get("mundane_items", []),
        material_items=lists.get("material_items", []),
        armor_items=lists.get("armor_items", []),
        weapon_items=lists.get("weapon_items", []),
        magic_items=lists.get("magic_items", []),
        formula_items=lists.get("formula_items", []),
        aon_url=aon_url,
    )

# app.py (only the index() function shown)
@app.route("/", methods=["GET"])
def index():
    df = load_items()
    shop_types = get_shop_types(df)
    dispositions = list(LOGIC_CONFIG.get("disposition_multipliers", {}).keys())
    return render_template(
        "index.html",
        shop_types=shop_types,
        shop_type=None,
        shop_size="medium",
        disposition="fair",
        dispositions=dispositions,
        party_level=5,
    )


@app.route("/query", methods=["POST"])
def query():
    df = load_items()

    # Inputs from the form
    shop_type = (request.form.get("shop_type") or "").strip()
    shop_size = (request.form.get("shop_size") or "medium").strip().lower()
    disposition = (request.form.get("disposition") or "fair").strip().lower()
    try:
        party_level = int(request.form.get("party_level") or 5)
    except Exception:
        party_level = 5

    # Run selections
    mundane_result  = select_mundane_items(df, shop_type, party_level, shop_size, disposition)
    armor_basic     = select_armor_items(df, shop_type, party_level, shop_size, disposition)
    weapons_result  = select_weapons_items(df, shop_type, party_level, shop_size, disposition)
    armor_magic     = select_specific_magic_armor(df, shop_type, party_level, shop_size, disposition)
    weapon_magic    = select_specific_magic_weapons(df, shop_type, party_level, shop_size, disposition)
    magic_basic     = select_magic_items(df, shop_type, party_level, shop_size, disposition)
    material_result = select_materials(df, shop_type, party_level, shop_size, disposition)
    result_formulas = select_formulas(df, shop_type, party_level, shop_size, disposition)
    spellbook_result = select_spellbooks(
        df=df,
        shop_type=shop_type,          # whatever var you already use (e.g. request form)
        party_level=party_level,
        shop_size=shop_size,
        disposition=disposition,
    )

    # Lists actually rendered in the UI
    material_items = (material_result.get("items") or [])
    mundane_items  = (mundane_result.get("items") or [])
    magic_armor    = (armor_magic.get("items") or [])
    magic_weapons  = (weapon_magic.get("items") or [])
    armor_items    = (armor_basic.get("items") or []) + magic_armor     # show specific-magic armor in Armor table
    weapon_items   = (weapons_result.get("items") or []) + magic_weapons  # show specific-magic weapons in Weapon table
    magic_items = (magic_basic.get("items") or [])
    magic_items += (spellbook_result.get("items") or [])
    
    # Helper: unique-by (name, price, rarity, level)
    def _uniq(items):
        seen, out = set(), []
        for it in items:
            key = (
                (it.get("name") or "").strip(),
                (it.get("price") or it.get("price_text") or "").strip(),
                (it.get("rarity") or "").strip(),
                int(it.get("level") or 0),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    # Runed weapons and armor for counts
    runed_weapons = [w for w in weapon_items if (w.get("category") == "Runed Weapon" or w.get("is_magic_countable"))]
    weapons_nonruned = [w for w in weapon_items if w not in runed_weapons]

    runed_armor = [a for a in armor_items if (a.get("category") == "Runed Armor" or a.get("is_magic_countable"))]
    armor_nonruned = [a for a in armor_items if a not in runed_armor]

    # Unique lists used for the "Picked" header math
    mundane_u   = _uniq(mundane_items)
    materials_u = _uniq(material_items)
    armor_u     = _uniq(armor_items)
    weapons_u   = _uniq(weapons_nonruned)
    magic_u     = _uniq(magic_items + magic_armor + magic_weapons + runed_weapons + runed_armor)

    # Summary counts based on what is actually displayed (rarity histogram can stay as-is)
    counts = rarity_counts(mundane_items + material_items + armor_items + weapon_items + magic_items)

    picked = {
        "mundane":   len(mundane_u),
        "materials": len(materials_u),
        "armor":     len(armor_u),
        "weapons":   len(weapons_u),   # runed weapons excluded here
        "magic":     len(magic_u),     # runed weapons included here
        "formulas": len(result_formulas.get("items", [])), 

        # Critical counts: follow the same partition as above
        "critical": (
            _count_crit(mundane_items) +
            _count_crit(material_items) +
            _count_crit(armor_items) +
            _count_crit(weapons_nonruned) +
            _count_crit(magic_armor) +
            _count_crit(magic_weapons) +
            _count_crit(magic_items) +
            _count_crit(runed_weapons)
        ),
        "critical_mundane":      _count_crit(mundane_items),
        "critical_materials":    _count_crit(material_items),
        "critical_armor_shield": _count_crit(armor_items),
        "critical_weapons":      _count_crit(weapons_nonruned),
        "critical_magic":        (
            _count_crit(magic_armor) +
            _count_crit(magic_weapons) +
            _count_crit(magic_items) +
            _count_crit(runed_weapons)
        ),
    }

    snapshot = {
        "shop": {
            "shop_type": shop_type,
            "shop_size": shop_size,
            "disposition": disposition,
            "party_level": party_level,
            "window": magic_basic.get("window") if isinstance(magic_basic, dict) else None,
        },
        "lists": {
            "mundane_items": mundane_items,
            "material_items": material_items,
            "armor_items": armor_items,
            "weapon_items": weapon_items,
            "magic_items": magic_items,
            "formula_items": result_formulas.get("items", []),
        }
    }

    channel = (request.form.get("channel") or "default").strip().lower()
    roll_id = uuid.uuid4().hex[:12]
    _publish(channel, roll_id)

    return render_template(
        "results.html",
        shop_type=shop_type,
        shop_size=shop_size,
        disposition=disposition,
        party_level=party_level,
        picked=picked,
        counts=counts,
        mundane_items=mundane_items,
        material_items=material_items,
        armor_items=armor_items,
        weapon_items=weapon_items,
        magic_items=magic_items,
        formula_items = result_formulas.get("items", []),
        aon_url=aon_url,
        snapshot=snapshot,
        roll_id=roll_id,
        channel=channel,
    )


@app.route("/export", methods=["POST"])
def export_csv():
    """
    Accepts JSON payload with {"items": [...dicts...]} and returns a CSV.
    You can POST the items currently shown on the Results page to this endpoint.
    """
    data = request.get_json(silent=True) or {}
    items = data.get("items") or []

    # CSV columns (order)
    columns = ["Name", "Level", "Rarity", "Price", "Quantity", "Category"]
    # Normalize keys from your item dicts
    def row(i):
        return [
            i.get("name", ""),
            i.get("level", ""),
            i.get("rarity", ""),
            i.get("price", ""),
            i.get("quantity", ""),
            i.get("category", ""),
        ]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(columns)
    for it in items:
        writer.writerow(row(it))

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name="inventory.csv",
        max_age=0,
    )


if __name__ == "__main__":
    # For local dev convenience; in production use a WSGI server
    app.run(host="0.0.0.0", port=5000, debug=True)
