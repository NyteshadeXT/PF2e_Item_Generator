# app.py

from flask import Flask, render_template, request, send_file, redirect
import io
import csv
import pandas as pd

from services.db import load_items
from services.logic import (
    select_mundane_items, select_weapons_items, select_armor_items,
    select_specific_magic_armor, select_specific_magic_weapons,
    select_magic_items, select_materials, CONFIG as LOGIC_CONFIG,
    select_formulas,)
from services.utils import rarity_counts, aon_url
from services.spellbooks import select_spellbooks

# Debug/health endpoints live in this blueprint
from services.debug import bp as debug_bp

app = Flask(__name__)
app.register_blueprint(debug_bp, url_prefix="/debug")

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
        aon_url=aon_url,   # 👈 add this
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
