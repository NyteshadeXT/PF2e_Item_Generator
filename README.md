# PF2e Shop Generator (Web)

A modular Flask web app that modernizes your Access-based shop generator with a SQLite backend (view: `v_items_norm`), with a CSV fallback.

## Run locally

```bash
pip install flask pandas
python app.py
# Open http://localhost:7860
```

## Configure
Edit `config.json` to switch between `sqlite` and `csv`, tweak counts, disposition multipliers, critical rates, and level spread.

## Port plan
We will port each VBA routine (`genMundane`, `gen_Weapon`, etc.) into focused Python pickers that filter the `v_items_norm` records precisely, replicate rerolls when no item is found, and honor shop types like Tattooist having tattoos only.
