import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.logic import (
    _fundamental_candidates,
    _load_runes_df,
    apply_weapon_runes,
)


def test_striking_rune_present_in_fundamental_candidates():
    runes_df = _load_runes_df()
    all_runes = runes_df.to_dict(orient="records")
    candidates = _fundamental_candidates(all_runes, weapon_level=4, party_level=4)
    names = {str(r.get("name")) for r in candidates}
    assert any(name.lower() == "striking" for name in names)


def test_apply_weapon_runes_can_choose_striking():
    runes_df = _load_runes_df()
    striking_df = runes_df[runes_df["name"].str.fullmatch("Striking", case=False)].copy()
    assert not striking_df.empty, "Expected Striking rune row in dataset"

    weapon = {"name": "Test Sword", "level": 4, "rarity": "Common", "price_text": "0 gp"}
    rng = random.Random(1337)

    fused = apply_weapon_runes(
        weapon,
        player_level=4,
        runes_df=striking_df,
        rng=rng,
        rune_cfg={
            "fundamental": {"apply_rate": 1.0},
            "property": {"apply_rate": 0.0, "per_slot_rate": 0.0},
        },
    )

    assert fused.get("_rune_fund_label", "").lower() == "striking"