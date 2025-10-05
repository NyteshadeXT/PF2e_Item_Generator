import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.logic import (
    _fundamental_candidates,
    _load_runes_df,
    apply_armor_runes,
    apply_weapon_runes,
)

class SequenceRandom(random.Random):
    """Deterministic RNG that yields a preset sequence for random()."""

    def __init__(self, sequence, *, seed=0):
        super().__init__(seed)
        self._sequence = list(sequence)
        self._index = 0

    def random(self):
        if self._index < len(self._sequence):
            value = self._sequence[self._index]
            self._index += 1
            return float(value)
        return super().random()
        
def test_striking_rune_present_in_fundamental_candidates():
    runes_df = _load_runes_df()
    all_runes = runes_df.to_dict(orient="records")
    candidates = _fundamental_candidates(all_runes, weapon_level=4, party_level=4)
    names = {str(r.get("name")) for r in candidates}
    assert any(name.lower() == "striking" for name in names)


def test_apply_weapon_runes_can_choose_striking_with_potency():
    runes_df = _load_runes_df()
    mask = (
        runes_df["name"].str.fullmatch("Striking", case=False)
        | runes_df["name"].str.fullmatch(r"Weapon Potency \+1", case=False)
    )
    subset = runes_df[mask].copy()
    assert not subset.empty, "Expected Striking and Potency runes in dataset"

    weapon = {"name": "Test Sword", "level": 4, "rarity": "Common", "price_text": "0 gp"}
    rng = SequenceRandom([0.0, 0.1, 0.2], seed=1337)

    fused = apply_weapon_runes(
        weapon,
        player_level=4,
        runes_df=subset,
        rng=rng,
        rune_cfg={
            "fundamental": {
                "apply_rate": 1.0,
                "potency_weights": {"0": 10, "1": 1},
            },
            "property": {"apply_rate": 0.0, "per_slot_rate": 0.0},
        },
    )

    label = fused.get("_rune_fund_label", "")
    assert "+1" in label
    assert "striking" in label.lower()
    rune_names = {r.get("name") for r in fused.get("runes", [])}
    assert "Weapon Potency +1" in rune_names
    assert "Striking" in rune_names


def test_weapon_fundamental_property_gate_can_fail():
    runes_df = _load_runes_df()
    mask = (
        runes_df["name"].str.fullmatch("Striking", case=False)
        | runes_df["name"].str.fullmatch(r"Weapon Potency \+1", case=False)
    )
    subset = runes_df[mask].copy()
    assert not subset.empty, "Expected Striking and Potency runes in dataset"

    weapon = {"name": "Test Sword", "level": 4, "rarity": "Common", "price_text": "0 gp"}
    rng = SequenceRandom([0.0, 0.1, 0.95], seed=42)

    fused = apply_weapon_runes(
        weapon,
        player_level=4,
        runes_df=subset,
        rng=rng,
        rune_cfg={
            "fundamental": {
                "apply_rate": 1.0,
                "potency_weights": {"0": 10, "1": 1},
            },
            "property": {"apply_rate": 0.0, "per_slot_rate": 0.0},
        },
    )

    rune_names = {r.get("name") for r in fused.get("runes", [])}
    assert "Weapon Potency +1" in rune_names
    assert "Striking" not in rune_names
    assert "striking" not in str(fused.get("_rune_fund_label", "")).lower()
    

def test_weapon_potency_always_applies_with_candidates():
    runes_df = _load_runes_df()
    mask = runes_df["name"].str.fullmatch(r"Weapon Potency \+1", case=False)
    subset = runes_df[mask].copy()
    assert not subset.empty, "Expected a potency rune in dataset"

    weapon = {"name": "Reliable Blade", "level": 4, "rarity": "Common", "price_text": "0 gp"}
    rng = SequenceRandom([0.0, 0.1], seed=2024)

    fused = apply_weapon_runes(
        weapon,
        player_level=4,
        runes_df=subset,
        rng=rng,
        rune_cfg={
            "fundamental": {"apply_rate": 0.1},
            "property": {"apply_rate": 0.0, "per_slot_rate": 0.0},
        },
    )

    rune_names = {r.get("name") for r in fused.get("runes", [])}
    assert "Weapon Potency +1" in rune_names


def test_armor_potency_always_applies_with_candidates():
    runes_df = _load_runes_df()
    mask = runes_df["name"].str.fullmatch(r"Armor Potency \+1", case=False)
    subset = runes_df[mask].copy()
    assert not subset.empty, "Expected an armor potency rune in dataset"

    armor = {"name": "Reliable Armor", "level": 5, "rarity": "Common", "price_text": "0 gp"}
    rng = SequenceRandom([0.0, 0.1], seed=2025)

    fused = apply_armor_runes(
        armor,
        player_level=5,
        runes_df=subset,
        rng=rng,
        rune_cfg={
            "fundamental": {"apply_rate": 0.05},
            "property": {"apply_rate": 0.0, "per_slot_rate": 0.0},
        },
    )

    rune_names = {r.get("name") for r in fused.get("runes", [])}
    assert "Armor Potency +1" in rune_names


def test_apply_armor_runes_pairs_resilient_when_gate_succeeds():
    runes_df = _load_runes_df()
    mask = (
        runes_df["name"].str.fullmatch("Resilient", case=False)
        | runes_df["name"].str.fullmatch(r"Armor Potency \+1", case=False)
    )
    subset = runes_df[mask].copy()
    assert not subset.empty, "Expected Resilient and Armor Potency runes in dataset"

    armor = {"name": "Reliable Armor", "level": 8, "rarity": "Common", "price_text": "0 gp"}
    rng = SequenceRandom([0.0, 0.1, 0.2], seed=99)

    fused = apply_armor_runes(
        armor,
        player_level=10,
        runes_df=subset,
        rng=rng,
        rune_cfg={
            "fundamental": {"apply_rate": 1.0},
            "property": {"apply_rate": 0.0, "per_slot_rate": 0.0},
        },
    )

    rune_names = {r.get("name") for r in fused.get("runes", [])}
    assert "Armor Potency +1" in rune_names
    assert "Resilient" in rune_names
    label = str(fused.get("_rune_fund_label", ""))
    assert "+1" in label and "resilient" in label.lower()


def test_armor_fundamental_property_gate_can_fail():
    runes_df = _load_runes_df()
    mask = (
        runes_df["name"].str.fullmatch("Resilient", case=False)
        | runes_df["name"].str.fullmatch(r"Armor Potency \+1", case=False)
    )
    subset = runes_df[mask].copy()
    assert not subset.empty, "Expected Resilient and Armor Potency runes in dataset"

    armor = {"name": "Reliable Armor", "level": 8, "rarity": "Common", "price_text": "0 gp"}
    rng = SequenceRandom([0.0, 0.1, 0.95], seed=199)

    fused = apply_armor_runes(
        armor,
        player_level=10,
        runes_df=subset,
        rng=rng,
        rune_cfg={
            "fundamental": {"apply_rate": 1.0},
            "property": {"apply_rate": 0.0, "per_slot_rate": 0.0},
        },
    )

    rune_names = {r.get("name") for r in fused.get("runes", [])}
    assert "Armor Potency +1" in rune_names
    assert "Resilient" not in rune_names
    assert "resilient" not in str(fused.get("_rune_fund_label", "")).lower()