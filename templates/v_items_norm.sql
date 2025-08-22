DROP VIEW IF EXISTS v_items_norm;

CREATE VIEW v_items_norm AS
SELECT * FROM v_mundane_norm
UNION ALL
SELECT * FROM v_weapon_norm
UNION ALL
SELECT * FROM v_armor_norm
UNION ALL
SELECT * FROM v_shield_norm
UNION ALL
SELECT * FROM v_spec_magic_armor_norm
UNION ALL
SELECT * FROM v_spec_magic_shield_norm
UNION ALL
SELECT * FROM v_spec_magic_weapon_norm
UNION ALL
SELECT * FROM v_grimoire_norm
UNION ALL
SELECT * FROM v_held_norm
UNION ALL
SELECT * FROM v_rune_norm
UNION ALL
SELECT * FROM v_snare_norm
UNION ALL
SELECT * FROM v_spellheart_norm
UNION ALL
SELECT * FROM v_staff_wand_norm
UNION ALL
SELECT * FROM v_worn_norm
UNION ALL
SELECT * FROM v_materials_norm
UNION ALL
SELECT * FROM v_adjustments_norm
UNION ALL
SELECT * FROM v_scrolls_norm
UNION ALL
SELECT * FROM v_alchemical_items_norm
UNION ALL
SELECT * FROM v_ccstructure_items_norm
UNION ALL
SELECT * FROM v_consumables_norm
UNION ALL
SELECT * FROM v_tattoo_norm
UNION ALL
SELECT * FROM v_ritual_norm
;
