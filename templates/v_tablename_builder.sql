DROP VIEW IF EXISTS v_tattoo_norm;

CREATE VIEW v_tattoo_norm AS
WITH base AS (
  SELECT
    'tattoo'                         AS category,
    'tattoos'                        AS source_table,
    CAST(NULL AS INTEGER)            AS source_id,   -- set to m.id if you have one
    COALESCE(m.Name, '')             AS name,
    COALESCE(m.ItemLevel, 0)         AS level,
    COALESCE(m.Rarity, 'Common')     AS rarity,
    COALESCE(m.Type, '')             AS type,
    'tattoo'                         AS subtype,
    COALESCE(m.Cost, '')             AS price_text,
    COALESCE(m.Traits, '')           AS tags,
    COALESCE(m.Bulk, '0')            AS Bulk,
    COALESCE(m.Source, '')           AS Source,

    CAST(COALESCE(NULLIF(TRIM(m.Adventuring),''),'0') AS INTEGER)   AS Adventuring_i,
    CAST(COALESCE(NULLIF(TRIM(m.Alchemist),''),'0')   AS INTEGER)   AS Alchemist_i,
    CAST(COALESCE(NULLIF(TRIM(m.Arcane),''),'0')      AS INTEGER)   AS Arcane_i,
    CAST(COALESCE(NULLIF(TRIM(m.Blacksmith),''),'0')  AS INTEGER)   AS Blacksmith_i,
    CAST(COALESCE(NULLIF(TRIM(m.Scribe),''),'0')      AS INTEGER)   AS Scribe_i,
    CAST(COALESCE(NULLIF(TRIM(m.Bowyer),''),'0')      AS INTEGER)   AS Bowyer_i,
    CAST(COALESCE(NULLIF(TRIM(m.General),''),'0')     AS INTEGER)   AS General_i,
    CAST(COALESCE(NULLIF(TRIM(m.Jewelry),''),'0')     AS INTEGER)   AS Jewelry_i,
    CAST(COALESCE(NULLIF(TRIM(m.Leatherworker),''),'0') AS INTEGER) AS Leatherworker_i,
    CAST(COALESCE(NULLIF(TRIM(m.Music_Games),''),'0') AS INTEGER)   AS Music_Games_i,
    CAST(COALESCE(NULLIF(TRIM(m.Illicit),''),'0')     AS INTEGER)   AS Illicit_i,
    CAST(COALESCE(NULLIF(TRIM(m.Tailor),''),'0')      AS INTEGER)   AS Tailor_i,
    CAST(COALESCE(NULLIF(TRIM(m.Temple),''),'0')      AS INTEGER)   AS Temple_i,
    CAST(COALESCE(NULLIF(TRIM(m.Armorer),''),'0')     AS INTEGER)   AS Armorer_i,
    CAST(COALESCE(NULLIF(TRIM(m.Weaponsmith),''),'0') AS INTEGER)   AS Weaponsmith_i,
    CAST(COALESCE(NULLIF(TRIM(m.Tattooist),''),'0')   AS INTEGER)   AS Tattooist_i
  FROM Tattoos AS m
),
fanout(shop_type_val, col_i) AS (
  VALUES
    ('adventuring',   'Adventuring_i'),
    ('alchemist',     'Alchemist_i'),
    ('arcane',        'Arcane_i'),
    ('blacksmith',    'Blacksmith_i'),
    ('scribe',        'Scribe_i'),
    ('bowyer',        'Bowyer_i'),
    ('general',       'General_i'),
    ('jewelry',       'Jewelry_i'),
    ('leatherworker', 'Leatherworker_i'),
    ('music_games',   'Music_Games_i'),
    ('illicit',       'Illicit_i'),
    ('tailor',        'Tailor_i'),
    ('temple',        'Temple_i'),
    ('armorer',       'Armorer_i'),
    ('weaponsmith',   'Weaponsmith_i'),
    ('tattooist',     'Tattooist_i')
)
SELECT
  b.category,
  b.source_table,
  b.source_id,
  b.name,
  b.level,
  b.rarity,
  b.type,
  b.subtype,
  b.price_text,
  b.tags,
  b.Bulk,
  b.Source,
  f.shop_type_val AS shop_type,
  CASE f.col_i
    WHEN 'Adventuring_i'   THEN b.Adventuring_i
    WHEN 'Alchemist_i'     THEN b.Alchemist_i
    WHEN 'Arcane_i'        THEN b.Arcane_i
    WHEN 'Blacksmith_i'    THEN b.Blacksmith_i
    WHEN 'Scribe_i'        THEN b.Scribe_i
    WHEN 'Bowyer_i'        THEN b.Bowyer_i
    WHEN 'General_i'       THEN b.General_i
    WHEN 'Jewelry_i'       THEN b.Jewelry_i
    WHEN 'Leatherworker_i' THEN b.Leatherworker_i
    WHEN 'Music_Games_i'   THEN b.Music_Games_i
    WHEN 'Illicit_i'       THEN b.Illicit_i
    WHEN 'Tailor_i'        THEN b.Tailor_i
    WHEN 'Temple_i'        THEN b.Temple_i
    WHEN 'Armorer_i'       THEN b.Armorer_i
    WHEN 'Weaponsmith_i'   THEN b.Weaponsmith_i
    WHEN 'Tattooist_i'     THEN b.Tattooist_i
    ELSE 0
  END AS stock_flag
FROM base b
CROSS JOIN fanout f
WHERE CASE f.col_i
    WHEN 'Adventuring_i'   THEN b.Adventuring_i
    WHEN 'Alchemist_i'     THEN b.Alchemist_i
    WHEN 'Arcane_i'        THEN b.Arcane_i
    WHEN 'Blacksmith_i'    THEN b.Blacksmith_i
    WHEN 'Scribe_i'        THEN b.Scribe_i
    WHEN 'Bowyer_i'        THEN b.Bowyer_i
    WHEN 'General_i'       THEN b.General_i
    WHEN 'Jewelry_i'       THEN b.Jewelry_i
    WHEN 'Leatherworker_i' THEN b.Leatherworker_i
    WHEN 'Music_Games_i'   THEN b.Music_Games_i
    WHEN 'Illicit_i'       THEN b.Illicit_i
    WHEN 'Tailor_i'        THEN b.Tailor_i
    WHEN 'Temple_i'        THEN b.Temple_i
    WHEN 'Armorer_i'       THEN b.Armorer_i
    WHEN 'Weaponsmith_i'   THEN b.Weaponsmith_i
    WHEN 'Tattooist_i'     THEN b.Tattooist_i
    ELSE 0
  END IN (1,2);
