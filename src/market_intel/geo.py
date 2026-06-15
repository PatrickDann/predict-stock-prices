"""Country-name → geographic centroid lookup for the world event map.

GDELT's DOC 2.0 API reports a human-readable English ``sourcecountry`` (e.g.
``"United States"``, ``"Slovak Republic"``) but no coordinates, so to place an
article on a map we map that name to an approximate country centroid.

``_CENTROIDS`` is keyed by the lowercased country name → ``(lat, lon, iso2)``.
``_ALIASES`` maps lowercased spelling variants (including GDELT's own quirks,
e.g. it emits "Slovak Republic" rather than "Slovakia") to a canonical key.
``country_centroid`` normalizes the input and resolves through both.
"""

from __future__ import annotations

# Approximate country centroids (latitude, longitude, ISO 3166-1 alpha-2).
# Keys are lowercased; coordinates are rough geographic centers — precise enough
# to place a marker on a world map, not for navigation.
_CENTROIDS: dict[str, tuple[float, float, str]] = {
    # --- Americas ---
    "united states": (39.8, -98.6, "US"),
    "canada": (56.1, -106.3, "CA"),
    "mexico": (23.6, -102.5, "MX"),
    "brazil": (-14.2, -51.9, "BR"),
    "argentina": (-38.4, -63.6, "AR"),
    "chile": (-35.7, -71.5, "CL"),
    "peru": (-9.2, -75.0, "PE"),
    "colombia": (4.6, -74.3, "CO"),
    "venezuela": (6.4, -66.6, "VE"),
    "bolivia": (-16.3, -63.6, "BO"),
    "ecuador": (-1.8, -78.2, "EC"),
    "paraguay": (-23.4, -58.4, "PY"),
    "uruguay": (-32.5, -55.8, "UY"),
    "cuba": (21.5, -77.8, "CU"),
    "dominican republic": (18.7, -70.2, "DO"),
    "guatemala": (15.8, -90.2, "GT"),
    "honduras": (15.2, -86.2, "HN"),
    "el salvador": (13.8, -88.9, "SV"),
    "nicaragua": (12.9, -85.2, "NI"),
    "panama": (8.5, -80.8, "PA"),
    "costa rica": (9.7, -83.8, "CR"),
    "jamaica": (18.1, -77.3, "JM"),
    "haiti": (19.1, -72.3, "HT"),
    "trinidad and tobago": (10.7, -61.2, "TT"),
    # --- Europe ---
    "united kingdom": (54.0, -2.0, "GB"),
    "ireland": (53.4, -8.2, "IE"),
    "france": (46.2, 2.2, "FR"),
    "germany": (51.2, 10.5, "DE"),
    "italy": (41.9, 12.6, "IT"),
    "spain": (40.5, -3.7, "ES"),
    "portugal": (39.4, -8.2, "PT"),
    "netherlands": (52.1, 5.3, "NL"),
    "belgium": (50.5, 4.5, "BE"),
    "switzerland": (46.8, 8.2, "CH"),
    "austria": (47.5, 14.6, "AT"),
    "poland": (51.9, 19.1, "PL"),
    "ukraine": (48.4, 31.2, "UA"),
    "sweden": (60.1, 18.6, "SE"),
    "norway": (60.5, 8.5, "NO"),
    "finland": (61.9, 25.7, "FI"),
    "denmark": (56.3, 9.5, "DK"),
    "iceland": (64.96, -19.0, "IS"),
    "greece": (39.07, 21.8, "GR"),
    "czech republic": (49.8, 15.5, "CZ"),
    "slovak republic": (48.7, 19.7, "SK"),
    "hungary": (47.2, 19.5, "HU"),
    "romania": (45.9, 24.97, "RO"),
    "bulgaria": (42.7, 25.5, "BG"),
    "croatia": (45.1, 15.2, "HR"),
    "serbia": (44.0, 21.0, "RS"),
    "bosnia and herzegovina": (43.9, 17.7, "BA"),
    "slovenia": (46.15, 14.99, "SI"),
    "albania": (41.15, 20.17, "AL"),
    "north macedonia": (41.6, 21.7, "MK"),
    "montenegro": (42.7, 19.4, "ME"),
    "kosovo": (42.6, 20.9, "XK"),
    "cyprus": (35.1, 33.4, "CY"),
    "malta": (35.9, 14.4, "MT"),
    "luxembourg": (49.8, 6.1, "LU"),
    "estonia": (58.6, 25.0, "EE"),
    "latvia": (56.9, 24.6, "LV"),
    "lithuania": (55.2, 23.9, "LT"),
    "belarus": (53.7, 27.95, "BY"),
    "moldova": (47.4, 28.4, "MD"),
    "russia": (61.5, 105.3, "RU"),
    # --- Middle East & Central Asia ---
    "turkey": (39.0, 35.2, "TR"),
    "israel": (31.05, 34.85, "IL"),
    "saudi arabia": (23.9, 45.1, "SA"),
    "united arab emirates": (23.4, 53.8, "AE"),
    "iran": (32.4, 53.7, "IR"),
    "iraq": (33.2, 43.7, "IQ"),
    "syria": (34.8, 38.99, "SY"),
    "jordan": (30.6, 36.2, "JO"),
    "lebanon": (33.85, 35.86, "LB"),
    "qatar": (25.35, 51.18, "QA"),
    "kuwait": (29.3, 47.5, "KW"),
    "bahrain": (26.0, 50.55, "BH"),
    "oman": (21.5, 55.9, "OM"),
    "yemen": (15.55, 48.5, "YE"),
    "azerbaijan": (40.14, 47.58, "AZ"),
    "armenia": (40.07, 45.0, "AM"),
    "georgia": (42.3, 43.4, "GE"),
    "kazakhstan": (48.0, 66.9, "KZ"),
    "uzbekistan": (41.4, 64.6, "UZ"),
    "turkmenistan": (38.97, 59.6, "TM"),
    "kyrgyzstan": (41.2, 74.8, "KG"),
    "tajikistan": (38.9, 71.3, "TJ"),
    "afghanistan": (33.9, 67.7, "AF"),
    # --- Asia & Oceania ---
    "china": (35.9, 104.2, "CN"),
    "japan": (36.2, 138.3, "JP"),
    "south korea": (35.9, 127.8, "KR"),
    "north korea": (40.3, 127.5, "KP"),
    "india": (20.6, 78.96, "IN"),
    "pakistan": (30.4, 69.3, "PK"),
    "bangladesh": (23.7, 90.4, "BD"),
    "sri lanka": (7.9, 80.8, "LK"),
    "nepal": (28.4, 84.1, "NP"),
    "bhutan": (27.5, 90.4, "BT"),
    "maldives": (3.2, 73.2, "MV"),
    "indonesia": (-0.8, 113.9, "ID"),
    "malaysia": (4.2, 101.98, "MY"),
    "thailand": (15.9, 100.99, "TH"),
    "vietnam": (14.06, 108.3, "VN"),
    "philippines": (12.9, 121.8, "PH"),
    "singapore": (1.35, 103.8, "SG"),
    "myanmar": (21.9, 95.96, "MM"),
    "cambodia": (12.6, 104.99, "KH"),
    "laos": (19.85, 102.5, "LA"),
    "brunei": (4.5, 114.7, "BN"),
    "mongolia": (46.9, 103.8, "MN"),
    "taiwan": (23.7, 121.0, "TW"),
    "hong kong": (22.35, 114.1, "HK"),
    "australia": (-25.3, 133.8, "AU"),
    "new zealand": (-40.9, 174.9, "NZ"),
    "papua new guinea": (-6.3, 143.96, "PG"),
    "fiji": (-17.7, 178.0, "FJ"),
    # --- Africa ---
    "egypt": (26.8, 30.8, "EG"),
    "libya": (26.3, 17.2, "LY"),
    "algeria": (28.0, 1.7, "DZ"),
    "morocco": (31.8, -7.1, "MA"),
    "tunisia": (33.9, 9.6, "TN"),
    "nigeria": (9.1, 8.7, "NG"),
    "south africa": (-30.6, 22.9, "ZA"),
    "kenya": (0.0, 37.9, "KE"),
    "ethiopia": (9.1, 40.5, "ET"),
    "ghana": (7.95, -1.02, "GH"),
    "senegal": (14.5, -14.5, "SN"),
    "ivory coast": (7.5, -5.5, "CI"),
    "mali": (17.6, -4.0, "ML"),
    "cameroon": (7.4, 12.4, "CM"),
    "democratic republic of the congo": (-4.0, 21.8, "CD"),
    "republic of the congo": (-0.2, 15.8, "CG"),
    "rwanda": (-1.9, 29.9, "RW"),
    "uganda": (1.4, 32.3, "UG"),
    "tanzania": (-6.4, 34.9, "TZ"),
    "sudan": (12.9, 30.2, "SD"),
    "south sudan": (6.9, 31.3, "SS"),
    "angola": (-11.2, 17.9, "AO"),
    "mozambique": (-18.7, 35.5, "MZ"),
    "zimbabwe": (-19.0, 29.2, "ZW"),
    "zambia": (-13.1, 27.8, "ZM"),
    "botswana": (-22.3, 24.7, "BW"),
    "namibia": (-22.96, 18.5, "NA"),
    "madagascar": (-18.8, 47.0, "MG"),
}

# Spelling variants → canonical key. Covers GDELT's own naming quirks and common
# alternates so a different source feed still resolves.
_ALIASES: dict[str, str] = {
    "united states of america": "united states",
    "usa": "united states",
    "us": "united states",
    "uk": "united kingdom",
    "great britain": "united kingdom",
    "russian federation": "russia",
    "slovakia": "slovak republic",
    "czechia": "czech republic",
    "macedonia": "north macedonia",
    "korea, south": "south korea",
    "korea, north": "north korea",
    "burma": "myanmar",
    "cote d'ivoire": "ivory coast",
    "côte d'ivoire": "ivory coast",
    "uae": "united arab emirates",
    "congo (kinshasa)": "democratic republic of the congo",
    "dr congo": "democratic republic of the congo",
    "congo (brazzaville)": "republic of the congo",
    "congo": "republic of the congo",
}


def country_centroid(name: str | None) -> tuple[float, float, str] | None:
    """Resolve a country name to ``(lat, lon, iso2)``, or ``None`` if unknown.

    Matching is case-insensitive and whitespace-tolerant, and follows aliases
    (e.g. "Slovakia" → "Slovak Republic", "USA" → "United States").
    """
    if not name:
        return None
    key = " ".join(name.strip().lower().split())
    if key in _CENTROIDS:
        return _CENTROIDS[key]
    canonical = _ALIASES.get(key)
    if canonical:
        return _CENTROIDS.get(canonical)
    return None
