"""Country-name → centroid resolution for the world event map."""

from market_intel.geo import _CENTROIDS, country_centroid


def test_known_country():
    lat, lon, iso = country_centroid("United States")
    assert iso == "US"
    assert 24 < lat < 50 and -125 < lon < -66  # roughly within the US


def test_case_and_whitespace_insensitive():
    assert country_centroid("  united KINGDOM  ") == country_centroid("United Kingdom")


def test_gdelt_naming_quirk_resolves():
    # GDELT emits "Slovak Republic"; the alias "Slovakia" resolves to the same point.
    assert country_centroid("Slovak Republic")[2] == "SK"
    assert country_centroid("Slovakia") == country_centroid("Slovak Republic")


def test_aliases():
    assert country_centroid("USA")[2] == "US"
    assert country_centroid("Czechia") == country_centroid("Czech Republic")
    assert country_centroid("Burma") == country_centroid("Myanmar")


def test_unknown_returns_none():
    assert country_centroid("Narnia") is None
    assert country_centroid("") is None
    assert country_centroid(None) is None


def test_all_centroids_are_valid_coordinates():
    for name, (lat, lon, iso) in _CENTROIDS.items():
        assert -90 <= lat <= 90, name
        assert -180 <= lon <= 180, name
        assert len(iso) == 2 and iso.isupper(), name
