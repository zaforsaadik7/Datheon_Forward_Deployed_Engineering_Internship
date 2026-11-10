from src.config import (
    get_settings,
    DEFAULT_TIER,
    TIERS,
    DEPARTMENTS,
    COSINE_COLLECTION_KWARGS,
)


def test_settings_defaults():
    settings = get_settings()
    assert settings.chunk_size == 900
    assert settings.chunk_overlap == 120
    assert settings.collection_name == "enterprise_docs"
    assert DEFAULT_TIER == 4
    assert TIERS == (1, 2, 3, 4)
    assert DEPARTMENTS == frozenset({
        "engineering",
        "finance",
        "legal",
        "hr",
        "executive",
        "operations",
    })
    assert "metadata" in COSINE_COLLECTION_KWARGS or "configuration" in COSINE_COLLECTION_KWARGS

