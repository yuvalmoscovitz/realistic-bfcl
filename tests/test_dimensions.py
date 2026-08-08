from __future__ import annotations

from realistic_bfcl.common import (
    article_facing_dimensions,
    realism_dimension_configs,
)


def test_dimension_config_reconciles_implemented_and_article_scopes() -> None:
    dimensions = realism_dimension_configs()
    assert len(dimensions) == 10
    assert article_facing_dimensions() == {
        "argumentative_challenge",
        "cursing",
        "irrelevant_context",
        "pasted_context_block",
        "removed_spaces",
        "telegraphic_request",
        "typos",
    }

    excluded = {
        name for name, config in dimensions.items() if not config["article_facing"]
    }
    assert excluded == {
        "argumentative_sandwich",
        "distractor_sandwich",
        "profane_sandwich",
    }
    assert all(dimensions[name]["status"] == "evaluated" for name in article_facing_dimensions())
    assert all(dimensions[name]["status"] == "pilot" for name in excluded)
    assert all(dimensions[name]["article_exclusion_reason"] for name in excluded)
