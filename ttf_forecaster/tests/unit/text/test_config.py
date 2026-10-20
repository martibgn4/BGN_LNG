"""configs/text.yaml and the prompt draft stay consistent with the code and with governance."""

from __future__ import annotations

from ttf_forecaster.settings import PROJECT_ROOT, load_config
from ttf_forecaster.text.validation import DEFAULT_MCMD_UNITS


def test_text_config_is_safe_by_default():
    cfg = load_config("text")
    assert cfg["extraction"]["enabled"] is False
    assert cfg["extraction"]["external_model_approved"] is False
    assert cfg["features"]["clean_backtest_eligible"] is False
    for name, src in cfg["sources"].items():
        assert src["authorised"] is False, name
        assert src["status"] == "requires_user_input", name
        assert src["access"] is None, name  # no invented endpoints


def test_units_and_prompt_match_code():
    cfg = load_config("text")
    assert tuple(cfg["validation"]["accepted_mcmd_units"]) == DEFAULT_MCMD_UNITS
    prompt = PROJECT_ROOT / "prompts" / f"{cfg['extraction']['prompt_version']}.md"
    text = prompt.read_text(encoding="utf-8")
    assert "NOT EXECUTED" in text and "null" in text and "verbatim" in text
