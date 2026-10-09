"""Known failures of the legacy code, kept visible as xfail (not fixed: castlegen.legacy is not developed)."""
import pytest

_PIPELINE = "NameError at castlegen/legacy/pipeline.py:219, _tiles (name 'an' is not defined)"
_CASTLE_EX = "needs cache/castle_ex.npy, which is not in the repo"
KNOWN = {
    "test_blockconn.py": {n: _CASTLE_EX for n in ("test_parts_match_reference", "test_own_pattern_scores_only_orphans",
                                                  "test_cand_scores", "test_site_scores_match_parts")},
    "test_blockfield.py": {"test_closed_form_matches_enumeration": _CASTLE_EX},
    "test_flow.py": {"test_end_to_end_mini_wilds": _PIPELINE},
    "test_promise.py": {n: _PIPELINE for n in ("test_determinism", "test_end_to_end", "test_bottom_up")},
    "test_support.py": {"test_end_to_end_mini_cliffs": _PIPELINE},
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path.parent.name != "legacy":
            continue
        reason = KNOWN.get(item.path.name, {}).get(item.originalname)
        if reason is not None:
            item.add_marker(pytest.mark.xfail(reason=reason))
