import pytest

from presslens.sim import realism


@pytest.mark.skipif(not (realism.MATCHES / "index.parquet").exists(), reason="season not generated")
def test_realism_report_builds_from_the_stored_season():
    text = realism.build_report()
    for section in ("## Match statistics", "## Physical", "## Attack funnel", "## xG calibration", "## League table"):
        assert section in text
