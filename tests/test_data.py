from pathlib import Path

import pytest

from subscription_analytics import data


def test_checksum_mismatch_is_refused(tmp_path: Path):
    src = data.Source("x.csv", "https://example.invalid/x.csv", "0" * 64)
    (tmp_path / "x.csv").write_text("a,b\n1,2\n")
    with pytest.raises(data.DataError):
        data.fetch(src, tmp_path)
