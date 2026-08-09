from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from plots import INPUT, generate_figures, read_significance


def test_generate_figures_writes_exact_expected_pngs(tmp_path: Path) -> None:
    generated = generate_figures(INPUT, tmp_path)
    assert {path.name for path in generated} == {
        "degradation_by_dimension.png",
        "discordance_decomposition.png",
    }
    assert set(tmp_path.iterdir()) == set(generated)
    for path in generated:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        image = plt.imread(path)
        assert image.ndim == 3
        assert image.shape[0] > 0 and image.shape[1] > 0
    assert plt.get_fignums() == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (",23,12,35,", ",23,12,34,"),
        (",False\n", ",maybe\n"),
        ("0.004678860059549128", "nan"),
        (",True\n", ",False\n"),
        (",2351,", ",2350,"),
        (",0.8315610378562314,0.8268821777966823,", ",0.8,0.8268821777966823,"),
    ],
)
def test_read_significance_rejects_inconsistent_rows(tmp_path: Path, old: str, new: str) -> None:
    malformed = tmp_path / "significance.csv"
    malformed.write_text(INPUT.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ValueError):
        read_significance(malformed)
