import subprocess
import sys

import numpy as np


def test_offline_demo_contract(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "so101_semantic.demo", "--output", str(tmp_path)], check=True
    )
    with np.load(tmp_path / "cost_field.npz", allow_pickle=False) as field:
        assert field["costs"].shape == field["observed"].shape == (20, 20, 60)
        assert field["costs"].dtype == np.float32
        assert field["observed"].dtype == bool
        assert field["observed"].any()
        assert not field["observed"].all()
        assert np.isfinite(field["costs"]).all()
        assert field["frame"].item() == "synthetic_base"
        assert field["axis_order"].item() == "xyz"
    assert (tmp_path / "cost_field.png").stat().st_size > 0
