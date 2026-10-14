import importlib.util
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "prepare_data", Path(__file__).resolve().parents[1] / "scripts/prepare_data.py")
prepare_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_data)


class PreparationTest(unittest.TestCase):
    def test_training_statistics_nan_and_chronology(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = np.arange(480 * 7, dtype=np.float64).reshape(480, 7)
            x[5, 2] = np.nan
            x[350:] += 10000  # held-out values must not change training statistics
            np.save(root / "raw.npy", x)
            prepare_data.prepare("ETTh1", root / "raw.npy", root / "data")
            with (root / "data/scaler.pkl").open("rb") as f:
                mean, std = pickle.load(f)
            np.testing.assert_array_equal(mean, np.nanmean(x[:288], axis=0))
            np.testing.assert_array_equal(std, np.nanstd(x[:288], axis=0))
            for name, raw in [("train", x[:288]), ("val", x[288:384]), ("test", x[384:])]:
                with (root / f"data/{name}_set.pkl").open("rb") as f:
                    prepared = pickle.load(f)
                np.testing.assert_array_equal(prepared, ((raw - mean) / std).astype(np.float32))
            with self.assertRaises(FileExistsError):
                prepare_data.prepare("ETTh1", root / "raw.npy", root / "data")


if __name__ == "__main__":
    unittest.main()
