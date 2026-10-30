"""Generate a small synthetic NPZ for the offline integration example."""
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
rng = np.random.default_rng(7)
X = rng.uniform(-1.0, 1.0, size=(600, 1))
X_test = rng.uniform(-1.0, 1.0, size=(200, 1))
X_ood = rng.uniform(1.0, 2.0, size=(200, 1))
f = lambda x: x[:, 0] ** 2 + x[:, 0] + 0.5
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez(args.output, X=X, y=f(X), feature_names=np.array(["x0"]),
         X_test_id=X_test, y_test_id=f(X_test), X_test_ood=X_ood, y_test_ood=f(X_ood))
