"""Predict with base/specialist/stack research bundles on the published format."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from ddos_iot.artifacts import atomic_output
from ddos_iot.multiclass import MulticlassBundle
from ddos_iot.experiments.study_models import SpecialistBundle, StackBundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-representation", choices=["published-scaled-23"], required=True,
                        help="These research bundles require the authors' selected/scaled representation, not raw flow values.")
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() in (args.input.resolve(), args.model.resolve()):
        raise FileExistsError("Choose a new prediction output path.")
    bundle = joblib.load(args.model)
    if not isinstance(bundle, (MulticlassBundle, SpecialistBundle, StackBundle)):
        raise ValueError("Expected a trusted local study bundle.")
    with atomic_output(args.output) as stream:
        stream.write(b"predicted_label\n")
        for chunk in pd.read_csv(args.input, chunksize=100_000):
            predicted = pd.DataFrame({"predicted_label": bundle.predict(chunk)})
            stream.write(predicted.to_csv(index=False, header=False).encode("utf-8"))
