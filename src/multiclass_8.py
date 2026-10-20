"""Paper-style eight-family CICIoT2023 classification (separate output directory)."""

from multiclass import parse_args
from ddos_iot.multiclass import run_multiclass_training
from ddos_iot.utils import configure_logging


def main() -> None:
    args = parse_args(default_num_classes=8)
    configure_logging(args.verbose)
    run_multiclass_training(args)


if __name__ == "__main__":
    main()
