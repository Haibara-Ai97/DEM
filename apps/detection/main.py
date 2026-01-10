from __future__ import annotations

import argparse

from . import train_det, eval_det


def parse_args(argv=None):
    parser = argparse.ArgumentParser("DEM detection entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train detection models")
    train_parser.add_argument("args", nargs=argparse.REMAINDER)

    eval_parser = subparsers.add_parser("eval", help="Evaluate detection models")
    eval_parser.add_argument("args", nargs=argparse.REMAINDER)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "train":
        train_det.main(args.args)
        return
    if args.command == "eval":
        eval_det.main(args.args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
