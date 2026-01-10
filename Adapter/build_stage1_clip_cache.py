import argparse

from dem.data.cache import build_clip_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--domain_vocab", type=str, required=True)
    ap.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch16")
    ap.add_argument("--out_dir", type=str, default="data/stage1_clip_cache")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16","float32"])
    args = ap.parse_args()

    build_clip_cache(
        train_csv=args.train_csv,
        domain_vocab=args.domain_vocab,
        clip_name=args.clip_name,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        topk=args.topk,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
