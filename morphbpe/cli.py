"""Command-line interface for MorphBPE."""

import argparse
import sys

from morphbpe.tokenizer import MorphBPETokenizer


def main():
    parser = argparse.ArgumentParser(
        prog="morphbpe",
        description="MorphBPE: Morphologically-aware BPE tokenizer for Polish",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train a tokenizer from a text file")
    p_train.add_argument("corpus", help="Path to plain-text corpus (one doc per line)")
    p_train.add_argument("-v", "--vocab-size", type=int, default=1000)
    p_train.add_argument("-a", "--alpha", type=float, default=2.0)
    p_train.add_argument(
        "-c", "--constraint", choices=["suffix", "prefix", "both"], default="suffix"
    )
    p_train.add_argument(
        "-o", "--output", default="morphbpe-model", help="Output directory"
    )

    # ── encode ───────────────────────────────────────────────────────
    p_enc = sub.add_parser("encode", help="Encode text using a trained tokenizer")
    p_enc.add_argument("model", help="Tokenizer directory")
    p_enc.add_argument(
        "text", nargs="?", default=None, help="Text to encode (or stdin)"
    )

    # ── info ─────────────────────────────────────────────────────────
    p_info = sub.add_parser("info", help="Show tokenizer info")
    p_info.add_argument("model", help="Tokenizer directory")

    args = parser.parse_args()

    if args.command == "train":
        _cmd_train(args)
    elif args.command == "encode":
        _cmd_encode(args)
    elif args.command == "info":
        _cmd_info(args)


def _cmd_train(args):
    tok = MorphBPETokenizer(
        vocab_size=args.vocab_size,
        alpha=args.alpha,
        constraint=args.constraint,
    )
    tok.train_from_file(args.corpus)
    tok.save(args.output)
    print(f"Saved to {args.output}/")


def _cmd_encode(args):
    tok = MorphBPETokenizer.load(args.model)
    if args.text:
        lines = [args.text]
    else:
        lines = sys.stdin.read().splitlines()

    for line in lines:
        tokens = []
        for word in line.split():
            tokens.extend(tok.encode_word(word))
        print(" ".join(tokens))


def _cmd_info(args):
    tok = MorphBPETokenizer.load(args.model)
    print(f"Vocab size:  {tok.get_vocab_size()}")
    print(f"Merges:      {len(tok.merges)}")
    print(f"Alpha:       {tok.alpha}")
    print(f"Constraint:  {tok.constraint}")
    print(f"Morfeusz:    {tok._backend.has_morfeusz}")


if __name__ == "__main__":
    main()
