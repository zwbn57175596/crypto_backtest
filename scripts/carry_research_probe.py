"""Run the read-only OKX probe. Needs a READ-ONLY key in env for account facts."""
from __future__ import annotations

import argparse
import os
import sys

import okx.Account as Account
import okx.PublicData as PublicData

from carry.okx_probe import collect_probe, render_markdown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-only", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    flag = "1" if args.demo else "0"
    public = PublicData.PublicAPI(flag=flag, debug=False)
    account = None
    if not args.public_only:
        key, secret, phrase = (os.environ.get(k, "") for k in
                               ("OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE"))
        if not (key and secret and phrase):
            sys.exit("Set OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSPHRASE (read-only key) or use --public-only")
        account = Account.AccountAPI(key, secret, phrase, False, flag=flag, debug=False)

    md = render_markdown(collect_probe(public, account))
    print(md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)


if __name__ == "__main__":
    main()
