"""Report exact point counts for the Qdrant server configured in the environment."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from qdrant_client import QdrantClient


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", help="Check only this collection (default: all collections)")
    args = parser.parse_args(argv)

    url = os.environ.get("QDRANT_URL")
    if not url:
        print("QDRANT_URL is required", file=sys.stderr)
        return 2

    try:
        client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY") or None)
        names = (
            [args.collection]
            if args.collection
            else sorted(collection.name for collection in client.get_collections().collections)
        )
        counts = [(name, client.count(collection_name=name, exact=True).count) for name in names]
    except Exception as exc:
        print(f"Qdrant check failed ({type(exc).__name__})", file=sys.stderr)
        return 1

    if not counts:
        print("No collections found")
    for name, count in counts:
        print(f"{name}: {count}")
    print(f"Has data: {'yes' if any(count > 0 for _, count in counts) else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
