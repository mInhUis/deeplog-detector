"""Merge all raw CloudTrail JSON files into a single chronologically sorted JSONL file."""

import json
import os
from pathlib import Path

RAW_DIR = Path(os.environ.get("RAW_DIR", Path(__file__).resolve().parent.parent / "data" / "raw"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", Path(__file__).resolve().parent.parent / "data" / "interim" / "flaws_merged.jsonl"))


def main() -> None:
    all_records: list[dict] = []

    json_files = sorted(RAW_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {RAW_DIR}")

    for filepath in json_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("Records", [])
        print(f"  {filepath.name}: {len(records)} records")
        all_records.extend(records)

    print(f"Total records before sort: {len(all_records)}")

    all_records.sort(key=lambda r: r.get("eventTime", ""))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(f"Wrote {len(all_records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
