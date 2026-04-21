"""Validate that flaws_merged.jsonl is sorted chronologically by eventTime."""

import json
import os
from pathlib import Path

INPUT_PATH = Path(
    os.environ.get(
        "INPUT_PATH",
        Path(__file__).resolve().parent.parent / "data" / "interim" / "flaws_merged.jsonl",
    )
)


def main() -> None:
    prev_time: str = ""
    count: int = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            record: dict = json.loads(line)
            event_time: str = record.get("eventTime", "")

            assert event_time >= prev_time, (
                f"Line {lineno}: eventTime '{event_time}' is before "
                f"previous '{prev_time}' — chronological order violated"
            )

            prev_time = event_time
            count += 1

            if count % 500_000 == 0:
                print(f"  ...validated {count:,} records so far")

    print(f"\nSUCCESS: All {count:,} records are in chronological order.")


if __name__ == "__main__":
    main()
