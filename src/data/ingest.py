from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_and_clean import find_telco_csv, load_raw_data


REPORT_PATH = PROJECT_ROOT / "reports" / "ingestion_metadata.json"


def main() -> None:
    data_path = find_telco_csv()
    data = load_raw_data(data_path)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "source_path": str(data_path),
                "rows": int(data.shape[0]),
                "columns": int(data.shape[1]),
                "column_names": data.columns.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Ingested {data.shape[0]} rows from {data_path}")


if __name__ == "__main__":
    main()
