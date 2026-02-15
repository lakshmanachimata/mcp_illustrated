#!/usr/bin/env python3
"""Generate OpenAPI (Swagger) JSON and YAML from the FastAPI app."""
import json
from pathlib import Path

import yaml
from main import app

OUT_DIR = Path(__file__).resolve().parent


def main():
    schema = app.openapi()
    out_json = OUT_DIR / "openapi.json"
    out_yaml = OUT_DIR / "swagger.yaml"
    with open(out_json, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Wrote {out_json}")
    with open(out_yaml, "w") as f:
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Wrote {out_yaml}")


if __name__ == "__main__":
    main()
