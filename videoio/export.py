"""JSON export helper."""

import json


def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
