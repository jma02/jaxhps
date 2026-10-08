"""Atomic JSON snapshots for benchmark workers and their local collectors."""

import json


def save_result(path, row):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(row))
    temporary.replace(path)
