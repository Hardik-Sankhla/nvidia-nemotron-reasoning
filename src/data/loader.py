"""Data loader utilities for the novel benchmark."""
import json
from pathlib import Path

def load_json(path):
    p = Path(path)
    with p.open('r', encoding='utf8') as f:
        return json.load(f)
