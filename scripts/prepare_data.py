"""Small helper to prepare dataset slices for prompting experiments."""
import json
from pathlib import Path

def slice_dataset(src_path, out_path, n=100):
    p = Path(src_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with p.open('r', encoding='utf8') as f:
        data = json.load(f)
    with out.open('w', encoding='utf8') as f:
        json.dump(data[:n], f, indent=2)

if __name__ == '__main__':
    print('Use slice_dataset(src, out, n)')
