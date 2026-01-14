from pathlib import Path

BASE = Path(".")
SPLITS = ["train", "valid", "test"]

for s in SPLITS:
    labels_dir = BASE / s / "labels"
    if not labels_dir.exists():
        continue
    total = pos = neg = 0
    for f in labels_dir.glob("*.txt"):
        total += 1
        has_box = any(l.strip() for l in f.read_text(encoding="utf-8").splitlines())
        if has_box:
            pos += 1
        else:
            neg += 1
    print(f"{s}: total={total} | pos={pos} ({pos/total:.1%}) | neg={neg} ({neg/total:.1%})")
