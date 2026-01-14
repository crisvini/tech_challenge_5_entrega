from pathlib import Path

BASE = Path(".")  # raiz onde tem train/ valid/ test/

SPLITS = ["train", "valid", "test"]

# Remapeamento do seu cenário
# Scissors: era 1 -> vira 0
# Knife: era 2 -> vira 1
REMAP = {1: 0, 2: 1}

def fix_labels(split: str):
    labels_dir = BASE / split / "labels"
    if not labels_dir.exists():
        return 0, 0, 0

    files = list(labels_dir.glob("*.txt"))
    changed = 0
    dropped = 0
    bad = 0

    for f in files:
        out_lines = []
        lines = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        for line in lines:
            parts = line.split()
            try:
                cls = int(float(parts[0]))
            except:
                bad += 1
                continue

            if cls in REMAP:
                parts[0] = str(REMAP[cls])
                out_lines.append(" ".join(parts))
            else:
                # se sobrou alguma classe estranha, descartamos
                dropped += 1

        new_text = "\n".join(out_lines) + ("\n" if out_lines else "")
        if new_text != f.read_text(encoding="utf-8"):
            changed += 1
        f.write_text(new_text, encoding="utf-8")

    return len(files), changed, dropped

for s in SPLITS:
    total, changed, dropped = fix_labels(s)
    if total:
        print(f"{s}: labels={total} | arquivos alterados={changed} | boxes descartadas={dropped}")
