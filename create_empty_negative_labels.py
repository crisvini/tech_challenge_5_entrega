from pathlib import Path

BASE = Path(".")  
SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

KEYWORDS_POS = ["knife", "scissors"]

created = 0
skipped = 0

for split in SPLITS:
    img_dir = BASE / split / "images"
    lbl_dir = BASE / split / "labels"

    if not img_dir.exists():
        continue

    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img in img_dir.iterdir():
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue

        name_lower = img.stem.lower()

        if any(k in name_lower for k in KEYWORDS_POS):
            skipped += 1
            continue

        lbl = lbl_dir / f"{img.stem}.txt"

        if not lbl.exists():
            lbl.write_text("", encoding="utf-8")
            created += 1

print(f"Labels vazios criados (negativos): {created}")
print(f"Imagens ignoradas (knife/scissors no nome): {skipped}")
