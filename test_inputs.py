from pathlib import Path
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
INPUTS = BASE / "inputs"
MODEL_PATH = BASE / "runs" / "yolov8n_scissors_knife_img960" / "weights" / "best.pt"

def main():
    model = YOLO(str(MODEL_PATH))

    video_files = sorted([p for p in INPUTS.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}])

    if not video_files:
        raise FileNotFoundError(f"Nenhum vídeo encontrado em: {INPUTS}")

    for video in video_files:
        print(f"\n=== Processando: {video.name} ===")

        results = model.predict(
            source=str(video),
            imgsz=960,
            conf=0.25,
            device=0,          # RTX 4060
            save=True,         # salva vídeo com boxes
            project=str(BASE / "runs"),
            name="predict_inputs",
            exist_ok=True,
            verbose=False,
        )

        # Log simples: quantas detecções por classe no vídeo (soma dos frames)
        counts = {0: 0, 1: 0}  # 0=knife, 1=scissors (ajuste se seu data.yaml for diferente)
        for r in results:
            if r.boxes is None:
                continue
            cls = r.boxes.cls
            if cls is None:
                continue
            for c in cls.tolist():
                c = int(c)
                if c in counts:
                    counts[c] += 1

        print(f"Detecções totais (frames somados): knife={counts[0]} | scissors={counts[1]}")
        print(f"Saída em: {BASE / 'runs' / 'predict_inputs'}")

if __name__ == "__main__":
    main()
