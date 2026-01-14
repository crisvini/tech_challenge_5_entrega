from pathlib import Path
from ultralytics import YOLO


def main():
    BASE = Path(__file__).resolve().parent
    DATA = BASE / "data.yaml"

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(DATA),
        imgsz=960,
        epochs=120,
        close_mosaic=20,
        patience=50,
        device="cpu",  # para processadores
        # device=0, # para placas de vídeo NVIDIA
        batch=8,
        workers=0,
        cache=True,
        amp=True,
        project=str(BASE / "runs"),
        name="yolov8n_scissors_knife_img960",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    print("Treino finalizado!")
    print("Melhor peso:", results.save_dir / "weights" / "best.pt")


if __name__ == "__main__":
    main()
