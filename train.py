from pathlib import Path
from ultralytics import YOLO

def main():
    BASE = Path(__file__).resolve().parent
    DATA = BASE / "data.yaml"   # seu yaml na raiz do projeto/dataset

    # Modelo base (transfer learning)
    model = YOLO("yolov8n.pt")  # leve e rápido (recomendado pra 8GB)

    results = model.train(
        data=str(DATA),

        # --- Qualidade p/ objetos pequenos ---
        imgsz=960,          # bom p/ faca/tesoura pequena
        epochs=120,
        close_mosaic=20,    # estabiliza no final
        patience=50,

        # --- GPU / performance no Windows ---
        device=0,           # usa a RTX (CUDA)
        batch=8,            # se der OOM, diminui pra 4
        workers=0,          # evita problemas no Windows
        cache=True,         # ajuda performance (se RAM/SSD aguentar)
        amp=True,           # mixed precision (normalmente melhora e economiza VRAM)

        # --- Projeto/saída ---
        project=str(BASE / "runs"),
        name="yolov8n_scissors_knife_img960",
        exist_ok=True,

        # --- Detalhes úteis ---
        plots=True,
        verbose=True,
    )

    print("Treino finalizado!")
    print("Melhor peso:", results.save_dir / "weights" / "best.pt")

if __name__ == "__main__":
    main()
