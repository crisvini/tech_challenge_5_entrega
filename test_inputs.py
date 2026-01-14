from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime

import cv2

BASE = Path(__file__).resolve().parent
INPUTS = BASE / "inputs"
MODEL_PATH = BASE / "runs" / "yolov8n_scissors_knife_img960" / "weights" / "best.pt"

ALERTS_DIR = BASE / "alerts"
ALERTS_DIR.mkdir(exist_ok=True)

CLASS_MAP = {
    0: "knife",
    1: "scissors",
}

ALERT_MIN_TOTAL_DETECTIONS = 3
TIME_ROUND_DECIMALS = 1


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return fps if fps > 0 else 30.0


def seconds_to_ranges(seconds_list: list[float], gap: float = 0.6) -> list[str]:
    if not seconds_list:
        return []

    s = sorted(set(seconds_list))
    ranges = []
    start = s[0]
    prev = s[0]

    for cur in s[1:]:
        if cur - prev <= gap:
            prev = cur
            continue
        if start == prev:
            ranges.append(f"{start:.{TIME_ROUND_DECIMALS}f}s")
        else:
            ranges.append(
                f"{start:.{TIME_ROUND_DECIMALS}f}–{prev:.{TIME_ROUND_DECIMALS}f}s"
            )
        start = prev = cur

    if start == prev:
        ranges.append(f"{start:.{TIME_ROUND_DECIMALS}f}s")
    else:
        ranges.append(
            f"{start:.{TIME_ROUND_DECIMALS}f}–{prev:.{TIME_ROUND_DECIMALS}f}s"
        )

    return ranges


def build_alert_message(
    video_name: str, counts: dict[int, int], times_by_class: dict[int, list[float]]
) -> str:
    knife_count = counts.get(0, 0)
    scissors_count = counts.get(1, 0)

    knife_ranges = seconds_to_ranges(times_by_class.get(0, []))
    scissors_ranges = seconds_to_ranges(times_by_class.get(1, []))

    parts = []
    parts.append(
        f"ALERTA: possível presença de objeto cortante detectada no vídeo '{video_name}'."
    )
    parts.append(
        f"Resumo: knife={knife_count} | scissors={scissors_count} (detecções por frame somadas)."
    )

    if knife_ranges:
        parts.append(f"Ocorrências de FACA (segundos): {', '.join(knife_ranges)}.")
    if scissors_ranges:
        parts.append(
            f"Ocorrências de TESOURA (segundos): {', '.join(scissors_ranges)}."
        )

    if knife_count > 0:
        parts.append(
            "Ação recomendada: acionar equipe de segurança e verificar imagens imediatamente."
        )
    else:
        parts.append(
            "Ação recomendada: monitorar e validar a cena (objeto pode ser similar)."
        )

    return " ".join(parts)


def emitir_alerta(
    video_name: str, counts: dict[int, int], times_by_class: dict[int, list[float]]
) -> Path:
    severity = "HIGH" if counts.get(0, 0) > 0 else "MEDIUM"

    message = build_alert_message(video_name, counts, times_by_class)

    alert = {
        "timestamp": datetime.now().isoformat(),
        "video": video_name,
        "severity": severity,
        "detections_total": {
            "knife": counts.get(0, 0),
            "scissors": counts.get(1, 0),
        },
        "detections_seconds": {
            "knife": seconds_to_ranges(times_by_class.get(0, [])),
            "scissors": seconds_to_ranges(times_by_class.get(1, [])),
        },
        "message": message,
    }

    alert_file = (
        ALERTS_DIR / f"alert_{video_name}_{int(datetime.now().timestamp())}.json"
    )
    with open(alert_file, "w", encoding="utf-8") as f:
        json.dump(alert, f, indent=2, ensure_ascii=False)

    print("\n🚨 ALERTA DE SEGURANÇA 🚨")
    print(message)
    print(f"📁 Alerta salvo em: {alert_file}")

    return alert_file


def main():
    model = YOLO(str(MODEL_PATH))

    video_files = sorted(
        [
            p
            for p in INPUTS.iterdir()
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        ]
    )
    if not video_files:
        raise FileNotFoundError(f"Nenhum vídeo encontrado em: {INPUTS}")

    for video in video_files:
        print(f"\n=== Processando: {video.name} ===")

        fps = get_video_fps(video)

        results = model.predict(
            source=str(video),
            imgsz=960,
            conf=0.25,
            device="cpu",  # para processadores
            # device=0, # para placas de vídeo NVIDIA
            save=True,
            project=str(BASE / "runs"),
            name="predict_inputs",
            exist_ok=True,
            verbose=False,
        )

        counts = {0: 0, 1: 0}
        times_by_class = {0: [], 1: []}

        for frame_idx, r in enumerate(results):
            if r.boxes is None:
                continue

            cls = r.boxes.cls
            if cls is None or len(cls) == 0:
                continue

            t = round(frame_idx / fps, TIME_ROUND_DECIMALS)

            for c in cls.tolist():
                c = int(c)
                if c in counts:
                    counts[c] += 1
                    times_by_class[c].append(t)

        print(
            f"Detecções totais (frames somados): knife={counts[0]} | scissors={counts[1]}"
        )
        print(f"Saída do vídeo com boxes em: {BASE / 'runs' / 'predict_inputs'}")

        total_cortantes = counts[0] + counts[1]
        if total_cortantes >= ALERT_MIN_TOTAL_DETECTIONS:
            emitir_alerta(video.name, counts, times_by_class)
        else:
            print("✅ Nenhum alerta necessário para este vídeo.")


if __name__ == "__main__":
    main()
