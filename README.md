# Industrial Defect API

Built in 2026 as a portfolio project. Demonstrates skills used in my logistics / automotive AI roles.

Theme: industrial vision / quality control — upload a panel image, get OK/NG + score + bounding boxes.

## Stack

- Python 3.11+, FastAPI, OpenCV (`opencv-python-headless`)
- Lightweight classical CV (adaptive threshold + contours) — no heavy DL required

## Metrics note

On the bundled **synthetic** OK/NG PNGs, the contour pipeline typically labels OK panels as `OK` (no boxes) and NG panels as `NG` with one or more boxes. This is a demo heuristic, not a production AOI model. Re-run `scripts/generate_samples.py` and spot-check with the example `curl` below.

## Quick start

```bash
cd 03-industrial-defect-api
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_samples.py
uvicorn app.main:app --reload --port 8003
```

Example:

```bash
curl -X POST http://127.0.0.1:8003/inspect -F "file=@data/samples/ng/ng_01.png"
```

## License

MIT — Abdalla Abdelsamie (2026).
