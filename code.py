import math
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from docx import Document
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".xml",
    ".html",
    ".htm",
    ".py",
    ".js",
    ".ts",
    ".java",
    ".c",
    ".cpp",
    ".pdf",
    ".docx",
}


class ClusterRequest(BaseModel):
    source_path: str = Field(..., description="Folder path containing documents")
    destination_path: Optional[str] = Field(
        default=None, description="Output folder for clustered files"
    )
    mode: str = Field(default="copy", description="copy or move")
    recursive: bool = Field(default=True, description="Scan subfolders")
    num_clusters: Optional[int] = Field(
        default=None, ge=2, description="Optional fixed number of clusters"
    )
    include_extensions: Optional[list[str]] = Field(
        default=None, description="Optional extensions to include, e.g. ['.pdf','.txt']"
    )


def normalize_extensions(values: Optional[list[str]]) -> set[str]:
    if not values:
        return SUPPORTED_EXTENSIONS
    normalized = set()
    for ext in values:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.add(ext)
    return normalized


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    chunks = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks).strip()


def read_docx_file(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs).strip()


def extract_text(path: Path) -> str:
    try:
        if path.suffix.lower() == ".pdf":
            return read_pdf_file(path)
        if path.suffix.lower() == ".docx":
            return read_docx_file(path)
        return read_text_file(path)
    except Exception:
        return ""


def collect_documents(source_path: Path, recursive: bool, extensions: set[str]) -> list[Path]:
    walker = source_path.rglob("*") if recursive else source_path.glob("*")
    files = [p for p in walker if p.is_file() and p.suffix.lower() in extensions]
    return sorted(files)


def infer_cluster_count(doc_count: int) -> int:
    return max(2, min(12, int(round(math.sqrt(doc_count)))))


def generate_topic_names(vectorizer: TfidfVectorizer, model: KMeans) -> dict[int, str]:
    terms = vectorizer.get_feature_names_out()
    names: dict[int, str] = {}
    for idx, centroid in enumerate(model.cluster_centers_):
        top_indices = np.argsort(centroid)[-3:][::-1]
        words = [terms[i] for i in top_indices if centroid[i] > 0]
        topic = "_".join(words) if words else f"topic_{idx + 1}"
        names[idx] = topic
    return names


def sanitize_folder_name(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9_\- ]+", "", value)
    value = value.replace(" ", "_")
    value = re.sub(r"_+", "_", value)
    return value[:60] or "misc"


def cluster_documents(doc_texts: list[str], requested_clusters: Optional[int]) -> tuple[np.ndarray, dict[int, str]]:
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_features=10000)
    matrix = vectorizer.fit_transform(doc_texts)
    n_docs = len(doc_texts)
    n_clusters = requested_clusters or infer_cluster_count(n_docs)
    n_clusters = max(2, min(n_clusters, n_docs))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = model.fit_predict(matrix)
    topics = generate_topic_names(vectorizer, model)
    return labels, topics


def resolve_unique_target_path(base_target: Path) -> Path:
    if not base_target.exists():
        return base_target
    stem = base_target.stem
    suffix = base_target.suffix
    parent = base_target.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def organize_files(
    files: list[Path],
    labels: np.ndarray,
    topics: dict[int, str],
    destination_root: Path,
    mode: str,
) -> list[dict]:
    destination_root.mkdir(parents=True, exist_ok=True)
    result = []
    for file_path, label in zip(files, labels):
        topic = sanitize_folder_name(topics[int(label)])
        cluster_dir = destination_root / f"{int(label) + 1:02d}_{topic}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        target_path = resolve_unique_target_path(cluster_dir / file_path.name)
        if mode == "move":
            shutil.move(str(file_path), str(target_path))
        else:
            shutil.copy2(file_path, target_path)
        result.append(
            {
                "source": str(file_path),
                "target": str(target_path),
                "cluster_id": int(label) + 1,
                "cluster_topic": topic,
            }
        )
    return result


app = FastAPI(title="Document Clustering System", version="1.0.0")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def home() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_file)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


@app.post("/api/cluster")
def cluster_endpoint(payload: ClusterRequest) -> dict:
    mode = payload.mode.lower().strip()
    if mode not in {"copy", "move"}:
        raise HTTPException(status_code=400, detail="mode must be 'copy' or 'move'")

    source_path = Path(payload.source_path).expanduser().resolve()
    if not source_path.exists() or not source_path.is_dir():
        raise HTTPException(status_code=400, detail="source_path must be an existing folder")

    if payload.destination_path:
        destination_root = Path(payload.destination_path).expanduser().resolve()
    else:
        destination_root = source_path.parent / f"{source_path.name}_clustered"

    extensions = normalize_extensions(payload.include_extensions)
    files = collect_documents(source_path, payload.recursive, extensions)
    if not files:
        raise HTTPException(
            status_code=400, detail="No supported files found for the selected folder"
        )

    docs = []
    used_files = []
    skipped = []
    for file_path in files:
        text = extract_text(file_path)
        if len(text) >= 30:
            docs.append(text)
            used_files.append(file_path)
        else:
            skipped.append(str(file_path))

    if len(docs) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 readable files with meaningful text (>= 30 chars)",
        )

    labels, topics = cluster_documents(docs, payload.num_clusters)
    moved = organize_files(used_files, labels, topics, destination_root, mode)
    cluster_counts = Counter(item["cluster_id"] for item in moved)

    return {
        "message": "Clustering completed",
        "source_path": str(source_path),
        "destination_path": str(destination_root),
        "mode": mode,
        "processed_file_count": len(used_files),
        "skipped_file_count": len(skipped),
        "skipped_files": skipped,
        "clusters": [
            {
                "cluster_id": cluster_id,
                "file_count": cluster_counts[cluster_id],
            }
            for cluster_id in sorted(cluster_counts)
        ],
        "files": moved,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("code:app", host="0.0.0.0", port=8000, reload=True)
