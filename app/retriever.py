import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Hit:
    score: float
    source_label: str
    text_preview: str
    meta: Dict[str, Any]


class MultiFaissRetriever:
    def __init__(self, indices_dir: str, model_name: str = "all-MiniLM-L6-v2"):
        self.indices_dir = Path(indices_dir)
        self.model = SentenceTransformer(model_name)
        self.corpora: List[Tuple[str, faiss.Index, List[Dict[str, Any]]]] = []

    def load(self, pairs: List[Tuple[str, str, str]]) -> None:
        for corpus_name, idx_file, meta_file in pairs:
            idx_path = self.indices_dir / idx_file
            meta_path = self.indices_dir / meta_file

            index = faiss.read_index(str(idx_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if index.ntotal != len(meta):
                raise ValueError(
                    f"Mismatch for {corpus_name}: "
                    f"FAISS vectors={index.ntotal} "
                    f"metadata records={len(meta)}"
                )

            self.corpora.append((corpus_name, index, meta))

    def _format_source(self, m: Dict[str, Any]) -> str:
        # Prefer PDF-style citations when available
        pdf = m.get("pdf_name")
        page = m.get("page_number")
        folder = m.get("folder_name")

        if pdf is not None or page is not None:
            folder = folder or "UNKNOWN_FOLDER"
            pdf = pdf or "UNKNOWN_PDF"
            page = page if page is not None else "UNKNOWN_PAGE"
            source = f"{folder} / {pdf} | Page: {page}"

            # Add optional device/version info if present
            device = m.get("device")
            version = m.get("version")
            if device:
                source += f" | device={device}"
            if version:
                source += f" | version={version}"

            # Add image hints if present
            if m.get("image_file"):
                source += f" | Image: {m['image_file']}"
            elif m.get("image_refs"):
                source += f" | ImageRefs: {m['image_refs']}"

            return source

        # Prefer URL-style citations when available
        url = m.get("source_url")
        if url:
            folder = folder or "UNKNOWN_SITE"
            lang = m.get("language", "")
            title = m.get("page_title", "")

            parts = [folder, url]
            if lang:
                parts.append(f"lang={lang}")
            if title:
                parts.append(f"title={title}")

            # Image hints
            if m.get("image_file"):
                parts.append(f"Image={m['image_file']}")
            elif m.get("image_refs"):
                parts.append(f"ImageRefs={m['image_refs']}")

            return " | ".join(parts)

        # Fallback: still produce a useful label (never UNKNOWN_SOURCE)
        device = m.get("device") or "unknown_device"
        version = m.get("version") or "unknown_version"
        chunk_id = m.get("chunk_id") or "unknown_chunk"
        chunk_type = m.get("chunk_type") or "unknown_type"
        return f"SCITIS_DOC | device={device} | version={version} | type={chunk_type} | chunk={chunk_id}"

    def search(self, query: str, top_k_per_corpus: int = 5, final_top_k: int = 10):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = np.asarray(q_emb, dtype="float32")

        hits: List[Hit] = []

        for corpus_name, index, meta in self.corpora:
            D, I = index.search(q_emb, top_k_per_corpus)

            for dist, idx in zip(D[0], I[0]):
                m = meta[int(idx)]
                text = m.get("text") or m.get("embedding_text") or ""
                preview = " ".join(str(text).split())[:220]

                hits.append(
                    Hit(
                        score=float(dist),
                        source_label=self._format_source(m),
                        text_preview=preview,
                        meta=m,
                    )
                )

        hits.sort(key=lambda h: h.score)
        return hits[:final_top_k]