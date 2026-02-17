import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from app.agent import build_retriever

APP_START_TS = time.time()
app = FastAPI(title="SCITIS RAG Chatbot", version="0.1")

RETRIEVER = None


# -------------------- QUERY UNDERSTANDING LAYER --------------------
def _expand_query_for_scitis(q: str) -> str:
    """
    Translates user language into documentation language.
    This fixes the 'deploy scitis -> introduction page' problem.
    """
    ql = q.lower()

    if "deploy" in ql or "deployment" in ql:
        return q + " installation setup configuration device connection gateway commissioning"

    if "connect device" in ql or "add device" in ql:
        return q + " gateway setup cloudplug configuration edge connection"

    if "login" in ql or "sign in" in ql:
        return q + " user account authentication portal access"

    if "error" in ql or "not working" in ql or "failed" in ql:
        return q + " troubleshooting diagnostics problem solution steps"

    if "cloudplug" in ql:
        return q + " cloudplug edge installation wiring mounting power supply configuration"

    return q


# -------------------- FILTER LOW-VALUE PAGES --------------------
def _filter_non_content_hits(hits):
    filtered = []
    for h in hits:
        m = h.meta or {}
        txt = (m.get("text") or m.get("embedding_text") or "").lower()

        # company imprint page
        if "scitis.io gmbh" in txt and "stuttgart" in txt:
            continue

        # cover / title page
        if "benutzerhandbuch" in txt and len(txt) < 80:
            continue

        # generic introduction page
        if "overview of the technical aspects" in txt:
            continue

        filtered.append(h)

    return filtered if filtered else hits


# -------------------- DATA MODELS --------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    top_k_per_corpus: int = 5
    final_top_k: int = 10
    language: Optional[str] = None
    session_id: Optional[str] = None
    scitis_version: Optional[str] = None


class Citation(BaseModel):
    source_label: str
    chunk_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer_text: str
    citations: List[Citation]
    image_refs: List[str]
    escalate: bool
    debug: Optional[Dict[str, Any]] = None


# -------------------- STARTUP --------------------
@app.on_event("startup")
def startup_event():
    global RETRIEVER
    indices_dir = os.environ.get("INDICES_DIR", "indices")
    RETRIEVER = build_retriever(indices_dir=indices_dir)


@app.get("/healthz")
def healthz():
    return {"ok": True, "uptime_s": int(time.time() - APP_START_TS)}


# -------------------- CHAT ENDPOINT --------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):

    expanded_query = _expand_query_for_scitis(req.message)

    hits = RETRIEVER.search(
        query=expanded_query,
        top_k_per_corpus=req.top_k_per_corpus,
        final_top_k=req.final_top_k,
    )

    # relevance threshold (L2 distance: lower is better)
    MAX_DISTANCE = 0.65
    strong_hits = [h for h in hits if h.score <= MAX_DISTANCE]
    if strong_hits:
        hits = strong_hits

    # remove cover/imprint/intro pages
    hits = _filter_non_content_hits(hits)

    if not hits:
        return ChatResponse(
            answer_text="I couldn’t find this in the indexed scitis documentation. Please contact a support engineer.",
            citations=[],
            image_refs=[],
            escalate=True,
        )

    citations: List[Citation] = []
    image_refs: List[str] = []

    for h in hits:
        m = h.meta or {}
        citations.append(Citation(source_label=h.source_label, chunk_id=m.get("chunk_id")))

        if m.get("image_file"):
            image_refs.append(str(m["image_file"]))
        if m.get("image_refs"):
            try:
                image_refs.extend([str(x) for x in m["image_refs"]])
            except Exception:
                pass

    # choose first hit with real text
    top = hits[0]
    snippet = ""

    for candidate in hits:
        meta = candidate.meta or {}
        t = meta.get("text") or meta.get("embedding_text") or ""
        t = " ".join(str(t).split())
        if t:
            top = candidate
            snippet = t
            break

    answer = f"Top match from: {top.source_label}\n\n{snippet[:900]}"

    # deduplicate images
    seen = set()
    image_refs_unique = []
    for x in image_refs:
        if x not in seen:
            seen.add(x)
            image_refs_unique.append(x)

    return ChatResponse(
        answer_text=answer,
        citations=citations,
        image_refs=image_refs_unique,
        escalate=False,
        debug={"top_score": top.score, "hits_returned": len(hits)},
    )


# -------------------- DEBUG ENDPOINT --------------------
@app.post("/debug_first_hit")
async def debug_first_hit(req: ChatRequest):

    expanded_query = _expand_query_for_scitis(req.message)

    hits = RETRIEVER.search(
        query=expanded_query,
        top_k_per_corpus=req.top_k_per_corpus,
        final_top_k=req.final_top_k,
    )

    MAX_DISTANCE = 0.65
    strong_hits = [h for h in hits if h.score <= MAX_DISTANCE]
    if strong_hits:
        hits = strong_hits

    hits = _filter_non_content_hits(hits)

    if not hits:
        return {"hits": 0}

    h = hits[0]
    m = h.meta or {}

    return {
        "score": h.score,
        "source_label": h.source_label,
        "chunk_id": m.get("chunk_id"),
        "folder_name": m.get("folder_name"),
        "pdf_name": m.get("pdf_name"),
        "page_number": m.get("page_number"),
        "has_text": bool((m.get("text") or "").strip()),
        "text_preview": (m.get("text") or m.get("embedding_text") or "")[:200],
        "image_refs": m.get("image_refs"),
        "image_file": m.get("image_file"),
    }