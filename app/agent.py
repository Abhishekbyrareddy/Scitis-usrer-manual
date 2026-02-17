
from app.retriever import MultiFaissRetriever

# All deployment-ready index + metadata pairs
# (These filenames must match exactly what you will place inside /content/indices)
PAIRS = [
    ("cloudplug_edge", "faiss_cloudplug_edge.index", "faiss_cloudplug_edge_metadata.json"),
    ("cloudplug_light", "faiss_cloudplug_light.index", "faiss_cloudplug_light_metadata.json"),
    ("containeranlage_hard", "faiss_containeranlage_hard.index", "faiss_containeranlage_hard_metadata.json"),
    ("armin_kunz", "faiss_armin_kunz_master.index", "faiss_armin_kunz_master_metadata.json"),
    ("portal_scitis", "faiss_portal_manual_scitis.index", "faiss_portal_manual_scitis_metadata.json"),
]

def build_retriever(indices_dir: str = "indices") -> MultiFaissRetriever:
    r = MultiFaissRetriever(indices_dir=indices_dir)
    r.load(PAIRS)
    return r
