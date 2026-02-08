"""
Memoria semantica local basada en FAISS.

Independiente del proveedor LLM. FAISS se usa por coste y control:
- Funciona offline, sin llamadas a APIs externas.
- Almacena el indice y metadata en disco para persistencia.
- Usa OllamaEmbeddings para generar vectores (configurable via factory).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    text: str
    metadata: Dict[str, Any]


class FaissMemory:
    """Memoria semantica local basada en FAISS.

    Guarda y recupera textos por similitud coseno (IndexFlatIP con L2 norm).
    El indice y metadata se persisten en disco para sobrevivir reinicios.
    """

    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embeddings: Any,
        normalize: bool = True,
        dim: Optional[int] = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embeddings = embeddings
        self.normalize = normalize
        self.enabled = _FAISS_AVAILABLE

        self._items: List[MemoryItem] = []
        self._index = None

        if not self.enabled:
            logger.warning(
                "FAISS no esta disponible. Memoria deshabilitada."
            )
            return

        self._dim = dim or self._resolve_dim()
        self._index = self._load_or_create_index(self._dim)
        self._items = self._load_metadata()

    def save(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Guarda un texto en la memoria semantica."""
        if not self.enabled or not text:
            return

        meta = metadata or {}
        vector = self._embed(text)
        if vector is None:
            return

        self._index.add(vector)  # type: ignore[union-attr]
        self._items.append(MemoryItem(text=text, metadata=meta))
        self._persist()

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Busca textos similares al query y devuelve resultados ordenados."""
        if not self.enabled or not query or self._index is None:
            return []

        if len(self._items) == 0:
            return []

        qvec = self._embed(query)
        if qvec is None:
            return []

        scores, idxs = self._index.search(qvec, min(k, len(self._items)))
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self._items):
                continue
            item = self._items[idx]
            results.append(
                {
                    "text": item.text,
                    "metadata": item.metadata,
                    "score": float(score),
                }
            )
        return results

    # -- Internal helpers --------------------------------------------------

    def _resolve_dim(self) -> int:
        sample = self.embeddings.embed_documents(["dimension_probe"])[0]
        return len(sample)

    def _embed(self, text: str) -> Optional[np.ndarray]:
        try:
            vec = self.embeddings.embed_documents([text])[0]
            arr = np.array([vec], dtype="float32")
            if self.normalize:
                faiss.normalize_L2(arr)  # type: ignore[attr-defined]
            return arr
        except Exception as exc:
            logger.error(
                "Fallo al generar embeddings para memoria: %s", exc
            )
            return None

    def _load_or_create_index(self, dim: int):
        try:
            if self.index_path.exists():
                return faiss.read_index(str(self.index_path))  # type: ignore[attr-defined]
            return faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.error("Fallo al cargar/crear indice FAISS: %s", exc)
            self.enabled = False
            return None

    def _load_metadata(self) -> List[MemoryItem]:
        if not self.metadata_path.exists():
            return []
        try:
            data = json.loads(
                self.metadata_path.read_text(encoding="utf-8")
            )
            return [MemoryItem(**item) for item in data]
        except Exception as exc:
            logger.error("Fallo al cargar metadata FAISS: %s", exc)
            return []

    def _persist(self) -> None:
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(
                self._index, str(self.index_path)
            )  # type: ignore[arg-type]
            payload = [item.__dict__ for item in self._items]
            self.metadata_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.error("Fallo al persistir FAISS: %s", exc)
