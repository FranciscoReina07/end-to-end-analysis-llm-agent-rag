"""
Pipeline de embeddings con cache y entrenamiento del modelo ML.

Responsabilidades:
- Generar embeddings via Ollama con cache en Parquet (hash SHA-256).
- Fragmentar textos largos en chunks.
- Entrenar, evaluar y persistir el modelo de clasificacion.
"""

import hashlib
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from app.config import (
    CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDINGS_CACHE_PATH,
    MAX_CHARS,
    MODEL_DIR,
    DataValidationError,
    EmbeddingError,
    ModelTrainingError,
    logger,
)

# ---------------------------------------------------------------------------
# Text utils
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_chunks_for_texts(
    texts: List[str],
) -> Tuple[List[str], List[List[str]]]:
    all_chunks: List[str] = []
    text_to_hashes: List[List[str]] = []
    for text in texts:
        chunks = chunk_text(text)
        hashes = [_hash_text(c) for c in chunks]
        all_chunks.extend(chunks)
        text_to_hashes.append(hashes)
    return all_chunks, text_to_hashes


# ---------------------------------------------------------------------------
# Cache de embeddings
# ---------------------------------------------------------------------------


def _load_cache(path: Path = EMBEDDINGS_CACHE_PATH) -> Dict[str, List[float]]:
    if not path.exists():
        return {}
    cache_df = pd.read_parquet(path)
    return dict(zip(cache_df["text_hash"], cache_df["embedding"]))


def _save_cache(
    cache: Dict[str, List[float]], path: Path = EMBEDDINGS_CACHE_PATH
) -> None:
    cache_df = pd.DataFrame(
        {"text_hash": list(cache.keys()), "embedding": list(cache.values())}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_parquet(path, index=False)
    logger.info("Cache de embeddings actualizado: %s", path)


# ---------------------------------------------------------------------------
# Generacion de embeddings
# ---------------------------------------------------------------------------


def _batch_iter(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def generate_embeddings_with_cache(
    texts: List[str],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> List[List[float]]:
    """Genera embeddings usando Ollama con cache en disco."""
    if not texts:
        raise EmbeddingError("No hay textos para vectorizar.")

    embedder = OllamaEmbeddings(model=model_name)
    cache = _load_cache()

    all_chunks, text_to_hashes = _build_chunks_for_texts(texts)

    unique_chunks: Dict[str, str] = {}
    for chunk in all_chunks:
        h = _hash_text(chunk)
        if h not in unique_chunks and h not in cache:
            unique_chunks[h] = chunk

    missing_hashes = list(unique_chunks.keys())
    missing_chunks = [unique_chunks[h] for h in missing_hashes]

    if missing_chunks:
        for batch in tqdm(
            _batch_iter(missing_chunks, batch_size),
            desc="Vectorizando chunks",
        ):
            batch_embs = embedder.embed_documents(batch)
            for text, emb in zip(batch, batch_embs):
                cache[_hash_text(text)] = emb
        _save_cache(cache)
    else:
        logger.info("Todos los chunks ya estaban en cache.")

    embeddings: List[List[float]] = []
    for hashes in text_to_hashes:
        chunk_embs = [cache[h] for h in hashes]
        embeddings.append(np.mean(chunk_embs, axis=0).tolist())

    return embeddings


# ---------------------------------------------------------------------------
# Preparacion de datos para ML
# ---------------------------------------------------------------------------


def preparar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara el dataset para entrenamiento: filtra, limpia y crea target."""
    required = {"full_conversation", "escalate_conversation"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Faltan columnas: {missing}")

    df_model = df.copy()
    df_model["full_conversation"] = (
        df_model["full_conversation"].fillna("").str.strip().str.slice(0, MAX_CHARS)
    )
    df_model = df_model[df_model["full_conversation"].str.len() > 0]
    df_model["target"] = df_model["escalate_conversation"].fillna(0).astype(int)
    return df_model


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------


def train_model(
    X_train: List[List[float]], y_train: pd.Series
) -> LogisticRegression:
    if len(np.unique(y_train)) < 2:
        raise ModelTrainingError("Target con una sola clase.")
    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


def save_model(model: LogisticRegression, version: str = "v1") -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"model_{version}.joblib"
    joblib.dump(model, path)
    logger.info("Modelo guardado: %s", path)
    return path


def load_model(version: str = "v1") -> LogisticRegression:
    path = MODEL_DIR / f"model_{version}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {path}")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------


def evaluate_model(
    model: LogisticRegression,
    X_test: List[List[float]],
    y_test: pd.Series,
) -> Dict[str, float]:
    """Evalua el modelo e imprime reporte completo. Devuelve metricas clave."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nREPORTE DE CLASIFICACION")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("MATRIZ DE CONFUSION")
    print(cm)
    print(f"Falsos Negativos: {cm[1][0]}")

    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {pr:.4f}")

    fn_cost, fp_cost = 5, 1
    total_cost = cm[1][0] * fn_cost + cm[0][1] * fp_cost
    print(f"Costo Esperado: {total_cost}")

    return {"roc_auc": roc, "pr_auc": pr, "cost": total_cost}


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------


def entrenar_clasificador_embeddings(
    df: pd.DataFrame,
    model_name: str = EMBEDDING_MODEL,
) -> Tuple[LogisticRegression, List[List[float]], pd.Series]:
    """Pipeline completo: prepara datos, genera embeddings, entrena y evalua."""
    df_model = preparar_dataset(df)
    texts = df_model["full_conversation"].tolist()

    embeddings = generate_embeddings_with_cache(texts, model_name=model_name)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        df_model["target"],
        test_size=0.2,
        random_state=42,
        stratify=df_model["target"],
    )

    clf = train_model(X_train, y_train)
    save_model(clf)
    evaluate_model(clf, X_test, y_test)

    return clf, X_test, y_test
