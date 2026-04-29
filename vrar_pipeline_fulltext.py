# -*- coding: utf-8 -*-
"""
VR/AR Patent Topic Modeling – FULL-TEXT, GPU-Optimized
=============================================================================

Dieses Skript implementiert die vollständige, final optimierte Pipeline:
- Full-Text Verarbeitung (title + abstract + description + claims)
- PatentSBERTa Embeddings (GPU)
- BERTopic datengetrieben (UMAP + HDBSCAN)
- Optionales Topic-Merging via Centroid-Cosine
- KPI-Berechnung & Bucketing ("STRONG", "WEAK", "LATENT", "NSWK")
- Community-Graphen, Heatmaps
- Executive Summary Export (CSV + XLSX)
- Histogramme
- Erweiterung: Top-Word-Sets für spätere LLM-Topic-Benennung
- KEINE LLM-Label-Integration im Skript!! Wird separat durch LLM-Anbindung gemacht

"""

# =========================================================
# IMPORTS
# =========================================================

import os
import re
import json
import warnings
import random
import inspect
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text as sk_text
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from bertopic import BERTopic

import matplotlib.pyplot as plt
import networkx as nx

# Optional OLS Trend-Modell
try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:
    _HAS_SM = False

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# REPRODUZIERBARKEIT
# =========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# =========================================================
# GPU SETUP
# =========================================================

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

print(f"[DEVICE] Verwende: {DEVICE}")

# Embedding Batch Size
EMB_BATCH = int(os.getenv("EMB_BATCH", 256 if DEVICE == "cuda" else 32))

# =========================================================
# INPUT / OUTPUT SETTINGS
# =========================================================

DATA_XLSX = "Patente 10k - komplett.xlsx"
DATE_COL = "publication_date"

TEXT_COLS = ["title", "abstract", "description", "claims"]

OUT_DIR = "."
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
COMMUNITY_DIR = os.path.join(PLOTS_DIR, "community_graphs")
HEATMAP_DIR = os.path.join(PLOTS_DIR, "heatmaps")
TOPK_DIR = os.path.join(OUT_DIR, "topk_neighbors")
AUDIT_JSON = os.path.join(OUT_DIR, "audit_trail.json")

for d in [PLOTS_DIR, COMMUNITY_DIR, HEATMAP_DIR, TOPK_DIR]:
    os.makedirs(d, exist_ok=True)

# =========================================================
# CONFIG: TOPIC MERGE + COMMUNITY GRAPH
# =========================================================

CENTROID_MERGE_COS_THR = 0.90    # 0.90 = stärkeres Merging
EDGE_PCTL = 0.70                 # 70%-Quantil für Community Graph Kanten

# =========================================================
# BUCKETING
# =========================================================

BUCKET_CFG = {
    "strong_prop_quantile": 0.66,
    "latent_prop_quantile": 0.20,
    "stability_min_strong": 0.60,
    "weak_recent_cagr": 20.0,
    "weak_tstat_min": 2.0,
    "recent_years": 3
}


# =========================================================
# DATUM PARSEN
# =========================================================

def parse_year(v):
    if pd.isna(v):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return int(v.year)
    if isinstance(v, (int, float)):
        try:
            return int(pd.to_datetime(v, origin="1899-12-30", unit="D").year)
        except Exception:
            if 1900 <= v <= 2100:
                return int(v)
    s = str(v).strip()
    for fmt in ("%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return int(datetime.strptime(s, fmt).year)
        except Exception:
            pass
    dt = pd.to_datetime(s, errors="ignore")
    return int(dt.year) if not pd.isna(dt) else None


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


# =========================================================
# FULL TEXT BUILDER (title + abstract + description + claims)
# =========================================================

def build_full_text(df: pd.DataFrame) -> List[str]:
    docs = []
    for _, r in df.iterrows():
        parts = []
        for col in TEXT_COLS:
            if col in df.columns:
                parts.append(clean_text(r.get(col, "")))
        full = " ".join(parts).strip()
        docs.append(full if isinstance(full, str) else "")
    return docs


# =========================================================
# DATEN LADEN
# =========================================================

print("[LOAD] Lese Excel …")
df = pd.read_excel(DATA_XLSX, engine="openpyxl")

assert DATE_COL in df.columns, f"Spalte '{DATE_COL}' fehlt!"

df["year"] = df[DATE_COL].apply(parse_year)
df = df.dropna(subset=["year"]).copy()
df["year"] = df["year"].astype(int)
df = df[(df["year"] >= 2000) & (df["year"] <= 2025)]

print(f"[LOAD] Datensätze (2000–2025): {len(df)}")


# =========================================================
# FULL TEXT ERSTELLEN
# =========================================================

print("[TEXT] Erzeuge FULL TEXT …")
df["full_text"] = build_full_text(df)


# =========================================================
# EMBEDDINGS (PatentSBERTa) MIT CACHE
# =========================================================

EMB_CACHE = "embeddings_fulltext.npy"

def load_or_make_embeddings(texts: List[str]) -> np.ndarray:
    if os.path.isfile(EMB_CACHE):
        try:
            print("[EMB] Lade Embeddings aus Cache …")
            return np.load(EMB_CACHE)
        except Exception:
            pass

    model = SentenceTransformer("AI-Growth-Lab/PatentSBERTa", device=DEVICE)
    emb = model.encode(
        texts,
        batch_size=EMB_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    np.save(EMB_CACHE, emb)
    return emb


print("[EMB] Erstelle oder lade Embeddings …")
embeddings = load_or_make_embeddings(df["full_text"].tolist())


# =========================================================
# STOPWORT-HANDLING
# =========================================================

def load_stopwords() -> list:
    eng = set(sk_text.ENGLISH_STOP_WORDS)

    patent_like = {
        "embodiment","embodiments","apparatus","device","system","method","component",
        "unit","module","portion","plurality","configured","according","comprising",
        "comprise","comprises","further","thereof","therein","therefrom","thereon",
        "example","examples","invention","present","present invention","may","can",
        "said","wherein","whereby","thereby","first","second","third","pluralities",
        "virtual","reality","ar","vr","augmented","user","users","data","information"
    }

    custom = set()
    if os.path.isfile("custom_stopwords.txt"):
        with open("custom_stopwords.txt", "r", encoding="utf-8") as f:
            custom = {ln.strip().lower() for ln in f if ln.strip()}

    return sorted(eng | patent_like | custom)

STOPWORDS = load_stopwords()

print(f"[STOPWORDS] Gesamtmenge: {len(STOPWORDS)} Wörter")


# =========================================================
# BERTopic – datengetriebenes Topic-Modell (kein fixes k)
# =========================================================

def make_topic_model() -> BERTopic:
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=SEED
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=50,
        min_samples=10,
        metric="euclidean",
        prediction_data=True,
        cluster_selection_method="eom"
    )

    vectorizer_model = CountVectorizer(
        stop_words=STOPWORDS,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.90
    )

    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True
    )


print("[TOPIC] Trainiere BERTopic-Modell …")

docs = df["full_text"].astype(str).tolist()

topic_model = make_topic_model()
topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
topics = np.array(topics, dtype=int)


# =========================================================
# TOPIC MERGING – sehr ähnliche Topics zusammenfassen
# =========================================================

def _topic_centroids(emb: np.ndarray, topic_ids: np.ndarray) -> Dict[int, np.ndarray]:
    res = {}
    for t in sorted(set(topic_ids)):
        if t == -1:
            continue
        idx = np.where(topic_ids == t)[0]
        if len(idx) == 0:
            continue
        res[t] = emb[idx].mean(axis=0)
    return res


def merge_similar_topics(model: BERTopic,
                         docs: List[str],
                         emb: np.ndarray,
                         topics_arr: np.ndarray,
                         thr: float = 0.0):

    if thr <= 0:
        return model, topics_arr

    changed = True
    while changed:
        changed = False

        cents = _topic_centroids(emb, topics_arr)
        ids = list(cents.keys())

        if len(ids) < 2:
            break

        M = np.vstack([cents[i] for i in ids])
        S = cosine_similarity(M, M)  # centroid similarities
        np.fill_diagonal(S, -1.0)

        i, j = np.unravel_index(np.argmax(S), S.shape)
        if S[i, j] >= thr:
            t1, t2 = ids[i], ids[j]
            print(f"[MERGE] Fasse Topics {t1} und {t2} zusammen (cos={S[i,j]:.3f})")

            # API-stabile Variante
            try:
                topic_model_new = model.merge_topics(docs, topics_arr, topics_to_merge=[(t1, t2)])
            except:
                try:
                    topic_model_new = model.merge_topics(docs, topics_to_merge=[(t1, t2)])
                except:
                    topic_model_new = model.merge_topics(docs, [t1, t2])

            if topic_model_new is not None:
                model = topic_model_new

            new_topics, _ = model.transform(docs, embeddings=emb)
            topics_arr = np.array(new_topics, dtype=int)

            changed = True

        else:
            break

    return model, topics_arr


print("[TOPIC] Führe Topic-Merging durch …")
topic_model, topics = merge_similar_topics(
    topic_model,
    docs,
    embeddings,
    topics,
    thr=CENTROID_MERGE_COS_THR
)

df["topic"] = topics
df_valid = df[df["topic"] != -1].copy()

n_topics_final = df_valid["topic"].nunique()
print(f"[TOPIC] Final number of topics: {n_topics_final}")


# =========================================================
# KPI-BERECHNUNG – mean_prop, CAGR, Stability, Cross-Sim
# =========================================================

print("[KPI] Berechne Jahresrepräsentationsvektoren …")

MIN_DOCS_PER_TOPIC_YEAR = 5

def topic_year_vectors(dfv: pd.DataFrame) -> Dict[Tuple[int, int], np.ndarray]:
    """Berechnet für jedes Topic & Jahr das mittlere Embedding."""
    res = {}
    for (t, y), grp in dfv.groupby(["topic", "year"]):
        idx = grp.index.values
        if len(idx) < MIN_DOCS_PER_TOPIC_YEAR:
            continue
        res[(int(t), int(y))] = embeddings[idx].mean(axis=0)
    return res


rep_vecs = topic_year_vectors(df_valid)


# =========================================================
# mean_prop je Topic/Jahr
# =========================================================
def mean_prop_by_topic_year(dfv: pd.DataFrame) -> pd.DataFrame:
    total_per_year = dfv.groupby("year")["full_text"].count().rename("total")
    cnt = dfv.groupby(["topic", "year"])["full_text"].count().rename("count").reset_index()

    cnt = cnt.merge(total_per_year.reset_index(), on="year", how="left")
    cnt["mean_prop"] = cnt["count"] / cnt["total"]

    return cnt


print("[KPI] Berechne mean_prop je Topic/Jahr …")
prop = mean_prop_by_topic_year(df_valid)


# =========================================================
# CAGR (über gesamten Zeitraum)
# =========================================================
def cagr_percent(topic_id: int) -> float:
    s = prop[prop["topic"] == topic_id].sort_values("year")

    if len(s) < 2:
        return 0.0

    start = float(s["mean_prop"].iloc[0])
    end   = float(s["mean_prop"].iloc[-1])
    periods = max(len(s) - 1, 1)

    eps = 1e-6
    return (((end + eps) / max(start, eps)) ** (1.0 / periods) - 1.0) * 100.0


# =========================================================
# recent CAGR (letzte N Jahre)
# =========================================================
def recent_cagr_percent(topic_id: int, n: int = BUCKET_CFG["recent_years"]) -> float:
    s = prop[prop["topic"] == topic_id].sort_values("year")

    if len(s) < 2:
        return 0.0

    tail = s.tail(min(n + 1, len(s)))

    start = float(tail["mean_prop"].iloc[0])
    end   = float(tail["mean_prop"].iloc[-1])
    periods = max(len(tail) - 1, 1)

    eps = 1e-6
    return (((end + eps) / max(start, eps)) ** (1.0 / periods) - 1.0) * 100.0


# =========================================================
# Stability (Ähnlichkeit konsekutiver Jahre)
# =========================================================
def avg_stability(topic_id: int) -> float:
    years_sorted = sorted(set(df_valid["year"]))

    vals = []
    prev = None

    for y in years_sorted:
        v = rep_vecs.get((topic_id, y))
        if prev is not None and v is not None:
            vals.append(float(cosine_similarity(prev.reshape(1, -1),
                                                v.reshape(1, -1))[0, 0]))
        if v is not None:
            prev = v

    return float(np.mean(vals)) if vals else np.nan


# =========================================================
# Cross-Similarity innerhalb eines Jahres
# =========================================================
def cross_similarity_year(topic_id: int, year: int) -> float:
    v = rep_vecs.get((topic_id, year))
    if v is None:
        return np.nan

    sims = []
    for (t2, y2), vec in rep_vecs.items():
        if y2 == year and t2 != topic_id:
            sims.append(float(cosine_similarity(v.reshape(1, -1),
                                                vec.reshape(1, -1))[0, 0]))
    return float(np.mean(sims)) if sims else np.nan


# =========================================================
# last_year_prop – Marktanteil im letzten verfügbaren Jahr
# =========================================================
def last_year_prop(topic_id: int) -> float:
    s = prop[prop["topic"] == topic_id]

    if s.empty:
        return 0.0

    y_max = int(s["year"].max())
    return float(s.loc[s["year"] == y_max, "mean_prop"].iloc[0])


# =========================================================
# OLS Trend – t-Statistik der Steigung (optional)
# =========================================================
def ols_tstat(topic_id: int) -> float:
    s = prop[prop["topic"] == topic_id].sort_values("year")

    if len(s) < 3 or not _HAS_SM:
        return 0.0

    eps = 1e-6
    y = np.log(s["mean_prop"].astype(float).values + eps)
    X = sm.add_constant(s["year"].astype(float).values)

    try:
        model = sm.OLS(y, X).fit()
        return float(model.tvalues[1])
    except Exception:
        return 0.0


# =========================================================
# BUCKETING – Zuweisung zu STRONG / WEAK / LATENT / NSWK
# =========================================================
def label_bucket(topic_id: int, prop_df: pd.DataFrame) -> str:

    s = prop_df[prop_df["topic"] == topic_id]
    if s.empty:
        return "LATENT"

    avg_share = float(s["mean_prop"].mean())
    cagr_all  = float(cagr_percent(topic_id))
    stab      = float(avg_stability(topic_id))
    last_share = float(last_year_prop(topic_id))
    rec_g     = float(recent_cagr_percent(topic_id, BUCKET_CFG["recent_years"]))
    tstat     = float(ols_tstat(topic_id))

    q66 = float(prop_df["mean_prop"].quantile(BUCKET_CFG["strong_prop_quantile"]))
    q20 = float(prop_df["mean_prop"].quantile(BUCKET_CFG["latent_prop_quantile"]))
    median_prop = float(prop_df["mean_prop"].median())

    # ---------- STRONG ----------
    if (avg_share >= q66) and (stab >= BUCKET_CFG["stability_min_strong"]) and (cagr_all >= 0.0):
        return "STRONG"

    # ---------- WEAK ----------
    if ((tstat > BUCKET_CFG["weak_tstat_min"]) or (rec_g > BUCKET_CFG["weak_recent_cagr"])) \
       and (last_share >= median_prop):
        return "WEAK"

    # ---------- LATENT ----------
    if avg_share <= q20:
        return "LATENT"

    # ---------- default ----------
    return "NSWK"


# =========================================================
# TOP-WORD EXTRACTION 
# =========================================================

print("[TOPWORDS] Bereite Top-Word-Extraktion vor …")

from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1) word_scores_top10 – Top-10 Wörter mit c-TF-IDF Scores
# ---------------------------------------------------------
def get_word_scores_top10(model: BERTopic, topic_id: int) -> list:
    """Gibt die 10 Wörter mit den höchsten c-TF-IDF Scores zurück."""
    try:
        topic_info = model.get_topic(topic_id)
        if topic_info is None:
            return []
        return [(w, float(s)) for w, s in topic_info[:10]]
    except Exception:
        return []


# ---------------------------------------------------------
# 2) topic_words_top15 – häufigste/exklusivste Wörter
# ---------------------------------------------------------
def get_topic_words_top15(model: BERTopic, topic_id: int) -> list:
    """Gibt die 15 wichtigsten Wörter (Frequenz + Exklusivität) zurück."""
    try:
        topic_info = model.get_topic(topic_id)
        if topic_info is None:
            return []
        return [w for w, _ in topic_info[:15]]
    except Exception:
        return []


# ---------------------------------------------------------
# 3) representative_terms – Wörter nahe am Topic-Zentrum
# ---------------------------------------------------------
def get_representative_terms(model: BERTopic, topic_id: int, top_n: int = 10) -> list:
    """
    Verwendet den c-TF-IDF-Vektor eines Topics als centroid-artige Repräsentation.
    Die am ähnlichsten liegenden Dimensionswörter werden zurückgegeben.
    """
    try:
        if model.c_tf_idf_ is None:
            return []

        topics_list = list(model.get_topics().keys())
        if topic_id not in topics_list:
            return []

        idx = topics_list.index(topic_id)
        centroid_vec = model.c_tf_idf_[idx].reshape(1, -1)

        sims = cosine_similarity(centroid_vec, model.c_tf_idf_)[0]
        top_idx = sims.argsort()[-top_n:][::-1]

        vocab = model.vectorizer_model.get_feature_names_out()
        return [vocab[i] for i in top_idx]
    except Exception:
        return []


# ---------------------------------------------------------
# 4) diversity_terms – MMR-basierte Wortdiversität
# ---------------------------------------------------------
def mmr(candidate_vectors, centroid_vector, lambda_param=0.5, top_n=10):
    """Standard MMR Algorithmus."""
    selected = []
    centroid_vector = centroid_vector.reshape(1, -1)

    while len(selected) < top_n and len(selected) < len(candidate_vectors):
        mmr_scores = []
        for i, v in enumerate(candidate_vectors):
            if i in selected:
                continue

            sim_centroid = cosine_similarity(v.reshape(1, -1), centroid_vector)[0][0]

            sim_selected = 0
            if selected:
                sim_selected = max(
                    cosine_similarity(v.reshape(1, -1), candidate_vectors[j].reshape(1, -1))[0][0]
                    for j in selected
                )

            score = lambda_param * sim_centroid - (1 - lambda_param) * sim_selected
            mmr_scores.append((i, score))

        if not mmr_scores:
            break

        best = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best)

    return selected


def get_diversity_terms(model: BERTopic, topic_id: int, top_n: int = 10) -> list:
    """Gibt diverseste Top-Wörter basierend auf MMR zurück."""
    try:
        topic_info = model.get_topic(topic_id)
        if topic_info is None:
            return []

        # Größerer Kandidatenpool erhöht Diversität
        candidate_words = [w for w, _ in topic_info[:40]]
        vocab = model.vectorizer_model.get_feature_names_out()
        vocab_index = {v: i for i, v in enumerate(vocab)}

        if model.c_tf_idf_ is None:
            return []

        topics_list = list(model.get_topics().keys())
        topic_idx = topics_list.index(topic_id)

        centroid = model.c_tf_idf_[topic_idx]

        candidate_vecs = []
        valid_words = []
        for w in candidate_words:
            if w not in vocab_index:
                continue
            i = vocab_index[w]
            # Wir nutzen nur die Spalte i des c-TF-IDF → 1D embedding der Wortdimension
            # und wandeln sie in ein konsistentes Format
            v = np.array([model.c_tf_idf_[topic_idx, i]])
            candidate_vecs.append(v)
            valid_words.append(w)

        if not candidate_vecs:
            return []

        candidate_vecs = np.array(candidate_vecs, dtype=float)
        centroid_vec = centroid.astype(float)

        sel = mmr(candidate_vecs, centroid_vec, lambda_param=0.5, top_n=top_n)
        return [valid_words[i] for i in sel]

    except Exception:
        return []


# =========================================================
# TOPWORD-PRECOMPUTATION 
# =========================================================

print("[TOPWORDS] Extrahiere Top-Words für alle Topics …")

topic_word_summary = {}

all_topics = sorted(df_valid["topic"].unique())
for t in all_topics:
    topic_word_summary[t] = {
        "word_scores_top10": get_word_scores_top10(topic_model, t),
        "topic_words_top15": get_topic_words_top15(topic_model, t),
        "representative_terms": get_representative_terms(topic_model, t),
        "diversity_terms": get_diversity_terms(topic_model, t),
    }


# =========================================================
# EXECUTIVE SUMMARY (mit allen Top-Word-Feldern)
# =========================================================

print("[SUMMARY] Erzeuge Executive Summary …")

support_docs_total = df_valid.groupby("topic")["full_text"].count().rename("support_docs_total")
support_years_active = prop.groupby("topic")["year"].nunique().rename("support_years_active")
peak_prop = prop.groupby("topic")["mean_prop"].max().rename("peak_prop")

rows = []
for t in sorted(df_valid["topic"].unique()):

    s = prop[prop["topic"] == t].sort_values("year")
    if s.empty:
        continue

    mean_prop_t = float(s["mean_prop"].mean())
    g_all  = float(cagr_percent(t))
    rec_g  = float(recent_cagr_percent(t))
    last_s = float(last_year_prop(t))
    stab   = float(avg_stability(t))

    xs_vals = [cross_similarity_year(t, int(y)) for y in s["year"]]
    xs_vals = [v for v in xs_vals if not np.isnan(v)]
    avg_cross = float(np.mean(xs_vals)) if xs_vals else np.nan

    py = int(s.loc[s["mean_prop"].idxmax(), "year"])
    bucket_val = label_bucket(t, prop)

    # ========== Top-Words ==========
    tw = topic_word_summary[t]
    word10 = ", ".join([f"{w}:{s:.4f}" for w, s in tw["word_scores_top10"]])
    top15  = ", ".join(tw["topic_words_top15"])
    repr10 = ", ".join(tw["representative_terms"])
    div10  = ", ".join(tw["diversity_terms"])

    rows.append({
        # Standard-Kennzahlen
        "topic": t,
        "bucket": bucket_val,
        "mean_prop": mean_prop_t,
        "growth_pct": round(g_all, 2),
        "recent_cagr_pct": round(rec_g, 2),
        "last_year_prop": last_s,
        "peak_prop": float(peak_prop.loc[t]) if t in peak_prop.index else np.nan,
        "support_docs_total": int(support_docs_total.loc[t]) if t in support_docs_total.index else 0,
        "support_years_active": int(support_years_active.loc[t]) if t in support_years_active.index else 0,
        "peak_year": py,
        "avg_stability_pct": None if np.isnan(stab) else round(stab * 100, 2),
        "avg_crosssim_pct": None if np.isnan(avg_cross) else round(avg_cross * 100, 2),

        # TOP-WORD Felder (für LLM-Labeling)
        "word_scores_top10": word10,
        "topic_words_top15": top15,
        "representative_terms": repr10,
        "diversity_terms": div10,
    })

summary = pd.DataFrame(rows).sort_values(["bucket", "mean_prop"], ascending=[True, False])


# =========================================================
# EXPORT – Executive Summary
# =========================================================

OUT_SUMMARY_CSV  = os.path.join(OUT_DIR, "executive_summary_yearly_full_optimiert.csv")
OUT_SUMMARY_XLSX = os.path.join(OUT_DIR, "executive_summary_yearly_full_optimiert.xlsx")

# CSV Export
summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8", float_format="%.6f")

# XLSX Export (inkl. Top-Words)
with pd.ExcelWriter(OUT_SUMMARY_XLSX, engine="openpyxl") as writer:
    summary.to_excel(writer, index=False, sheet_name="summary")

print(f"[OK] Summary CSV gespeichert:  {OUT_SUMMARY_CSV}")
print(f"[OK] Summary XLSX gespeichert: {OUT_SUMMARY_XLSX}")


# =========================================================
# TOP-k NEIGHBORS PRO JAHR
# =========================================================

TOPK_NEIGHBORS_K = 3

def topk_neighbors_year(rep_vecs: Dict[Tuple[int, int], np.ndarray],
                        y: int,
                        k: int = TOPK_NEIGHBORS_K):

    items = [(t, vec) for (t, yy), vec in rep_vecs.items() if yy == y]

    if len(items) < 2:
        return pd.DataFrame()

    topics_y = [t for t, _ in items]
    X = np.vstack([v for _, v in items])
    S = cosine_similarity(X, X)

    rows_out = []

    for i, t in enumerate(topics_y):
        sims = S[i].copy()
        sims[i] = -1.0
        idx = np.argsort(-sims)[:k]

        for rank, j in enumerate(idx, start=1):
            rows_out.append({
                "year": y,
                "topic": int(t),
                "neighbor": int(topics_y[j]),
                "cosine": float(sims[j]),
                "rank": rank
            })

    return pd.DataFrame(rows_out)


print("[TOPK] Berechne Top-k-Nachbarn je Jahr …")

all_topk = []
years = sorted(set(y for (_, y) in rep_vecs.keys()))

os.makedirs(TOPK_DIR, exist_ok=True)

for y in years:
    dfy = topk_neighbors_year(rep_vecs, y, TOPK_NEIGHBORS_K)
    if dfy.empty:
        print(f"[TOPK] {y}: Zu wenige Topics – übersprungen.")
        continue

    out_y = os.path.join(TOPK_DIR, f"top{TOPK_NEIGHBORS_K}_neighbors_{y}.csv")
    dfy.to_csv(out_y, index=False, float_format="%.4f")
    print(f"[TOPK] Jahr {y} gespeichert → {out_y}")

    all_topk.append(dfy)

if all_topk:
    df_all = pd.concat(all_topk, ignore_index=True)
    out_all = os.path.join(TOPK_DIR, f"top{TOPK_NEIGHBORS_K}_neighbors_all_years.csv")
    df_all.to_csv(out_all, index=False, float_format="%.4f")
    print(f"[TOPK] Gesamttabelle gespeichert → {out_all}")


# =========================================================
# COMMUNITY-GRAPHS & HEATMAPS
# =========================================================

os.makedirs(COMMUNITY_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

EDGE_ALPHA = 0.25
SHAPE_BY_BUCKET = {
    "STRONG": "s",
    "WEAK": "^",
    "LATENT": "o",
    "NSWK": "D",
    "UNK": "o"
}

def _bucket_for_topic(topic_id: int) -> str:
    try:
        return summary.loc[summary["topic"] == topic_id, "bucket"].iloc[0]
    except Exception:
        return "UNK"


# ----------------------------- COMMUNITY GRAPH -----------------------------
def plot_community_graph(year: int):
    nodes = [t for (t, y) in rep_vecs.keys() if y == year]
    if len(nodes) < 2:
        return

    X = np.vstack([rep_vecs[(t, year)] for t in nodes])
    S = cosine_similarity(X, X)

    tri = np.triu_indices_from(S, k=1)
    if len(tri[0]) == 0:
        return

    thr = np.quantile(S[tri], EDGE_PCTL)

    G = nx.Graph()
    for t in nodes:
        G.add_node(int(t))

    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if S[i, j] >= thr:
                G.add_edge(int(nodes[i]), int(nodes[j]), weight=float(S[i, j]))

    if G.number_of_edges() == 0:
        return

    pos = nx.spring_layout(G, seed=SEED, k=0.6, weight="weight")

    plt.figure(figsize=(8, 6), dpi=130)

    widths = [G[u][v]["weight"] * 1.5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=EDGE_ALPHA)

    for bucket, shape in SHAPE_BY_BUCKET.items():
        nlist = [n for n in G.nodes if _bucket_for_topic(n) == bucket]
        if nlist:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nlist,
                node_shape=shape,
                edgecolors="k",
                linewidths=0.7,
                node_size=350
            )

    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes}, font_size=7)

    plt.title(f"Topic Community Graph – {year}")
    out = os.path.join(COMMUNITY_DIR, f"community_{year}.png")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


# ------------------------------- HEATMAP -------------------------------
def plot_heatmap(year: int):
    nodes = [t for (t, y) in rep_vecs.keys() if y == year]
    if len(nodes) < 2:
        return

    X = np.vstack([rep_vecs[(t, year)] for t in nodes])
    S = cosine_similarity(X, X)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
    im = ax.imshow(S, vmin=0, vmax=1)

    ax.set_xticks(range(len(nodes)))
    ax.set_yticks(range(len(nodes)))
    ax.set_xticklabels(nodes, rotation=90, fontsize=7)
    ax.set_yticklabels(nodes, fontsize=7)

    plt.colorbar(im, ax=ax, label="cosine similarity")
    plt.title(f"Topic Similarity Heatmap – {year}")

    out = os.path.join(HEATMAP_DIR, f"heatmap_{year}.png")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


print("[PLOTS] Erzeuge Community-Graphs & Heatmaps …")

for y in years:
    plot_community_graph(y)
    plot_heatmap(y)


# =========================================================
# HISTOGRAMME
# =========================================================

def plot_histograms(df_summary: pd.DataFrame):

    # mean_prop
    plt.figure(figsize=(7, 5), dpi=130)
    plt.hist(df_summary["mean_prop"].dropna().astype(float), bins=30)
    plt.xlabel("mean_prop")
    plt.ylabel("count")
    plt.title("Distribution of mean_prop")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "hist_mean_prop.png"))
    plt.close()

    # CAGR
    plt.figure(figsize=(7, 5), dpi=130)
    plt.hist(df_summary["growth_pct"].dropna().astype(float), bins=30)
    plt.xlabel("growth_pct (CAGR, %)")
    plt.ylabel("count")
    plt.title("Distribution of growth_pct")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "hist_growth_pct.png"))
    plt.close()


print("[PLOTS] Erzeuge Histogramme …")
plot_histograms(summary)


# =========================================================
# AUDIT TRAIL
# =========================================================

audit = {
    "seed": SEED,
    "device": DEVICE,
    "embedding_model": "AI-Growth-Lab/PatentSBERTa",
    "emb_batch": EMB_BATCH,
    "vectorizer": {
        "ngram_range": [1, 2],
        "min_df": 5,
        "max_df": 0.90
    },
    "umap": {"n_neighbors": 15, "n_components": 5, "min_dist": 0.0},
    "hdbscan": {"min_cluster_size": 50, "min_samples": 10},
    "merge_threshold": CENTROID_MERGE_COS_THR,
    "bucket_cfg": BUCKET_CFG,
}

with open(AUDIT_JSON, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)

print("[OK] Audit Trail gespeichert:", AUDIT_JSON)
print("\n[DONE] Pipeline abgeschlossen.")
