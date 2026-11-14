# -*- coding: utf-8 -*-
"""
Big Data Analytics Suite - one Streamlit app for all three projects
===================================================================
  Project 1  SHEIN social-media sentiment analysis
  Project 2  AdSpend IQ: advertising performance and audience insights
  Project 3  Pakistan social audience and campaign intelligence engine

"""
from __future__ import annotations

import io
import json
import re
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

warnings.filterwarnings("ignore")

# Must be the first Streamlit call.
st.set_page_config(page_title="Big Data Analytics Suite", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Where the app looks for your real project outputs (edit if yours differ)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
P1_DIRS = [
    BASE_DIR,
    BASE_DIR / "outputs",
    BASE_DIR / "data",
    BASE_DIR / "project1",
    BASE_DIR / "project1" / "outputs",
    BASE_DIR / "shein_project_outputs",
    BASE_DIR.parent / "SHEIN-REVIEWS-SENTIMENT-ANALYSIS" / "outputs",
]
P1_FILES = ["review_comments_analyzed.csv", "review_comments_clean.csv"]
P2_DIRS = [BASE_DIR / "data"]
P3_DIRS = [BASE_DIR / "project3", BASE_DIR / "data"]


# ===========================================================================
# SHARED HELPERS
# ===========================================================================
def _skip(path: Path) -> bool:
    bad = {".venv", "venv", "env", "node_modules", "__pycache__", ".git", "site-packages"}
    return any(part in bad for part in path.parts)


def find_csvs(patterns, dirs, recursive=False, limit=40):
    """Return CSV files matching glob patterns inside the given folders."""
    found, seen = [], set()
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for pat in patterns:
            for p in (d.rglob(pat) if recursive else d.glob(pat)):
                if p in seen or _skip(p) or not p.is_file():
                    continue
                seen.add(p)
                found.append(p)
                if len(found) >= limit:
                    return found
    return found


def safe_read(src) -> pd.DataFrame:
    """Read a CSV path or uploaded file, tolerating odd encodings."""
    try:
        return pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        if hasattr(src, "seek"):
            src.seek(0)
        return pd.read_csv(src, encoding="latin-1", low_memory=False)


@st.cache_data(show_spinner=False)
def read_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    return safe_read(path)


def pick_col(df: pd.DataFrame, candidates):
    lower = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def header_has(path: Path, required: dict) -> bool:
    try:
        cols = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    except Exception:
        return False
    lower = {str(c).lower().strip() for c in cols}
    return all(any(a in lower for a in [canon] + aliases) for canon, aliases in required.items())


def map_columns(df, required: dict, key: str, optional: dict | None = None):
    """Rename columns to canonical names; offer manual mapping if some are missing.

    required / optional: {canonical_name: [aliases]}.  Returns None while required
    columns are still unmapped (the caller should stop).
    """
    lower = {str(c).lower().strip(): c for c in df.columns}
    rename, missing = {}, []
    for canon, aliases in required.items():
        hit = next((lower[a] for a in [canon] + aliases if a in lower), None)
        if hit is None:
            missing.append(canon)
        else:
            rename[hit] = canon
    for canon, aliases in (optional or {}).items():
        hit = next((lower[a] for a in [canon] + aliases if a in lower), None)
        if hit is not None and hit not in rename:
            rename[hit] = canon
    if missing:
        st.warning("Some required columns were not found: " + ", ".join(f"`{m}`" for m in missing))
        with st.expander("Map columns manually", expanded=True):
            for m in missing:
                choice = st.selectbox(
                    f"Which column is `{m}`?", ["- select -"] + list(df.columns), key=f"{key}_{m}"
                )
                if choice != "- select -":
                    rename[choice] = m
        if any(m not in rename.values() for m in missing):
            return None
    return df.rename(columns=rename)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def sdiv(a, b):
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fmt_int(x) -> str:
    return f"{int(round(x)):,}"


def fmt_money(x) -> str:
    return f"${x:,.0f}"


def fmt_pct(x, d=1) -> str:
    return f"{x * 100:.{d}f}%"


def show(fig, key=None):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, key=key)


# ===========================================================================
# PROJECT 1 - SENTIMENT ANALYSIS
# ===========================================================================
SENT_ORDER = ["positive", "neutral", "negative", "mixed"]
SENT_COLORS = {"positive": "#2E9E6B", "neutral": "#8A94A6", "negative": "#D64545", "mixed": "#E0A030"}
ENGINES = {
    "Use labels from file (if present)": "file",
    "Built-in lexicon (fast, English)": "lexicon",
    "RoBERTa (needs transformers + torch)": "roberta",
}
TEXT_COLS = ["comment_text", "comment", "text", "textoriginal", "textdisplay", "review_text",
             "review", "body", "content", "clean_text", "cleaned_text", "comment_clean"]
LABEL_COLS = ["overall_sentiment", "sentiment", "predicted_sentiment", "sentiment_label",
              "overall_label", "pred_label", "label", "final_sentiment", "overall_pred"]
MIXED_COLS = ["is_mixed", "mixed", "mixed_review", "is_mixed_review"]
VIDEO_COLS = ["video_title", "video_id", "video", "video_url", "source_video"]

POS_WORDS = set("""love loved loving lovely great good amazing awesome beautiful gorgeous cute perfect
perfectly nice pretty stunning fits comfortable comfy soft worth recommend recommended happy obsessed
favorite favourite fantastic excellent best wonderful flattering stylish affordable bargain fast
elegant adorable""".split())
NEG_WORDS = set("""bad terrible awful horrible worst poor cheaply disappointed disappointing disappointment
waste scam fake ripped tore torn small tight loose itchy thin transparent sheer smell smells stinks late
delayed never refund return returned wrong faded shrunk shrank uncomfortable ugly flimsy overpriced regret
hate hated issue issues problem problems broke broken different weird""".split())
NEGATORS = {"not", "no", "never", "don't", "dont", "didn't", "didnt", "doesn't", "doesnt", "isn't",
            "isnt", "wasn't", "wasnt", "cant", "can't", "won't", "wont", "hardly", "barely", "without"}
SPLIT_RE = re.compile(r"[.!?;\n]+|\b(?:but|however|although|though|yet|except|unfortunately|sadly)\b", re.I)

NEG_THEMES = {
    "Size & fit": ["size", "sizing", "small", "tight", "loose", "fit", "big", "large", "tiny", "short"],
    "Fabric & quality": ["fabric", "thin", "material", "quality", "cheap", "cheaply", "flimsy", "stitching",
                         "ripped", "torn", "tore", "see-through", "sheer", "itchy", "smell", "smells",
                         "faded", "shrunk", "transparent"],
    "Shipping & delivery": ["delivery", "shipping", "shipped", "arrive", "arrived", "late", "delayed",
                            "tracking", "customs", "forever"],
    "Returns & refunds": ["return", "returned", "refund", "exchange", "customer service", "support", "scam"],
    "Looks different from photos": ["photo", "photos", "picture", "pictures", "different", "expected",
                                    "website", "advertised", "colour", "color"],
    "Price & value": ["price", "expensive", "overpriced", "waste", "money"],
}
THEME_RES = {t: re.compile(r"\b(?:" + "|".join(re.escape(w) for w in ws) + r")\b", re.I)
             for t, ws in NEG_THEMES.items()}
THEME_ACTIONS = {
    "Size & fit": "publish size charts and model measurements, and say plainly when an item runs small or large",
    "Fabric & quality": "show close-ups of fabric and stitching and state the fabric composition",
    "Shipping & delivery": "give realistic delivery windows and proactive tracking updates",
    "Returns & refunds": "make return and refund steps simple, visible and fast to resolve",
    "Looks different from photos": "use unedited, well-lit photos and customer try-on content",
    "Price & value": "lead with bundle value and transparent pricing instead of discount pressure",
}

_POS_T = ["I love this dress, the fabric feels so soft", "Gorgeous colour and it fits perfectly",
          "Honestly the quality is amazing for the price", "So cute and comfortable, I am obsessed",
          "Perfect fit and it looks stunning", "Great haul, everything looks beautiful",
          "Fast delivery and the dress is lovely", "Worth every penny, highly recommend",
          "The stitching is neat and the colour is gorgeous", "Best SHEIN purchase I have made"]
_NEG_T = ["The fabric is so thin and see-through", "Delivery took forever and it arrived late",
          "Size runs very small and it was too tight", "It looked completely different from the photos",
          "The stitching ripped after one wash", "Customer service never replied about my refund",
          "Overpriced for such poor quality", "It smells weird and the colour faded",
          "I had to return it because it did not fit", "The material feels cheaply made"]
_NEU_T = ["Which size did you order?", "Where can I find the link for this dress?", "Does this come in blue?",
          "What is the model's height?", "Can you do a try-on of the green one?",
          "How long did shipping take to the UK?", "Is this the same dress from the last video?",
          "What is the discount code?"]
_VIDEOS = ["SHEIN try-on haul (summer dresses)", "SHEIN wedding guest dresses review",
           "I tested viral SHEIN dresses", "SHEIN vs Zara dress comparison",
           "Honest SHEIN review: worth it?", "SHEIN curve dress haul"]


@st.cache_data(show_spinner=False)
def synthetic_comments(n: int = 800, seed: int = 7) -> pd.DataFrame:
    """SYNTHETIC comments for demonstration only - not real YouTube data."""
    rng = np.random.default_rng(seed)
    base = np.array([0.40, 0.22, 0.24, 0.14])  # positive, negative, neutral, mixed
    vid_p = [rng.dirichlet(base * 25) for _ in _VIDEOS]
    rows = []
    for i in range(n):
        v = int(rng.integers(len(_VIDEOS)))
        kind = rng.choice(4, p=vid_p[v])
        if kind == 0:
            text = rng.choice(_POS_T)
        elif kind == 1:
            text = rng.choice(_NEG_T)
        elif kind == 2:
            text = rng.choice(_NEU_T)
        else:
            neg = str(rng.choice(_NEG_T))
            text = f"{rng.choice(_POS_T)}, but {neg[0].lower() + neg[1:]}"
        rows.append((f"c{i:05d}", _VIDEOS[v], str(text), int(rng.poisson(3))))
    return pd.DataFrame(rows, columns=["comment_id", "video_title", "comment_text", "like_count"])


def lex_label(text: str) -> str:
    """Lexicon score per clause. A negator flips the next sentiment word within 3 tokens,
    but its scope ends at the first sentiment word it meets."""
    score = 0
    for clause in re.split(r"[,.;!?\n]+", text.lower()):
        toks = re.findall(r"[a-z']+", clause)
        for i, t in enumerate(toks):
            s = 1 if t in POS_WORDS else -1 if t in NEG_WORDS else 0
            if not s:
                continue
            for w in reversed(toks[max(0, i - 3):i]):
                if w in NEGATORS:
                    s = -s
                    break
                if w in POS_WORDS or w in NEG_WORDS:
                    break
            score += s
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def split_segments(text: str):
    parts = [p.strip(" ,-") for p in SPLIT_RE.split(text)]
    return [p for p in parts if len(p) >= 3]


@st.cache_resource(show_spinner="Loading the RoBERTa model (first run downloads ~500 MB)...")
def load_roberta():
    from transformers import pipeline  # imported lazily: optional dependency

    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    truncation=True, max_length=128)


def roberta_labels(texts):
    out = load_roberta()(list(texts), batch_size=32)
    return [o["label"].lower() for o in out]


def norm_label(x) -> str:
    s = str(x).strip().lower()
    if s in {"positive", "pos", "label_2"}:
        return "positive"
    if s in {"negative", "neg", "label_0"}:
        return "negative"
    if s == "mixed":
        return "mixed"
    return "neutral"


@st.cache_data(show_spinner="Analysing comments...")
def analyze_sentiment(texts: tuple, engine: str):
    """Segment-level sentiment. A comment is 'mixed' when at least one segment is positive
    AND at least one is negative (a derived flag, not a trained fourth class)."""
    texts = list(texts)
    segs = [split_segments(t) or [t] for t in texts]
    flat = [s for ss in segs for s in ss]
    labeller = lambda xs: [lex_label(x) for x in xs]  # noqa: E731
    used = "lexicon"
    if engine == "roberta":
        try:
            roberta_labels(["test"])
            labeller, used = roberta_labels, "roberta"
        except Exception as exc:  # transformers/torch missing or no internet
            used = f"lexicon (RoBERTa unavailable: {type(exc).__name__})"
    seg_lab = labeller(flat)
    multi = [i for i, ss in enumerate(segs) if len(ss) > 1]
    over = dict(zip(multi, labeller([texts[i] for i in multi]))) if multi else {}
    rows, pos = [], 0
    for i, ss in enumerate(segs):
        sl = seg_lab[pos:pos + len(ss)]
        pos += len(ss)
        overall = over.get(i, sl[0])
        mixed = "positive" in sl and "negative" in sl
        neg_text = " . ".join(s for s, lab in zip(ss, sl) if lab == "negative")
        seg_str = " | ".join(f"{s} [{lab}]" for s, lab in zip(ss, sl))
        rows.append((overall, mixed, "mixed" if mixed else overall, neg_text, seg_str))
    cols = ["overall_sentiment", "is_mixed", "final_sentiment", "neg_text", "segments"]
    return pd.DataFrame(rows, columns=cols), used


def analyze_dataframe(raw: pd.DataFrame, tcol: str, engine: str, max_rows: int):
    df = raw.copy()
    df["text"] = df[tcol].astype(str).str.strip()
    df = df[df["text"].str.len() >= 3].reset_index(drop=True)
    vcol = pick_col(df, VIDEO_COLS)
    df["video"] = df[vcol].astype(str).str.slice(0, 45) if vcol else "All comments"
    lab_col, mixed_col = pick_col(raw, LABEL_COLS), pick_col(raw, MIXED_COLS)
    if engine == "file" and lab_col:
        df["overall_sentiment"] = df[lab_col].map(norm_label)
        flag = (df[mixed_col].astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})
                if mixed_col else df["overall_sentiment"].eq("mixed"))
        df["is_mixed"] = flag
        df["final_sentiment"] = np.where(flag, "mixed", df["overall_sentiment"])
        df["neg_text"] = np.where(df["overall_sentiment"].eq("negative"), df["text"], "")
        df["segments"] = ""
        return df, "labels from file"
    sub = df.head(max_rows) if engine == "roberta" else df
    sub = sub.drop(columns=[c for c in ["overall_sentiment", "is_mixed", "final_sentiment",
                                         "neg_text", "segments"] if c in sub.columns]).reset_index(drop=True)
    res, used = analyze_sentiment(tuple(sub["text"]), "roberta" if engine == "roberta" else "lexicon")
    return pd.concat([sub, res], axis=1), used


def theme_counts(df: pd.DataFrame) -> pd.DataFrame:
    texts = df.loc[df["neg_text"].str.len() > 0, "neg_text"]
    rows = [(t, int(texts.str.contains(rx).sum())) for t, rx in THEME_RES.items()]
    return pd.DataFrame(rows, columns=["theme", "comments"]).sort_values("comments", ascending=False)


def summarize_p1(df, themes, source):
    top = themes.iloc[0] if len(themes) and themes.iloc[0]["comments"] > 0 else None
    return {
        "source": source,
        "comments": int(len(df)),
        "share": {s: float((df["final_sentiment"] == s).mean()) for s in SENT_ORDER},
        "top_negative_theme": None if top is None else str(top["theme"]),
        "top_theme_comments": 0 if top is None else int(top["comments"]),
    }


def default_p1_summary():
    auto = find_csvs(P1_FILES, P1_DIRS)
    raw, source = None, "synthetic demo"
    if auto:
        try:
            raw, source = read_csv_cached(str(auto[0]), auto[0].stat().st_mtime), f"file: {auto[0].name}"
            tcol = pick_col(raw, TEXT_COLS)
            if tcol is None:
                raw = None
        except Exception:
            raw = None
    if raw is None:
        raw, source = synthetic_comments(800), "synthetic demo"
        tcol = "comment_text"
    df, _ = analyze_dataframe(raw, tcol, "file", 300)
    return summarize_p1(df, theme_counts(df), source)


def p1_report(df, themes, used, source) -> str:
    n = len(df)
    pct = {s: (df["final_sentiment"] == s).mean() * 100 for s in SENT_ORDER}
    lines = [
        "SHEIN SOCIAL-MEDIA SENTIMENT - MARKETING REPORT", "=" * 50,
        f"Generated : {datetime.now():%Y-%m-%d %H:%M}", f"Source    : {source}",
        f"Engine    : {used}", f"Comments  : {n:,}", "", "RESULTS",
    ]
    lines += [f"  {s.capitalize():9s} {pct[s]:5.1f}%  ({int((df['final_sentiment'] == s).sum()):,})" for s in SENT_ORDER]
    lines += ["", "TOP NEGATIVE THEMES (keyword-derived)"]
    shown = themes[themes["comments"] > 0].head(3)
    lines += [f"  - {r.theme}: {r.comments:,} comments" for r in shown.itertuples()] or ["  (none detected)"]
    lines += ["", "RECOMMENDATION"]
    lean = "positive" if pct["positive"] >= pct["negative"] else "negative"
    lines.append(f"  Conversation leans {lean}: {pct['positive']:.0f}% positive vs {pct['negative']:.0f}% negative.")
    if pct["mixed"] >= 8:
        lines.append(f"  {pct['mixed']:.0f}% of comments are mixed (praise plus a specific complaint) - the most actionable feedback.")
    for r in shown.itertuples():
        lines.append(f"  - {r.theme}: {THEME_ACTIONS[r.theme]}.")
    lines += ["", "LIMITATIONS",
              "  - YouTube comments are not verified purchases or SHEIN product reviews.",
              "  - Commenters are self-selected; search ranking, sarcasm, humour and mixed languages bias results.",
              "  - Mixed = one positive and one negative segment (heuristic split on contrast words).",
              "  - Theme counts are keyword matches, not a trained topic model."]
    if "synthetic" in source:
        lines.append("  - THIS RUN USED SYNTHETIC DEMO DATA. Do not cite it as evidence.")
    return "\n".join(lines)


def page_sentiment():
    st.title("💬 Sentiment analysis")
    st.caption("Project 1: SHEIN social-listening study. Public comments are classified as positive, "
               "negative, neutral or mixed.")
    auto = find_csvs(P1_FILES, P1_DIRS)
    with st.sidebar:
        st.markdown("### Project 1 settings")
        options = ["Synthetic demo comments", "Upload CSV"]
        if auto:
            options.insert(0, f"Project 1 output: {auto[0].name}")
        src = st.radio("Data source", options, key="p1_src")
        engine = ENGINES[st.radio("Sentiment engine", list(ENGINES), key="p1_engine")]
        n_demo, upload, max_rows = 800, None, 300
        if src == "Synthetic demo comments":
            n_demo = st.slider("Demo comments", 200, 3000, 800, 100, key="p1_n")
        if src == "Upload CSV":
            upload = st.file_uploader("Comments CSV", type="csv", key="p1_up")
        if engine == "roberta":
            max_rows = st.slider("Max comments for RoBERTa (CPU is slow)", 50, 2500, 300, 50, key="p1_max")

    if src == "Upload CSV":
        if upload is None:
            st.info("Upload a CSV with a comment-text column, or switch to the demo data in the sidebar.")
            st.stop()
        raw, source = safe_read(upload), f"upload: {upload.name}"
    elif src.startswith("Project 1 output"):
        raw, source = read_csv_cached(str(auto[0]), auto[0].stat().st_mtime), f"file: {auto[0].name}"
    else:
        raw, source = synthetic_comments(n_demo), "synthetic demo"
        st.warning("Showing SYNTHETIC demo comments. Do not cite these numbers as real findings.")

    tcol = pick_col(raw, TEXT_COLS)
    if tcol is None:
        tcol = st.sidebar.selectbox("Which column holds the comment text?", list(raw.columns), key="p1_tcol")
    df, used = analyze_dataframe(raw, tcol, engine, max_rows)
    if engine == "file" and used != "labels from file":
        st.info("No sentiment column found in this file, so the built-in analyser was used.")
    if df.empty:
        st.error("No usable comments found.")
        st.stop()
    themes = theme_counts(df)
    st.session_state["sum_p1"] = summarize_p1(df, themes, source)
    st.caption(f"Source: {source}  |  Engine: {used}  |  Comments: {len(df):,}")

    tab_dash, tab_explore, tab_live, tab_report = st.tabs(
        ["Dashboard", "Comment explorer", "Try it live", "Report & validation"])

    with tab_dash:
        cols = st.columns(5)
        cols[0].metric("Comments", fmt_int(len(df)))
        for col, s in zip(cols[1:], SENT_ORDER):
            col.metric(s.capitalize(), fmt_pct((df["final_sentiment"] == s).mean()))
        c1, c2 = st.columns(2)
        dist = pd.DataFrame({"sentiment": SENT_ORDER,
                             "comments": [int((df["final_sentiment"] == s).sum()) for s in SENT_ORDER]})
        with c1:
            show(px.bar(dist, x="sentiment", y="comments", color="sentiment", text="comments",
                        color_discrete_map=SENT_COLORS, title="Sentiment distribution"), "p1_dist")
        with c2:
            top_vids = df["video"].value_counts().head(12).index
            ct = pd.crosstab(df.loc[df["video"].isin(top_vids), "video"],
                             df.loc[df["video"].isin(top_vids), "final_sentiment"])
            ct = ct.reindex(columns=SENT_ORDER, fill_value=0)
            pct = (ct.div(ct.sum(axis=1), axis=0) * 100).reset_index().melt(
                id_vars="video", var_name="sentiment", value_name="percent")
            show(px.bar(pct, y="video", x="percent", color="sentiment", orientation="h",
                        color_discrete_map=SENT_COLORS, title="Sentiment by video (%)"), "p1_video")
        c3, c4 = st.columns(2)
        with c3:
            show(px.bar(themes.sort_values("comments"), x="comments", y="theme", orientation="h",
                        title="Negative themes (keyword-derived)", color_discrete_sequence=["#D64545"]), "p1_themes")
        with c4:
            stop = set(ENGLISH_STOP_WORDS) | {"dress", "dresses", "shein", "just", "really"}
            words = Counter(w for t in df.loc[df["neg_text"].str.len() > 0, "neg_text"]
                            for w in re.findall(r"[a-z']{3,}", t.lower()) if w not in stop)
            wdf = pd.DataFrame(words.most_common(15), columns=["word", "count"]).sort_values("count")
            if wdf.empty:
                st.info("No negative segments to summarise.")
            else:
                show(px.bar(wdf, x="count", y="word", orientation="h",
                            title="Frequent words in negative segments"), "p1_words")

    with tab_explore:
        c1, c2 = st.columns([1, 2])
        pick = c1.multiselect("Sentiment", SENT_ORDER, default=SENT_ORDER, key="p1_pick")
        query = c2.text_input("Search text", key="p1_query")
        view = df[df["final_sentiment"].isin(pick)]
        if query:
            view = view[view["text"].str.contains(query, case=False, regex=False)]
        st.caption(f"{len(view):,} comments match")
        cols_show = [c for c in ["video", "text", "final_sentiment", "segments"] if c in view.columns]
        st.dataframe(view[cols_show].head(500), hide_index=True)
        st.download_button("Download analysed comments (CSV)", csv_bytes(df.drop(columns=["neg_text"])),
                           "review_comments_analyzed_app.csv", "text/csv")

    with tab_live:
        text = st.text_area("Type or paste a comment", "I love the colour of this dress, but the fabric feels thin.",
                            key="p1_live_text")
        if st.button("Analyse comment", key="p1_live_btn") and text.strip():
            res, used_live = analyze_sentiment((text.strip(),), "roberta" if engine == "roberta" else "lexicon")
            r = res.iloc[0]
            st.markdown(f"**Final label:** {r['final_sentiment']}  |  **Overall:** {r['overall_sentiment']}  |  "
                        f"**Mixed:** {'yes' if r['is_mixed'] else 'no'}  |  engine: {used_live}")
            icon = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
            for seg in r["segments"].split(" | "):
                s, lab = seg.rsplit(" [", 1)
                st.markdown(f"{icon.get(lab.strip(']'), '⚪')} {s}  `{lab.strip(']')}`")

    with tab_report:
        report = p1_report(df, themes, used, source)
        st.text(report)
        c1, c2 = st.columns(2)
        c1.download_button("Download marketing report (TXT)", report, "marketing_report.txt")
        sample = df.sample(min(40, len(df)), random_state=1)[["text", "final_sentiment"]].rename(
            columns={"final_sentiment": "predicted_sentiment"})
        sample["manual_label"] = ""
        c2.download_button("Download 40-comment manual review sheet (CSV)", csv_bytes(sample),
                           "manual_validation_sample.csv", "text/csv")
        st.caption("Label the sheet by hand, then compare with the predictions to estimate error rates.")


# ===========================================================================
# PROJECT 2 - ADSPEND IQ (PERFORMANCE & AUDIENCE INSIGHTS)
# ===========================================================================
REQ2 = ["date", "campaign", "platform", "impressions", "clicks", "spend",
        "conversions", "revenue", "age", "gender", "city", "device"]
NUM2 = ["impressions", "clicks", "spend", "conversions", "revenue"]
AGE_BINS = [0, 17, 24, 34, 44, 54, 64, 200]
AGE_LABELS = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
PLAT_PARAMS = {  # ctr, cvr, cpc, average order value
    "Google Ads": (0.045, 0.060, 1.10, 85), "Meta": (0.020, 0.035, 0.65, 70),
    "TikTok": (0.015, 0.020, 0.40, 55), "YouTube": (0.010, 0.018, 0.50, 65),
    "LinkedIn": (0.008, 0.030, 2.20, 140), "Snapchat": (0.012, 0.015, 0.35, 50),
}
CAMPAIGNS_2 = ["Spring Launch", "Summer Sale", "Back to School", "Festive Season",
               "Brand Awareness", "Retargeting Push", "New Customer Drive", "Winter Clearance"]
CITIES_2 = ["Lahore", "Karachi", "Islamabad", "Dubai", "London", "New York", "Toronto", "Singapore"]
METRIC_LABELS = {"roas": "ROAS", "cvr": "Conversion rate", "ctr": "Click-through rate", "cpa": "Cost per acquisition",
                 "revenue": "Revenue", "conversions": "Conversions", "spend": "Spend"}


def _generate_adspend(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plat = rng.choice(list(PLAT_PARAMS), n, p=[0.28, 0.30, 0.14, 0.10, 0.08, 0.10])
    pp = pd.DataFrame(PLAT_PARAMS, index=["ctr", "cvr", "cpc", "aov"]).T.loc[plat].to_numpy()
    camp_idx = rng.integers(0, len(CAMPAIGNS_2), n)
    camp_mod = np.random.default_rng(1).uniform(0.8, 1.25, len(CAMPAIGNS_2))[camp_idx]
    age = np.clip(rng.normal(34, 11, n), 18, 65).astype(int)
    age_mod = np.select([age < 25, age < 35, age < 45], [0.9, 1.2, 1.1], 0.85)
    dev = rng.choice(["Mobile", "Desktop", "Tablet"], n, p=[0.62, 0.30, 0.08])
    ctr_mod = pd.Series(dev).map({"Mobile": 1.1, "Desktop": 0.9, "Tablet": 1.0}).to_numpy()
    cvr_mod = pd.Series(dev).map({"Mobile": 0.85, "Desktop": 1.3, "Tablet": 1.0}).to_numpy()
    imp = rng.lognormal(7.6, 0.7, n).astype(int) + 200
    clicks = rng.binomial(imp, np.clip(pp[:, 0] * ctr_mod * rng.lognormal(0, 0.15, n), 0, 0.5))
    spend = np.round(clicks * pp[:, 2] * rng.lognormal(0, 0.2, n) + imp / 1000 * 0.2, 2)
    conv = rng.binomial(clicks, np.clip(pp[:, 1] * cvr_mod * age_mod * camp_mod, 0, 0.9))
    revenue = np.round(conv * pp[:, 3] * rng.lognormal(0, 0.25, n), 2)
    return pd.DataFrame({
        "date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        "campaign": np.array(CAMPAIGNS_2)[camp_idx], "platform": plat, "impressions": imp,
        "clicks": clicks, "spend": spend, "conversions": conv, "revenue": revenue, "age": age,
        "gender": rng.choice(["Female", "Male", "Other"], n, p=[0.49, 0.47, 0.04]),
        "city": rng.choice(CITIES_2, n), "device": dev,
    })


def validate_adspend(df: pd.DataFrame):
    missing = [c for c in REQ2 if c not in df.columns]
    if missing:
        return None, [f"Missing required columns: {', '.join(missing)}"]
    d = df[REQ2].copy()
    errors = []
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in NUM2 + ["age"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["campaign", "platform", "gender", "city", "device"]:
        d[c] = d[c].astype("string").str.strip()
        d.loc[d[c] == "", c] = pd.NA
    bad = d.isna().any(axis=1)
    if bad.any():
        errors.append(f"{int(bad.sum()):,} rows have missing or invalid values in required columns.")
        return None, errors
    if (d[NUM2 + ["age"]] < 0).any().any():
        errors.append("Numeric metrics must be non-negative.")
    if (d["clicks"] > d["impressions"]).any():
        errors.append(f"{int((d['clicks'] > d['impressions']).sum()):,} rows have clicks greater than impressions.")
    if (d["conversions"] > d["clicks"]).any():
        errors.append(f"{int((d['conversions'] > d['clicks']).sum()):,} rows have conversions greater than clicks.")
    if errors:
        return None, errors
    d["age"] = d["age"].round().astype(int)
    for c in ["campaign", "platform", "gender", "city", "device"]:
        d[c] = d[c].astype(str)
    return d, []


def add_derived(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["age_group"] = pd.cut(d["age"], AGE_BINS, labels=AGE_LABELS)
    d["month"] = d["date"].dt.to_period("M").astype(str)
    return d


@st.cache_data(show_spinner="Generating synthetic ad data...")
def load_demo_adspend(n: int) -> pd.DataFrame:
    clean, _ = validate_adspend(_generate_adspend(n))
    return add_derived(clean)


def add_ratios(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["ctr"], g["cvr"] = sdiv(g["clicks"], g["impressions"]), sdiv(g["conversions"], g["clicks"])
    g["cpc"], g["cpa"] = sdiv(g["spend"], g["clicks"]), sdiv(g["spend"], g["conversions"])
    g["roas"] = sdiv(g["revenue"], g["spend"])
    return g


def agg_metrics(df: pd.DataFrame, by) -> pd.DataFrame:
    g = df.groupby(by, observed=True)[NUM2].sum().reset_index()
    return add_ratios(g)


def optimise_budget(cur, roas, aggr, lo, hi):
    """Shift budget towards efficient channels, bounded to [lo, hi] x current spend."""
    cur, roas = np.asarray(cur, float), np.asarray(roas, float)
    total = cur.sum()
    w = np.maximum(cur * np.power(np.maximum(roas, 1e-6), aggr), 1e-9)
    new = w / w.sum() * total
    lo_b, hi_b = cur * lo, cur * hi
    for _ in range(60):
        new = np.clip(new, lo_b, hi_b)
        diff = total - new.sum()
        if abs(diff) < 1e-6 * total:
            break
        mask = (new < hi_b - 1e-9) if diff > 0 else (new > lo_b + 1e-9)
        if not mask.any():
            break
        new[mask] += diff * w[mask] / w[mask].sum()
    return np.clip(new, lo_b, hi_b)


def project_revenue(cur_spend, cur_rev, new_spend, elasticity):
    cur_spend, new_spend = np.asarray(cur_spend, float), np.asarray(new_spend, float)
    ratio = np.where(cur_spend > 0, new_spend / np.maximum(cur_spend, 1e-9), 1.0)
    return np.asarray(cur_rev, float) * np.power(ratio, elasticity)


def run_benchmark(sizes):
    rows = []
    for n in sizes:
        t0 = time.perf_counter()
        raw = _generate_adspend(n, 1)
        t1 = time.perf_counter()
        clean, _ = validate_adspend(raw)
        d = add_derived(clean)
        t2 = time.perf_counter()
        for by in ["platform", "campaign", ["age_group", "device"], "month"]:
            agg_metrics(d, by)
        t3 = time.perf_counter()
        rows.append({"rows": n, "generate_s": t1 - t0, "validate_s": t2 - t1, "aggregate_s": t3 - t2,
                     "total_s": t3 - t0, "memory_mb": d.memory_usage(deep=True).sum() / 1e6})
    return pd.DataFrame(rows)


def summarize_p2(f: pd.DataFrame, source: str) -> dict:
    tot = f[NUM2].sum()
    plat = agg_metrics(f, "platform").sort_values("roas", ascending=False)
    out = {"source": source, "rows": int(len(f)), "spend": float(tot["spend"]), "revenue": float(tot["revenue"]),
           "conversions": float(tot["conversions"]), "roas": float(tot["revenue"] / tot["spend"]) if tot["spend"] else 0.0,
           "best_platform": plat.iloc[0]["platform"], "best_platform_roas": float(plat.iloc[0]["roas"]),
           "worst_platform": plat.iloc[-1]["platform"], "worst_platform_roas": float(plat.iloc[-1]["roas"])}
    for key, col in [("best_age_group", "age_group"), ("best_device", "device"), ("best_city", "city")]:
        g = agg_metrics(f, col)
        g = g[g["spend"] > 0].sort_values("roas", ascending=False)
        out[key] = str(g.iloc[0][col]) if len(g) else "n/a"
    return out


def default_p2_summary():
    return summarize_p2(load_demo_adspend(20000), "synthetic demo")


def prepare_adspend(raw: pd.DataFrame, key: str) -> pd.DataFrame:
    mapped = map_columns(raw, {c: [] for c in REQ2}, key)
    if mapped is None:
        st.stop()
    clean, errors = validate_adspend(mapped)
    if errors:
        st.error("This dataset did not pass validation:")
        for e in errors:
            st.markdown(f"- {e}")
        st.stop()
    return add_derived(clean)


def p2_report(f: pd.DataFrame, s: dict) -> str:
    plat = agg_metrics(f, "platform").sort_values("roas", ascending=False)
    camp = agg_metrics(f, "campaign").sort_values("roas", ascending=False)
    lines = ["ADSPEND IQ - EXECUTIVE SUMMARY", "=" * 40, f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
             f"Source: {s['source']}  |  Rows: {s['rows']:,}", "",
             f"Spend        : {fmt_money(s['spend'])}", f"Revenue      : {fmt_money(s['revenue'])}",
             f"Conversions  : {fmt_int(s['conversions'])}", f"ROAS         : {s['roas']:.2f}x", "", "PLATFORMS (by ROAS)"]
    lines += [f"  {r.platform:12s} ROAS {r.roas:5.2f}  spend {fmt_money(r.spend)}  CPA {fmt_money(r.cpa)}" for r in plat.itertuples()]
    lines += ["", "TOP / BOTTOM CAMPAIGNS"]
    lines += [f"  best : {camp.iloc[0].campaign} (ROAS {camp.iloc[0].roas:.2f})",
              f"  worst: {camp.iloc[-1].campaign} (ROAS {camp.iloc[-1].roas:.2f})", "",
              "AUDIENCE", f"  Best age group: {s['best_age_group']}  |  best device: {s['best_device']}  |  best city: {s['best_city']}", "",
              "NOTE: ROAS here is revenue attributed in the dataset. Budget scenarios are model assumptions, not forecasts."]
    if "synthetic" in s["source"]:
        lines.append("NOTE: SYNTHETIC demo data - not real advertising results.")
    return "\n".join(lines)


def page_adspend():
    st.title("📈 Ad performance & audience insights")
    st.caption("Project 2: AdSpend IQ. Spend, revenue and ROAS across campaigns and platforms, "
               "audience breakdowns and budget what-ifs.")
    files = [p for p in find_csvs(["*.csv"], P2_DIRS) if header_has(p, {c: [] for c in REQ2})][:3]
    labels = {f"Project file: {p.name}": p for p in files}
    with st.sidebar:
        st.markdown("### Project 2 data")
        src = st.radio("Data source", list(labels) + ["Synthetic Demo Data", "Upload CSV"], key="p2_src")
        n_rows, upload = 20000, None
        if src == "Synthetic Demo Data":
            n_rows = st.select_slider("Synthetic rows", [5000, 10000, 20000, 50000, 100000], 20000, key="p2_rows")
        if src == "Upload CSV":
            upload = st.file_uploader("Ad data CSV", type="csv", key="p2_up")
    if src == "Upload CSV":
        if upload is None:
            st.info("Upload a CSV with these columns: " + ", ".join(REQ2))
            st.stop()
        df, source = prepare_adspend(safe_read(upload), "p2m_up"), f"upload: {upload.name}"
    elif src in labels:
        p = labels[src]
        df, source = prepare_adspend(read_csv_cached(str(p), p.stat().st_mtime), "p2m_file"), f"file: {p.name}"
    else:
        df, source = load_demo_adspend(n_rows), "synthetic demo"
        st.warning("Showing SYNTHETIC demo data. It does not represent real advertising results.")

    sig = f"{src}_{len(df)}_{df['date'].min():%Y%m%d}"
    with st.sidebar:
        st.markdown("### Filters")
        dmin, dmax = df["date"].min().date(), df["date"].max().date()
        dr = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax, key=f"p2_d_{sig}")
        d0, d1 = (dr if isinstance(dr, (tuple, list)) and len(dr) == 2 else (dmin, dmax))
        sel = {}
        for col, lab in [("platform", "Platform"), ("campaign", "Campaign"), ("device", "Device"), ("city", "City")]:
            opts = sorted(df[col].unique())
            sel[col] = st.multiselect(lab, opts, default=opts, key=f"p2_{col}_{sig}")
    mask = (df["date"].dt.date >= d0) & (df["date"].dt.date <= d1)
    for col, vals in sel.items():
        mask &= df[col].isin(vals)
    f = df[mask]
    if f.empty:
        st.warning("No rows match the current filters.")
        st.stop()
    summ = summarize_p2(f, source)
    st.session_state["sum_p2"] = summ
    st.caption(f"Source: {source}  |  {len(f):,} of {len(df):,} rows after filters")

    t_over, t_camp, t_aud, t_budget, t_bench, t_rep = st.tabs(
        ["Overview", "Campaigns", "Audience insights", "Budget optimizer", "Benchmarks", "Reports"])

    with t_over:
        tot = f[NUM2].sum()
        k = st.columns(6)
        k[0].metric("Spend", fmt_money(tot["spend"]))
        k[1].metric("Revenue", fmt_money(tot["revenue"]))
        k[2].metric("Conversions", fmt_int(tot["conversions"]))
        k[3].metric("ROAS", f"{summ['roas']:.2f}x")
        k[4].metric("CTR", fmt_pct(tot["clicks"] / tot["impressions"], 2))
        k[5].metric("CPA", fmt_money(tot["spend"] / tot["conversions"]) if tot["conversions"] else "n/a")
        c1, c2 = st.columns(2)
        monthly = agg_metrics(f, "month").sort_values("month")
        fig = go.Figure()
        fig.add_bar(x=monthly["month"], y=monthly["spend"], name="Spend")
        fig.add_scatter(x=monthly["month"], y=monthly["revenue"], name="Revenue", mode="lines+markers")
        fig.update_layout(title="Spend and revenue by month", legend=dict(orientation="h"))
        with c1:
            show(fig, "p2_monthly")
        plat = agg_metrics(f, "platform").sort_values("roas", ascending=False)
        with c2:
            show(px.bar(plat, x="platform", y="roas", color="platform", text_auto=".2f", title="ROAS by platform"), "p2_plat")

    with t_camp:
        camp = agg_metrics(f, "campaign").sort_values("roas", ascending=False)
        st.dataframe(camp, hide_index=True, column_config={
            "spend": st.column_config.NumberColumn(format="$%.0f"), "revenue": st.column_config.NumberColumn(format="$%.0f"),
            "ctr": st.column_config.NumberColumn(format="%.3f"), "cvr": st.column_config.NumberColumn(format="%.3f"),
            "cpc": st.column_config.NumberColumn(format="$%.2f"), "cpa": st.column_config.NumberColumn(format="$%.2f"),
            "roas": st.column_config.NumberColumn(format="%.2f")})
        c1, c2 = st.columns(2)
        choice = c1.selectbox("Funnel for", ["All campaigns"] + list(camp["campaign"]), key="p2_funnel")
        ff = f if choice == "All campaigns" else f[f["campaign"] == choice]
        funnel = go.Figure(go.Funnel(y=["Impressions", "Clicks", "Conversions"],
                                     x=[ff["impressions"].sum(), ff["clicks"].sum(), ff["conversions"].sum()]))
        funnel.update_layout(title="Conversion funnel")
        with c1:
            show(funnel, "p2_funnel_fig")
        cp = agg_metrics(f, ["campaign", "platform"]).pivot(index="campaign", columns="platform", values="roas")
        with c2:
            show(px.imshow(cp, text_auto=".2f", aspect="auto", color_continuous_scale="Blues",
                           title="ROAS: campaign x platform"), "p2_cp_heat")

    with t_aud:
        metric = st.selectbox("Metric", list(METRIC_LABELS), format_func=METRIC_LABELS.get, key="p2_metric")
        dims = [("city", "City"), ("device", "Device"), ("age_group", "Age group"), ("gender", "Gender")]
        for row in (dims[:2], dims[2:]):
            cols = st.columns(2)
            for col, (d, lab) in zip(cols, row):
                g = agg_metrics(f, d)
                g = g if d == "age_group" else g.sort_values(metric, ascending=False)
                with col:
                    show(px.bar(g, x=d, y=metric, text_auto=".2s" if metric in ("revenue", "spend", "conversions") else ".2f",
                                title=f"{METRIC_LABELS[metric]} by {lab.lower()}"), f"p2_aud_{d}")
        heat = agg_metrics(f, ["age_group", "device"]).pivot(index="age_group", columns="device", values=metric)
        show(px.imshow(heat, text_auto=".2f", aspect="auto", color_continuous_scale="Blues",
                       title=f"{METRIC_LABELS[metric]}: age group x device"), "p2_aud_heat")
        share = st.slider("Minimum spend share for a segment to be listed", 0.0, 5.0, 1.0, 0.5, key="p2_minshare") / 100
        seg = agg_metrics(f, ["age_group", "gender", "device"])
        seg = seg[seg["spend"] >= share * seg["spend"].sum()].sort_values("roas", ascending=False)
        c1, c2 = st.columns(2)
        c1.markdown("**Top segments by ROAS**")
        c1.dataframe(seg.head(8)[["age_group", "gender", "device", "spend", "revenue", "roas", "cvr"]], hide_index=True)
        c2.markdown("**Weakest segments by ROAS**")
        c2.dataframe(seg.tail(5)[["age_group", "gender", "device", "spend", "revenue", "roas", "cvr"]], hide_index=True)

    with t_budget:
        st.caption("Recommendations move spend towards higher-ROAS items within the limits you set. "
                   "Projected revenue assumes diminishing returns; it is a scenario, not a forecast.")
        c1, c2, c3, c4 = st.columns(4)
        level = c1.radio("Optimise by", ["platform", "campaign"], horizontal=True, key="p2_level")
        aggr = c2.slider("Aggressiveness", 0.0, 2.0, 1.0, 0.1, key="p2_aggr")
        swing = c3.slider("Max change per item", 0.1, 0.8, 0.4, 0.05, key="p2_swing")
        elast = c4.slider("Return elasticity", 0.3, 1.0, 0.7, 0.05, key="p2_elast")
        m = agg_metrics(f, level)
        m = m[m["spend"] > 0].reset_index(drop=True)
        m["recommended_spend"] = optimise_budget(m["spend"], m["roas"], aggr, 1 - swing, 1 + swing)
        m["change_pct"] = (m["recommended_spend"] / m["spend"] - 1) * 100
        m["projected_revenue"] = project_revenue(m["spend"], m["revenue"], m["recommended_spend"], elast)
        cur_roas = m["revenue"].sum() / m["spend"].sum()
        new_roas = m["projected_revenue"].sum() / m["recommended_spend"].sum()
        k = st.columns(3)
        k[0].metric("Current ROAS", f"{cur_roas:.2f}x")
        k[1].metric("Projected ROAS", f"{new_roas:.2f}x", f"{new_roas - cur_roas:+.2f}")
        k[2].metric("Projected revenue", fmt_money(m["projected_revenue"].sum()),
                    fmt_pct(m["projected_revenue"].sum() / m["revenue"].sum() - 1))
        long = m.melt(id_vars=level, value_vars=["spend", "recommended_spend"], var_name="plan", value_name="amount")
        show(px.bar(long, x=level, y="amount", color="plan", barmode="group", title="Current vs recommended spend"), "p2_budget")
        st.dataframe(m[[level, "spend", "recommended_spend", "change_pct", "revenue", "projected_revenue", "roas"]],
                     hide_index=True, column_config={"change_pct": st.column_config.NumberColumn(format="%.1f%%")})
        with st.expander("What-if scenario: set your own platform budgets"):
            pm = agg_metrics(f, "platform")
            pm = pm[pm["spend"] > 0].reset_index(drop=True)
            cols = st.columns(3)
            changes = [cols[i % 3].slider(f"{r.platform} spend change (%)", -50, 100, 0, 5, key=f"p2_wi_{r.platform}")
                       for i, r in enumerate(pm.itertuples())]
            new_sp = pm["spend"] * (1 + np.array(changes) / 100)
            new_rev = project_revenue(pm["spend"], pm["revenue"], new_sp, elast)
            a, b, c = st.columns(3)
            a.metric("Total spend", fmt_money(new_sp.sum()), fmt_money(new_sp.sum() - pm["spend"].sum()))
            b.metric("Projected revenue", fmt_money(new_rev.sum()), fmt_money(new_rev.sum() - pm["revenue"].sum()))
            c.metric("Projected ROAS", f"{new_rev.sum() / new_sp.sum():.2f}x",
                     f"{new_rev.sum() / new_sp.sum() - pm['revenue'].sum() / pm['spend'].sum():+.2f}")

    with t_bench:
        st.caption("Times the processing steps (generate, validate, aggregate) on synthetic data of different sizes.")
        sizes = st.multiselect("Dataset sizes (rows)", [10000, 50000, 100000, 250000, 500000],
                               default=[10000, 50000, 100000], key="p2_sizes")
        if st.button("Run benchmark", key="p2_bench_btn") and sizes:
            with st.spinner("Benchmarking..."):
                st.session_state["p2_bench"] = run_benchmark(sorted(sizes))
        bench = st.session_state.get("p2_bench")
        if bench is not None:
            show(px.line(bench, x="rows", y=["generate_s", "validate_s", "aggregate_s", "total_s"], markers=True,
                         title="Processing time by dataset size", labels={"value": "seconds"}), "p2_bench_fig")
            st.dataframe(bench.round(3), hide_index=True)
            st.caption("Timings depend on your machine; use them to compare sizes, not as absolute figures.")

    with t_rep:
        rep = p2_report(f, summ)
        st.text(rep)
        c1, c2, c3 = st.columns(3)
        c1.download_button("Executive summary (TXT)", rep, "adspend_summary.txt")
        c2.download_button("Filtered data (CSV)", csv_bytes(f.drop(columns=["age_group", "month"])), "adspend_filtered.csv", "text/csv")
        c3.download_button("Platform metrics (CSV)", csv_bytes(agg_metrics(f, "platform")), "platform_metrics.csv", "text/csv")


# ===========================================================================
# PROJECT 3 - SOCIAL AUDIENCE & CAMPAIGN INTELLIGENCE
# ===========================================================================
POSTS_REQ = {"post_id": ["id"], "platform": ["network", "channel"],
             "content_format": ["format", "content_type", "media_type", "post_type"],
             "impressions": ["reach", "views"], "likes": ["like_count", "likes_count"],
             "comments_count": ["comments", "comment_count", "num_comments"],
             "shares": ["share_count", "shares_count"]}
POSTS_OPT = {"campaign": ["campaign_name"], "post_date": ["date", "created_at", "timestamp", "posted_at"]}
COMMENTS_REQ = {"post_id": [], "comment_text": ["text", "comment", "message", "body", "content"]}

PLATFORM_FORMATS = {"Instagram": ["Reel", "Carousel", "Image", "Story"],
                    "Facebook": ["Video", "Image", "Link", "Carousel"],
                    "TikTok": ["Short Video", "Live", "Duet/Stitch", "Photo Mode"]}
FORMAT_ER = {"Reel": 1.5, "Carousel": 1.2, "Image": 0.9, "Story": 0.6, "Video": 1.1, "Link": 0.5,
             "Short Video": 1.8, "Live": 1.0, "Duet/Stitch": 1.3, "Photo Mode": 0.8}
FORMAT_REACH = {"Reel": 1.4, "Carousel": 1.0, "Image": 0.9, "Story": 0.6, "Video": 1.1, "Link": 0.5,
                "Short Video": 1.8, "Live": 0.7, "Duet/Stitch": 1.2, "Photo Mode": 0.9}
PLATFORM_ER = {"Instagram": 0.045, "Facebook": 0.020, "TikTok": 0.070}
CAMPAIGNS_3 = ["Eid Collection", "Wedding Season", "Summer Lawn", "Winter Sale", "Back to Campus", "Ramadan Edit"]
TOPICS = ["price_sensitivity", "quality_fabric", "size_availability", "style_praise", "delivery_service", "occasion_event"]
PLATFORM_TOPIC_P = {"Instagram": [.12, .15, .18, .30, .10, .15], "Facebook": [.25, .12, .13, .12, .28, .10],
                    "TikTok": [.15, .08, .12, .38, .07, .20]}
PHRASES = {
    "price_sensitivity": {
        "english": ["price is too high for this quality", "any discount or sale coming?", "too expensive, please reduce the price", "is there an Eid sale on this price?", "can you offer a discount on the price", "the price looks expensive for this", "when is the next sale or discount?"],
        "roman_urdu": ["price bohat zyada hai", "kya is par sale lagegi", "price thori kam karein", "discount kab milega price kam karein", "bohat expensive hai sale kab hogi", "sale ya discount ka price batao", "is ka price kya hai bohat expensive lag raha hai"],
        "urdu": ["قیمت بہت زیادہ ہے", "کیا اس پر سیل لگے گی", "قیمت کم کریں", "سیل اور ڈسکاؤنٹ کب ملے گا", "قیمت بہت مہنگی ہے", "ڈسکاؤنٹ دیں قیمت کم کریں"]},
    "quality_fabric": {
        "english": ["the fabric looks so soft and premium", "is it pure cotton or lawn fabric?", "stitching quality looks amazing", "fabric quality was disappointing last time", "what fabric is this, cotton or lawn?", "love the soft fabric and stitching", "the stitching and fabric quality is great"],
        "roman_urdu": ["fabric bohat acha hai", "stitching ki quality kamal hai", "fabric kaisa hai lawn hai ya cotton", "kapra thora patla lag raha hai fabric quality", "kapra soft hai stitching bohat acha", "cotton lawn ka kapra quality kaisi hai", "stitching quality kamal ki hai"],
        "urdu": ["کپڑا بہت اچھا ہے", "کوالٹی بہترین ہے", "سلائی بہت خوبصورت ہے", "کپڑے کی کوالٹی اور سلائی بہترین", "کپڑا لان ہے یا کاٹن", "کپڑا نرم اور کوالٹی اچھی ہے"]},
    "size_availability": {
        "english": ["please share the size chart", "is this available in medium?", "do you have XL in stock?", "size small available?", "what sizes are in stock?", "size chart and availability please", "is the medium size available in stock?"],
        "roman_urdu": ["size chart bhej dein", "medium size available hai?", "XL stock mein hai kya", "stock kab ayega size small", "kya sizes available hain stock mein", "size chart aur availability batao", "medium size ka stock available hai"],
        "urdu": ["سائز چارٹ بھیجیں", "کیا میڈیم سائز دستیاب ہے", "ایکس ایل سائز اسٹاک میں ہے", "سائز دستیاب ہیں کیا", "اسٹاک اور سائز چارٹ بتائیں"]},
    "style_praise": {
        "english": ["gorgeous design, love the colours", "this looks stunning on the model", "so elegant and stylish design", "beautiful collection, mashallah", "stunning colours and gorgeous style", "love this elegant collection", "beautiful design and stunning look"],
        "roman_urdu": ["bohat khoobsurat design hai", "colour combination kamal ka hai", "mashallah bohat pyara design", "kya baat hai kamal ka collection", "stunning colours mashallah khoobsurat", "elegant design bohat pyara hai", "gorgeous collection kamal ka style"],
        "urdu": ["بہت خوبصورت ہے", "ماشاءاللہ بہت پیارا ہے", "رنگ بہت اچھا ہے", "ڈیزائن اور رنگ خوبصورت ہیں", "ماشاءاللہ بہت خوبصورت کلیکشن", "پیارا ڈیزائن اور رنگ"]},
    "delivery_service": {
        "english": ["how many days will delivery take?", "ordered last week and still not received", "do you deliver to Karachi?", "customer service never replied to my DM", "my order delivery is late, no reply", "delivery to Lahore how many days?", "order not received, please reply to my DM"],
        "roman_urdu": ["delivery kitne din mein hogi", "order abhi tak nahi aaya", "Karachi delivery hoti hai?", "DM ka reply nahi mila order", "order ki delivery late hai reply karein", "delivery Lahore kitne din order", "order nahi mila customer service reply"],
        "urdu": ["ڈیلیوری کتنے دن میں ہوگی", "آرڈر ابھی تک نہیں ملا", "کراچی ڈیلیوری ہوتی ہے کیا", "آرڈر کی ڈیلیوری لیٹ ہے", "ڈیلیوری کے دن بتائیں آرڈر"]},
    "occasion_event": {
        "english": ["perfect for a wedding guest outfit", "looking for an outfit for Eid day", "need a mehndi outfit, any suggestions?", "will this work for a bridal shower outfit?", "wedding outfit ideas for Eid", "outfit for a mehndi or walima", "perfect wedding outfit for the bridal events"],
        "roman_urdu": ["shadi ke liye perfect outfit hai", "Eid ke liye outfit chahiye", "mehndi ka dress chahiye koi outfit suggestion", "walima ke liye koi outfit hai?", "wedding outfit Eid ke liye batao", "shadi aur mehndi ke liye outfit", "bridal wedding outfit chahiye"],
        "urdu": ["شادی کے لیے بہترین ہے", "عید کے لیے کچھ چاہیے", "مہندی کا جوڑا چاہیے", "شادی اور مہندی کے لیے جوڑا", "عید کے لیے شادی کا جوڑا", "ولیمہ کے لیے جوڑا بتائیں"]},
}
TOPIC_LEXICON = {
    "Price-sensitive shoppers": ["price", "sale", "discount", "expensive", "qeemat", "reduce", "cost", "قیمت", "سیل", "ڈسکاؤنٹ", "مہنگی"],
    "Quality & fabric evaluators": ["fabric", "quality", "cotton", "lawn", "stitching", "kapra", "soft", "patla", "کپڑا", "کپڑے", "کوالٹی", "سلائی", "کاٹن", "نرم"],
    "Size & availability seekers": ["size", "sizes", "chart", "stock", "available", "availability", "medium", "xl", "small", "دستیاب", "سائز", "اسٹاک"],
    "Style admirers": ["gorgeous", "stunning", "elegant", "beautiful", "design", "colour", "colours", "khoobsurat", "mashallah", "pyara", "kamal", "خوبصورت", "ماشاءاللہ", "رنگ", "ڈیزائن", "پیارا"],
    "Delivery & service concerned": ["delivery", "order", "days", "received", "service", "dm", "reply", "karachi", "lahore", "late", "ڈیلیوری", "آرڈر", "لیٹ"],
    "Occasion shoppers": ["wedding", "eid", "mehndi", "shadi", "walima", "bridal", "outfit", "شادی", "عید", "مہندی", "جوڑا", "ولیمہ"],
}
THEME_PLAYBOOK = {
    "Price-sensitive shoppers": "Lead with value: bundles, limited-time offers and clear prices in the first line of the caption.",
    "Quality & fabric evaluators": "Show fabric close-ups, state the material and stitching details, and add a short 'feel' video.",
    "Size & availability seekers": "Pin a size chart and stock updates; answer 'is it available' questions in the comments quickly.",
    "Style admirers": "Spotlight styling, colour stories and lookbooks; this audience amplifies and shares.",
    "Delivery & service concerned": "Publish delivery timelines by city and a clear support route (DM or WhatsApp) with fast replies.",
    "Occasion shoppers": "Build occasion edits (wedding, Eid, mehndi) and launch them 3-4 weeks before the event.",
    "General discussion": "Review a sample of these comments before deciding on a message.",
}
RU_MARKERS = {"hai", "hain", "kya", "bohat", "bahut", "boht", "nahi", "nahin", "kab", "kitne", "kitna", "dein",
              "karein", "mein", "ka", "ki", "ke", "ko", "aur", "acha", "achi", "kapra", "qeemat", "shadi", "mil",
              "jayega", "chahiye", "abhi", "tak", "kamal", "pyara", "khoobsurat", "bhej", "ayega", "aaya", "mila",
              "koi", "thori", "lagegi", "hogi", "hoti", "kaisa", "lag", "raha", "walima", "mehndi"}
RU_NORM = {"bahut": "bohat", "boht": "bohat", "bohot": "bohat", "kapda": "kapra", "keemat": "qeemat",
           "kimat": "qeemat", "nahin": "nahi", "nai": "nahi", "kitnay": "kitne", "kitney": "kitne",
           "achha": "acha", "achhi": "acha", "achi": "acha", "khubsurat": "khoobsurat", "khobsurat": "khoobsurat",
           "shaadi": "shadi"}
RU_STOP = {"hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "kya", "aur", "bhi", "ho", "to", "ye", "yeh",
           "wo", "woh", "tha", "thi", "thay", "kar", "koi", "raha", "rahi"}
UR_STOP = {"ہے", "ہیں", "کا", "کی", "کے", "کو", "سے", "میں", "کیا", "اور", "بھی", "ہو", "تو", "یہ", "وہ", "تھا", "تھی"}
STOP_ALL = (set(ENGLISH_STOP_WORDS) - {"not", "no", "never"}) | RU_STOP | UR_STOP
URL_RE = re.compile(r"https?://\S+|www\.\S+")
URDU_RE = re.compile(r"[\u0600-\u06FF]")
DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")
ARABIC_MAP = str.maketrans({"ي": "ی", "ك": "ک", "ۂ": "ہ"})
AD_FEATURES = ["engagement_rate", "like_rate", "share_rate", "comment_rate", "share_to_like", "comment_to_like", "log_impressions"]
RATE_FEATURES = ["engagement_rate", "like_rate", "share_rate", "comment_rate"]


def detect_language(text: str) -> str:
    if URDU_RE.search(text):
        return "urdu"
    toks = set(re.findall(r"[a-z]+", text.lower()))
    return "roman_urdu" if toks & RU_MARKERS else "english"


def clean_text(text: str) -> str:
    t = str(text).lower()
    t = URL_RE.sub(" ", t)
    t = re.sub(r"@\w+", " ", t).replace("#", " ")
    t = DIACRITICS_RE.sub("", t).translate(ARABIC_MAP)
    t = re.sub(r"[^\w\s]|_", " ", t)  # drops punctuation and emoji, keeps Urdu letters
    t = re.sub(r"\d+", " ", t)
    return " ".join(RU_NORM.get(w, w) for w in t.split())


def _gen_posts(n_posts: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    platform = rng.choice(list(PLATFORM_FORMATS), n_posts, p=[0.40, 0.30, 0.30])
    fmt = np.array([rng.choice(PLATFORM_FORMATS[p]) for p in platform])
    followers = rng.lognormal(10.6, 0.7, n_posts)
    impressions = (followers * rng.uniform(0.15, 0.9, n_posts) * pd.Series(fmt).map(FORMAT_REACH).to_numpy()).astype(int) + 500
    er = (pd.Series(platform).map(PLATFORM_ER).to_numpy() * pd.Series(fmt).map(FORMAT_ER).to_numpy()
          * rng.lognormal(0, 0.3, n_posts))
    eng = impressions * er
    likes, shares, comments = (rng.poisson(eng * w).astype(float) for w in (0.80, 0.07, 0.13))
    kind = np.array(["none"] * n_posts, dtype=object)
    idx = rng.choice(n_posts, max(2, int(n_posts * 0.03)), replace=False)
    spikes, drops = idx[: len(idx) // 2], idx[len(idx) // 2:]
    m = rng.uniform(4, 9, len(spikes))
    likes[spikes] *= m
    shares[spikes] *= m * 2.5
    comments[spikes] *= m * 0.6
    d = rng.uniform(0.05, 0.2, len(drops))
    likes[drops] *= d
    shares[drops] *= d
    comments[drops] *= d
    kind[spikes], kind[drops] = "spike", "drop"
    return pd.DataFrame({
        "post_id": [f"P{i:05d}" for i in range(n_posts)], "platform": platform, "content_format": fmt,
        "campaign": rng.choice(CAMPAIGNS_3, n_posts),
        "post_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n_posts), unit="D"),
        "impressions": impressions,
        "likes": np.minimum(likes, impressions * 0.35).round().astype(int),
        "comments_count": np.minimum(comments, impressions * 0.05).round().astype(int),
        "shares": np.minimum(shares, impressions * 0.08).round().astype(int),
        "synthetic_anomaly": kind})


def _gen_comments(posts: pd.DataFrame, n_comments: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    w = posts["comments_count"].to_numpy() + 1.0
    pidx = rng.choice(len(posts), n_comments, p=w / w.sum())
    suffixes = ["", "", "", " 😍", " 🔥", " pls", " !!", " ??", " #fashion", " @brand"]
    rows = []
    for i, pi in enumerate(pidx):
        topic = str(rng.choice(TOPICS, p=PLATFORM_TOPIC_P[posts["platform"].iat[pi]]))
        lang = str(rng.choice(["english", "roman_urdu", "urdu"], p=[0.45, 0.40, 0.15]))
        text = str(rng.choice(PHRASES[topic][lang]))
        if rng.random() < 0.15:  # noisy comment that touches a second topic
            text += (" ، " if lang == "urdu" else ", ") + str(rng.choice(PHRASES[str(rng.choice(TOPICS))][lang]))
        if lang == "roman_urdu" and rng.random() < 0.3:
            text = text.replace("bohat", "bahut")
        text += str(rng.choice(suffixes))
        if rng.random() < 0.01:
            text += " https://bit.ly/xyz"
        rows.append((f"C{i:05d}", posts["post_id"].iat[pi], text, topic))
    return pd.DataFrame(rows, columns=["comment_id", "post_id", "comment_text", "synthetic_topic"])


@st.cache_data(show_spinner="Generating synthetic posts...")
def gen_posts(n: int) -> pd.DataFrame:
    return _gen_posts(n)


@st.cache_data(show_spinner="Generating synthetic comments...")
def gen_comments(posts: pd.DataFrame, n: int) -> pd.DataFrame:
    return _gen_comments(posts, n)


def prep_posts(raw: pd.DataFrame, quality: dict) -> pd.DataFrame:
    p = raw.copy()
    quality["posts_raw"] = len(p)
    p["post_id"] = p["post_id"].astype(str)
    for c in ["impressions", "likes", "comments_count", "shares"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.dropna(subset=["impressions", "likes", "comments_count", "shares"]).drop_duplicates("post_id")
    p = p[(p["impressions"] > 0) & (p[["likes", "comments_count", "shares"]] >= 0).all(axis=1)].copy()
    quality["posts_valid"] = len(p)
    if "campaign" not in p.columns:
        p["campaign"] = "Unspecified"
    p["campaign"] = p["campaign"].astype(str)
    p["post_date"] = pd.to_datetime(p["post_date"], errors="coerce") if "post_date" in p.columns else pd.NaT
    p["month"] = p["post_date"].dt.to_period("M").astype(str).where(p["post_date"].notna())
    p["engagements"] = p["likes"] + p["comments_count"] + p["shares"]
    p["engagement_rate"] = p["engagements"] / p["impressions"]
    p["like_rate"], p["share_rate"] = p["likes"] / p["impressions"], p["shares"] / p["impressions"]
    p["comment_rate"] = p["comments_count"] / p["impressions"]
    p["share_to_like"], p["comment_to_like"] = p["shares"] / (p["likes"] + 1), p["comments_count"] / (p["likes"] + 1)
    p["log_impressions"] = np.log1p(p["impressions"])
    return p.reset_index(drop=True)


def prep_comments(raw: pd.DataFrame, posts: pd.DataFrame, quality: dict) -> pd.DataFrame:
    c = raw.copy()
    quality["comments_raw"] = len(c)
    c["post_id"], c["comment_text"] = c["post_id"].astype(str), c["comment_text"].astype(str)
    c = c[c["post_id"].isin(set(posts["post_id"]))].copy()
    quality["comments_orphans_dropped"] = quality["comments_raw"] - len(c)
    c["language"] = c["comment_text"].map(detect_language)
    c["clean_text"] = c["comment_text"].map(clean_text)
    c = c[c["clean_text"].str.split().str.len().fillna(0) >= 1]
    quality["comments_clean"] = len(c)
    return c.reset_index(drop=True)


def detect_anomalies(p: pd.DataFrame, contamination: float) -> pd.DataFrame:
    p = p.copy()
    X = p[AD_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).copy()
    X[RATE_FEATURES] = np.log1p(X[RATE_FEATURES] * 100)
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=42).fit(StandardScaler().fit_transform(X))
    Xs = StandardScaler().fit_transform(X)
    p["anomaly_score"] = -iso.score_samples(Xs)
    p["is_anomaly"] = iso.predict(Xs) == -1
    peer = p.groupby(["platform", "content_format"])["engagement_rate"].transform("median")
    p["er_vs_peer"] = p["engagement_rate"] / peer.replace(0, np.nan)
    p["direction"] = np.where(p["er_vs_peer"].fillna(1) >= 1, "High engagement", "Low engagement")
    return p


def _vectorize(clean_texts):
    n = len(clean_texts)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3 if n >= 300 else 1, max_df=0.6, max_features=4000,
                          sublinear_tf=True, stop_words=sorted(STOP_ALL))
    return vec, vec.fit_transform(clean_texts)


def _embed(X, seed: int = 42):
    """LSA: SVD on TF-IDF, then L2-normalise. Stops K-Means from piling short, sparse comments into one cluster."""
    n_comp = int(min(50, X.shape[1] - 1, X.shape[0] - 1))
    if n_comp < 2:
        return Normalizer().fit_transform(X.toarray())
    return make_pipeline(TruncatedSVD(n_comp, random_state=seed), Normalizer(copy=False)).fit_transform(X)


@st.cache_data(show_spinner="Training TF-IDF + LSA + K-Means...")
def run_clustering(clean_texts: tuple, k: int, seed: int = 42) -> dict:
    vec, X = _vectorize(clean_texts)
    Z = _embed(X, seed)
    k = int(min(k, max(2, X.shape[0] - 1)))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    terms = vec.get_feature_names_out()
    top_terms = {}
    for cid in range(k):  # label clusters with the most characteristic TF-IDF terms
        mean = np.asarray(X[km.labels_ == cid].mean(axis=0)).ravel()
        top_terms[cid] = [terms[i] for i in mean.argsort()[::-1][:12]]
    try:
        sil = float(silhouette_score(Z, km.labels_, metric="cosine", sample_size=min(2000, Z.shape[0]), random_state=seed))
    except Exception:
        sil = float("nan")
    coords = TruncatedSVD(2, random_state=seed).fit_transform(X)
    return {"labels": km.labels_, "top_terms": top_terms, "silhouette": sil, "inertia": float(km.inertia_), "coords": coords}


@st.cache_data(show_spinner="Scoring K values...")
def k_diagnostics(clean_texts: tuple) -> pd.DataFrame:
    _, X = _vectorize(clean_texts)
    Z = _embed(X)
    rows = []
    for k in range(3, 11):
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(Z)
        sil = silhouette_score(Z, km.labels_, metric="cosine", sample_size=min(2000, Z.shape[0]), random_state=42)
        rows.append((k, km.inertia_, sil))
    return pd.DataFrame(rows, columns=["k", "inertia", "silhouette"])


def base_theme(name: str) -> str:
    return re.sub(r"\s*\(\d+\)$", "", name)


def name_clusters(top_terms: dict) -> dict:
    """Name a cluster after the theme its top terms point to. Weak or ambiguous evidence stays unnamed."""
    names, used = {}, Counter()
    for cid, terms in top_terms.items():
        scores = {}
        for theme, words in TOPIC_LEXICON.items():
            scores[theme] = sum(12 - rank for rank, term in enumerate(terms[:12]) if any(w in term.split() for w in words))
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        (best, s1), (_, s2) = ranked[0], ranked[1]
        label = best if s1 >= 8 and s1 > s2 * 1.25 else "General discussion"
        used[label] += 1
        names[cid] = label if used[label] == 1 else f"{label} ({used[label]})"
    return names


@st.cache_data(show_spinner="Running the Project 3 pipeline (clean, features, K-Means, Isolation Forest)...")
def build_p3(posts_raw: pd.DataFrame, comments_raw: pd.DataFrame, k: int, contamination: float) -> dict:
    quality: dict = {}
    p = detect_anomalies(prep_posts(posts_raw, quality), contamination)
    c = prep_comments(comments_raw, p, quality)
    cl = run_clustering(tuple(c["clean_text"]), k)
    c["cluster"] = cl["labels"]
    names = name_clusters(cl["top_terms"])
    c["segment"] = c["cluster"].map(names)
    c["x"], c["y"] = cl["coords"][:, 0], cl["coords"][:, 1]
    c = c.merge(p[["post_id", "platform", "content_format", "campaign", "engagement_rate"]], on="post_id", how="left")
    rows = []
    for seg, g in c.groupby("segment"):
        cid = int(g["cluster"].iloc[0])
        plat = g["platform"].dropna()
        rows.append({"segment": seg, "cluster": cid, "comments": len(g), "share": len(g) / len(c),
                     "top_terms": ", ".join(cl["top_terms"][cid][:8]), "dominant_language": g["language"].mode().iat[0],
                     "dominant_platform": plat.mode().iat[0] if len(plat) else "n/a",
                     "playbook": THEME_PLAYBOOK.get(base_theme(seg), THEME_PLAYBOOK["General discussion"])})
    seg_table = pd.DataFrame(rows).sort_values("comments", ascending=False).reset_index(drop=True)
    ev = {"silhouette": cl["silhouette"], "inertia": cl["inertia"], "k": len(cl["top_terms"])}
    if "synthetic_topic" in c.columns:
        ev["ari"] = float(adjusted_rand_score(c["synthetic_topic"], c["cluster"]))
        ev["nmi"] = float(normalized_mutual_info_score(c["synthetic_topic"], c["cluster"]))
    return {"posts": p, "comments": c, "segments": seg_table, "quality": quality, "eval": ev}


def p3_insights(ctx: dict) -> dict:
    p, c, seg = ctx["posts"], ctx["comments"], ctx["segments"]
    g = p.groupby(["platform", "content_format"]).agg(posts=("post_id", "count"), median_er=("engagement_rate", "median")).reset_index()
    g_ok = g[g["posts"] >= 10]
    g = g_ok if len(g_ok) else g
    best = g.sort_values("median_er", ascending=False).groupby("platform").head(1).sort_values("median_er", ascending=False)
    camp = p.groupby("campaign").agg(posts=("post_id", "count"), median_er=("engagement_rate", "median")).reset_index()
    camp = camp.sort_values("median_er", ascending=False)
    lifts = pd.DataFrame(columns=["platform", "segment", "lift"])
    if c["platform"].notna().any():
        ct = pd.crosstab(c["platform"], c["segment"], normalize="index")
        lift = ct.div(c["segment"].value_counts(normalize=True), axis=1)
        lifts = lift.stack().reset_index()
        lifts.columns = ["platform", "segment", "lift"]
        lifts = lifts[lifts["lift"] >= 1.15].sort_values("lift", ascending=False).head(6)
    spikes = p[p["is_anomaly"] & (p["direction"] == "High engagement")].nlargest(5, "anomaly_score")
    drops = p[p["is_anomaly"] & (p["direction"] == "Low engagement")].nlargest(5, "anomaly_score")
    return {"best_formats": best, "campaigns": camp, "lifts": lifts, "spikes": spikes, "drops": drops,
            "top_segments": seg.head(3)}


def summarize_p3(ctx: dict) -> dict:
    ins = p3_insights(ctx)
    best = ins["best_formats"].iloc[0] if len(ins["best_formats"]) else None
    top = ctx["segments"].iloc[0]
    return {"source": ctx["source"], "posts": int(len(ctx["posts"])), "comments": int(len(ctx["comments"])),
            "top_segment": str(top["segment"]), "top_segment_share": float(top["share"]),
            "best_format": None if best is None else f"{best['content_format']} on {best['platform']}",
            "best_format_er": None if best is None else float(best["median_er"]),
            "anomaly_share": float(ctx["posts"]["is_anomaly"].mean()), "silhouette": ctx["eval"]["silhouette"]}


def default_p3_summary():
    posts = gen_posts(2400)
    comments = gen_comments(posts, 5000)
    ctx = build_p3(posts, comments, 6, 0.03)
    ctx = {**ctx, "source": "synthetic demo"}
    return summarize_p3(ctx)


def p3_report(ctx: dict) -> str:
    ins, ev = p3_insights(ctx), ctx["eval"]
    lines = ["SOCIAL AUDIENCE & CAMPAIGN INTELLIGENCE - SUMMARY", "=" * 52, f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
             f"Source: {ctx['source']}", f"Posts: {len(ctx['posts']):,}  |  Comments: {len(ctx['comments']):,}  |  Clusters: {ev['k']}",
             f"Silhouette (cosine): {ev['silhouette']:.3f}", "", "WHERE TO POST (best median engagement rate per platform)"]
    lines += [f"  - {r.platform}: {r.content_format} ({r.median_er * 100:.2f}% median ER, {r.posts} posts)" for r in ins["best_formats"].itertuples()]
    lines += ["", "WHO TO TALK TO"]
    for r in ins["top_segments"].itertuples():
        lines += [f"  - {r.segment} ({r.share * 100:.0f}% of comments): {r.playbook}"]
    lines += ["", "CAMPAIGNS (median ER)"] + [f"  - {r.campaign}: {r.median_er * 100:.2f}%" for r in ins["campaigns"].head(3).itertuples()]
    lines += ["", f"UNUSUAL POSTS: {int(ctx['posts']['is_anomaly'].sum())} flagged "
              f"({ctx['posts']['is_anomaly'].mean() * 100:.1f}%). Flags are prompts for review, not proof of manipulation or fraud.",
              "", "LIMITATIONS", "  - Clusters are exploratory and need human interpretation.",
              "  - Engagement signals cannot establish advertising ROAS."]
    if "synthetic" in ctx["source"]:
        lines.append("  - SYNTHETIC data: results do not measure real brands and are not evidence of real-world performance.")
    return "\n".join(lines)


def p3_context() -> dict:
    posts_ok = [p for p in find_csvs(["*post*.csv"], P3_DIRS, recursive=True) if header_has(p, POSTS_REQ)]
    comm_ok = [p for p in find_csvs(["*comment*.csv"], P3_DIRS, recursive=True) if header_has(p, COMMENTS_REQ)]
    rel = lambda p: str(p.relative_to(BASE_DIR)) if BASE_DIR in p.parents else str(p)  # noqa: E731
    with st.sidebar:
        st.markdown("### Project 3 data & models")
        opts = ["Built-in synthetic generator"] + (["Project 3 pipeline files"] if posts_ok and comm_ok else []) + ["Upload CSVs"]
        src = st.radio("Data source", opts, key="p3_src")
        n_posts, n_comments = 2400, 5000
        if src == "Built-in synthetic generator":
            n_posts = st.slider("Synthetic posts", 600, 5000, 2400, 200, key="p3_np")
            n_comments = st.slider("Synthetic comments", 1000, 10000, 5000, 500, key="p3_nc")
        pf = cf = None
        if src == "Project 3 pipeline files":
            pf = st.selectbox("Posts file", posts_ok, format_func=rel, key="p3_pf")
            cf = st.selectbox("Comments file", comm_ok, format_func=rel, key="p3_cf")
        up_p = up_c = None
        if src == "Upload CSVs":
            up_p = st.file_uploader("Posts CSV", type="csv", key="p3_upp")
            up_c = st.file_uploader("Comments CSV", type="csv", key="p3_upc")
        k = st.slider("Audience clusters (K)", 3, 10, 6, key="p3_k")
        contamination = st.slider("Expected anomaly share", 0.01, 0.10, 0.03, 0.01, key="p3_cont")

    if src == "Built-in synthetic generator":
        posts_raw = gen_posts(n_posts)
        comments_raw, source = gen_comments(posts_raw, n_comments), "synthetic demo"
        st.warning("Showing SYNTHETIC posts and comments. Results do not describe real brands or audiences.")
    elif src == "Project 3 pipeline files":
        posts_raw, comments_raw = read_csv_cached(str(pf), pf.stat().st_mtime), read_csv_cached(str(cf), cf.stat().st_mtime)
        source = f"files: {pf.name} + {cf.name}"
    else:
        if up_p is None or up_c is None:
            st.info("Upload a posts CSV and a comments CSV in the sidebar (required columns: "
                    + ", ".join(POSTS_REQ) + " | " + ", ".join(COMMENTS_REQ) + ").")
            st.stop()
        posts_raw, comments_raw, source = safe_read(up_p), safe_read(up_c), "upload"
    posts_m = map_columns(posts_raw, POSTS_REQ, "p3m_posts", POSTS_OPT)
    comments_m = map_columns(comments_raw, COMMENTS_REQ, "p3m_comments", {"comment_id": ["id"]})
    if posts_m is None or comments_m is None:
        st.stop()
    if len(comments_m) < 30 or len(posts_m) < 30:
        st.error("Need at least 30 posts and 30 comments to run the pipeline.")
        st.stop()
    ctx = build_p3(posts_m, comments_m, k, contamination)
    if len(ctx["comments"]) < 30:
        st.error("Too few comments link to the posts provided (check the post_id values).")
        st.stop()
    ctx = {**ctx, "source": source, "synthetic": source == "synthetic demo"}
    st.session_state["sum_p3"] = summarize_p3(ctx)
    return ctx


def p3_quality_note(ctx):
    q = ctx["quality"]
    st.caption(f"Source: {ctx['source']}  |  Posts {q['posts_valid']:,}/{q['posts_raw']:,} valid  |  Comments "
               f"{q['comments_clean']:,}/{q['comments_raw']:,} kept ({q['comments_orphans_dropped']:,} without a matching post)")


def page_p3_clusters():
    ctx = p3_context()
    st.title("🧠 Audience clustering")
    st.caption("TF-IDF + LSA + K-Means over cleaned English, Roman Urdu and Urdu comments. Clusters are exploratory and need interpretation.")
    p3_quality_note(ctx)
    c, seg, ev = ctx["comments"], ctx["segments"], ctx["eval"]
    k = st.columns(4)
    k[0].metric("Comments clustered", fmt_int(len(c)))
    k[1].metric("Clusters", ev["k"])
    k[2].metric("Silhouette (cosine)", f"{ev['silhouette']:.3f}")
    k[3].metric("Largest segment", f"{seg.iloc[0]['share'] * 100:.0f}%", seg.iloc[0]["segment"], delta_color="off")
    c1, c2 = st.columns([3, 2])
    samp = c.sample(min(2500, len(c)), random_state=1)
    with c1:
        show(px.scatter(samp, x="x", y="y", color="segment", opacity=0.6, hover_data=["comment_text"],
                        title="Comment map (2-D projection of TF-IDF)"), "p3_map")
    with c2:
        show(px.bar(seg.sort_values("comments"), x="comments", y="segment", orientation="h", title="Segment sizes"), "p3_sizes")
    st.dataframe(seg[["segment", "comments", "share", "top_terms", "dominant_language", "dominant_platform", "playbook"]],
                 hide_index=True, column_config={"share": st.column_config.NumberColumn(format="percent")})
    c3, c4 = st.columns(2)
    lang = pd.crosstab(c["segment"], c["language"], normalize="index").reset_index().melt(id_vars="segment", value_name="share")
    with c3:
        show(px.bar(lang, x="segment", y="share", color="language", title="Language mix by segment"), "p3_lang")
    ct = pd.crosstab(c["platform"], c["segment"], normalize="index")
    lift = ct.div(c["segment"].value_counts(normalize=True), axis=1)
    with c4:
        show(px.imshow(lift, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu", color_continuous_midpoint=1,
                       title="Over-indexing: segment share on platform vs overall (1.0 = average)"), "p3_lift")
    pick = st.selectbox("Read comments from a segment", list(seg["segment"]), key="p3_pick")
    st.dataframe(c[c["segment"] == pick][["comment_text", "language", "platform"]].sample(
        min(10, int((c["segment"] == pick).sum())), random_state=2), hide_index=True)
    with st.expander("Choosing K (silhouette and inertia)"):
        if st.checkbox("Compute diagnostics for K = 3 to 10", key="p3_kdiag"):
            kd = k_diagnostics(tuple(c["clean_text"]))
            a, b = st.columns(2)
            with a:
                show(px.line(kd, x="k", y="silhouette", markers=True, title="Silhouette (higher is better)"), "p3_kd1")
            with b:
                show(px.line(kd, x="k", y="inertia", markers=True, title="Inertia (look for the elbow)"), "p3_kd2")
    if "ari" in ev:
        with st.expander("Check against the synthetic ground truth"):
            a, b = st.columns(2)
            a.metric("Adjusted Rand index", f"{ev['ari']:.3f}")
            b.metric("Normalised mutual information", f"{ev['nmi']:.3f}")
            show(px.imshow(pd.crosstab(c["synthetic_topic"], c["segment"]), text_auto=True, aspect="auto",
                           title="Synthetic topic vs discovered segment"), "p3_truth")
            st.caption("The comments are generated from templates, so high scores here show the pipeline works on "
                       "synthetic data. They are not evidence of performance on real, unseen campaigns.")


def page_p3_content():
    ctx = p3_context()
    st.title("🎬 Content intelligence")
    st.caption("Which formats, platforms and campaigns earn the strongest engagement, and which audience segments comment on them.")
    p3_quality_note(ctx)
    p, c = ctx["posts"], ctx["comments"]
    k = st.columns(4)
    k[0].metric("Posts", fmt_int(len(p)))
    k[1].metric("Median engagement rate", fmt_pct(p["engagement_rate"].median(), 2))
    best = p3_insights(ctx)["best_formats"]
    if len(best):
        k[2].metric("Best format", f"{best.iloc[0]['content_format']}", f"{best.iloc[0]['platform']}", delta_color="off")
    k[3].metric("Total engagements", fmt_int(p["engagements"].sum()))
    piv = p.pivot_table(index="platform", columns="content_format", values="engagement_rate", aggfunc="median") * 100
    c1, c2 = st.columns(2)
    with c1:
        show(px.imshow(piv, text_auto=".2f", aspect="auto", color_continuous_scale="Blues",
                       title="Median engagement rate (%): platform x format"), "p3_ci_heat")
    with c2:
        show(px.box(p, x="content_format", y="engagement_rate", color="platform", points=False,
                    title="Engagement rate distribution by format"), "p3_ci_box")
    camp = p.groupby("campaign").agg(posts=("post_id", "count"), median_er=("engagement_rate", "median"),
                                     engagements=("engagements", "sum")).reset_index().sort_values("median_er", ascending=False)
    c3, c4 = st.columns(2)
    with c3:
        show(px.bar(camp, x="campaign", y="median_er", text_auto=".3f", title="Median engagement rate by campaign"), "p3_ci_camp")
    with c4:
        if p["month"].notna().any():
            tr = p.dropna(subset=["month"]).groupby(["month", "platform"])["engagement_rate"].median().reset_index().sort_values("month")
            show(px.line(tr, x="month", y="engagement_rate", color="platform", markers=True, title="Median engagement rate by month"), "p3_ci_trend")
        else:
            st.info("No post dates in this dataset, so the monthly trend is hidden.")
    mix = pd.crosstab(c["content_format"], c["segment"], normalize="index").mul(100).reset_index().melt(
        id_vars="content_format", var_name="segment", value_name="percent")
    show(px.bar(mix, x="content_format", y="percent", color="segment", title="Which audience segments comment on each format (%)"), "p3_ci_mix")
    st.markdown("**Top posts by engagement rate** (posts with at least median reach)")
    top = p[p["impressions"] >= p["impressions"].median()].nlargest(15, "engagement_rate")
    st.dataframe(top[["post_id", "platform", "content_format", "campaign", "impressions", "likes", "comments_count", "shares", "engagement_rate"]],
                 hide_index=True, column_config={"engagement_rate": st.column_config.NumberColumn(format="%.3f")})


def page_p3_anomaly():
    ctx = p3_context()
    st.title("🚨 Anomaly detection")
    st.caption("Isolation Forest over engagement ratios. A flag marks an unusual post for review. It is not proof of manipulation or fraud.")
    p3_quality_note(ctx)
    p = ctx["posts"]
    flagged = p[p["is_anomaly"]]
    k = st.columns(4)
    k[0].metric("Posts flagged", fmt_int(len(flagged)), fmt_pct(len(flagged) / len(p)), delta_color="off")
    k[1].metric("Unusually high engagement", fmt_int((flagged["direction"] == "High engagement").sum()))
    k[2].metric("Unusually low engagement", fmt_int((flagged["direction"] == "Low engagement").sum()))
    k[3].metric("Posts analysed", fmt_int(len(p)))
    c1, c2 = st.columns(2)
    with c1:
        show(px.scatter(p, x="impressions", y="engagement_rate", color="is_anomaly", log_x=True, log_y=True, opacity=0.6,
                        color_discrete_map={True: "#D64545", False: "#8A94A6"}, hover_data=["post_id", "platform", "content_format"],
                        title="Impressions vs engagement rate (log scales)"), "p3_an_scatter")
    with c2:
        show(px.histogram(p, x="anomaly_score", color="is_anomaly", nbins=50, title="Anomaly score distribution",
                          color_discrete_map={True: "#D64545", False: "#8A94A6"}), "p3_an_hist")
    g = p.groupby(["platform", "content_format"]).agg(posts=("post_id", "count"), flagged=("is_anomaly", "sum")).reset_index()
    g["flag_rate"] = g["flagged"] / g["posts"]
    show(px.bar(g, x="content_format", y="flag_rate", color="platform", barmode="group", title="Share of posts flagged by format and platform"), "p3_an_rate")
    st.markdown("**Flagged posts (most unusual first)**")
    st.dataframe(flagged.sort_values("anomaly_score", ascending=False)[
        ["post_id", "platform", "content_format", "campaign", "impressions", "engagement_rate", "er_vs_peer", "direction", "anomaly_score"]].head(200),
        hide_index=True, column_config={"engagement_rate": st.column_config.NumberColumn(format="%.3f"),
                                        "er_vs_peer": st.column_config.NumberColumn("x peer median", format="%.2f"),
                                        "anomaly_score": st.column_config.NumberColumn(format="%.3f")})
    st.download_button("Download flagged posts (CSV)", csv_bytes(flagged), "flagged_posts.csv", "text/csv")
    if "synthetic_anomaly" in p.columns:
        truth = p["synthetic_anomaly"] != "none"
        tp = int((truth & p["is_anomaly"]).sum())
        with st.expander("Check against the injected synthetic anomalies"):
            a, b = st.columns(2)
            a.metric("Precision", fmt_pct(tp / max(1, int(p["is_anomaly"].sum()))))
            b.metric("Recall", fmt_pct(tp / max(1, int(truth.sum()))))
            st.caption("Anomalies were injected into the synthetic data, so this only shows the detector works on that "
                       "pattern. It says nothing about real-world fraud detection.")


def page_p3_marketing():
    ctx = p3_context()
    st.title("🎯 Marketing opportunities")
    st.caption("Recommendations drawn from the clustering, content and anomaly results.")
    p3_quality_note(ctx)
    ins = p3_insights(ctx)
    with st.container(border=True):
        st.subheader("Where to post")
        for r in ins["best_formats"].itertuples():
            st.markdown(f"- **{r.platform}**: {r.content_format} has the highest median engagement rate "
                        f"({r.median_er * 100:.2f}%, {r.posts} posts).")
        camp = ins["campaigns"]
        if len(camp):
            st.markdown(f"- Strongest campaign: **{camp.iloc[0]['campaign']}** ({camp.iloc[0]['median_er'] * 100:.2f}% median). "
                        f"Weakest: **{camp.iloc[-1]['campaign']}** ({camp.iloc[-1]['median_er'] * 100:.2f}%).")
    with st.container(border=True):
        st.subheader("Who to talk to, and what to say")
        for r in ins["top_segments"].itertuples():
            st.markdown(f"- **{r.segment}** ({r.share * 100:.0f}% of comments, mostly {r.dominant_language.replace('_', ' ')}): {r.playbook}")
        for r in ins["lifts"].itertuples():
            st.markdown(f"- *{r.segment}* over-indexes on **{r.platform}** ({r.lift:.2f}x the overall share).")
    cols = st.columns(2)
    with cols[0].container(border=True):
        st.subheader("Posts to learn from")
        st.caption("Unusually high engagement: study and reuse what worked.")
        st.dataframe(ins["spikes"][["post_id", "platform", "content_format", "campaign", "engagement_rate"]], hide_index=True)
    with cols[1].container(border=True):
        st.subheader("Posts to review")
        st.caption("Unusually low engagement: check targeting, timing or delivery issues.")
        st.dataframe(ins["drops"][["post_id", "platform", "content_format", "campaign", "engagement_rate"]], hide_index=True)
    report = p3_report(ctx)
    with st.expander("Full text report"):
        st.text(report)
    st.download_button("Download report (TXT)", report, "social_intelligence_summary.txt")
    st.info("Limitations: clusters are exploratory; anomaly flags are not proof of manipulation; engagement signals "
            "cannot establish advertising ROAS; synthetic data does not measure real brands.")


# ===========================================================================
# HOME & INTEGRATED INSIGHTS
# ===========================================================================
def get_summary(key: str):
    if key not in st.session_state:
        fn = {"sum_p1": default_p1_summary, "sum_p2": default_p2_summary, "sum_p3": default_p3_summary}[key]
        with st.spinner("Preparing default data..."):
            st.session_state[key] = fn()
    return st.session_state[key]


def page_home():
    st.title("📊 Big Data Analytics Suite")
    st.markdown("Three projects, one app. Use the sidebar to move between them. Each project works on built-in "
                "synthetic demo data, and switches to your own files when it finds them or when you upload them.")
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.subheader("💬 Sentiment analysis")
        st.write("Project 1. Public SHEIN-related YouTube comments classified as positive, negative, neutral or mixed, "
                 "with negative themes and a marketing report.")
    with c2.container(border=True):
        st.subheader("📈 Ad performance & audience insights")
        st.write("Project 2. AdSpend IQ: spend, revenue and ROAS by campaign and platform, audience breakdowns and "
                 "budget what-ifs.")
    with c3.container(border=True):
        st.subheader("🧠 Social audience intelligence")
        st.write("Project 3. Audience clustering, content performance, anomaly detection and marketing opportunities "
                 "for Pakistani fashion campaigns.")
    st.markdown("#### Real data found in this folder")
    p1 = find_csvs(P1_FILES, P1_DIRS)
    p2 = [p for p in find_csvs(["*.csv"], P2_DIRS) if header_has(p, {c: [] for c in REQ2})]
    p3p = [p for p in find_csvs(["*post*.csv"], P3_DIRS, recursive=True) if header_has(p, POSTS_REQ)]
    p3c = [p for p in find_csvs(["*comment*.csv"], P3_DIRS, recursive=True) if header_has(p, COMMENTS_REQ)]
    status = pd.DataFrame([
        {"Project": "1 Sentiment", "Found": ", ".join(p.name for p in p1[:3]) or "none (demo data in use)"},
        {"Project": "2 AdSpend IQ", "Found": ", ".join(p.name for p in p2[:3]) or "none (demo data in use)"},
        {"Project": "3 Social intelligence", "Found": (f"{p3p[0].name} + {p3c[0].name}" if p3p and p3c else "none (demo data in use)")}])
    st.dataframe(status, hide_index=True)
    st.caption("Drop your Project 1 outputs (for example review_comments_analyzed.csv) into this folder or an outputs/ "
               "folder, Project 2 CSVs into data/, and Project 3 posts and comments CSVs into project3/ or data/. "
               "You can also upload any file from the sidebar of each project.")
    st.warning("Demo data is synthetic. Do not present demo numbers as real findings.")


def page_integrated():
    st.title("🧩 Integrated insights")
    st.caption("A side-by-side summary of the three projects. The projects use different datasets with no shared key, "
               "so this page combines their conclusions, not their rows. Treat the links as hypotheses to test.")
    s1, s2, s3 = get_summary("sum_p1"), get_summary("sum_p2"), get_summary("sum_p3")
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.subheader("💬 Sentiment")
        st.caption(f"Source: {s1['source']}")
        st.metric("Comments", fmt_int(s1["comments"]))
        st.write(f"Positive {fmt_pct(s1['share']['positive'], 0)}  |  Negative {fmt_pct(s1['share']['negative'], 0)}  |  "
                 f"Mixed {fmt_pct(s1['share']['mixed'], 0)}")
        st.write(f"Top complaint theme: **{s1['top_negative_theme'] or 'none detected'}**")
    with c2.container(border=True):
        st.subheader("📈 Ad performance")
        st.caption(f"Source: {s2['source']}")
        st.metric("ROAS", f"{s2['roas']:.2f}x")
        st.write(f"Best platform: **{s2['best_platform']}** ({s2['best_platform_roas']:.2f}x)")
        st.write(f"Weakest: **{s2['worst_platform']}** ({s2['worst_platform_roas']:.2f}x)")
        st.write(f"Best audience: **{s2['best_age_group']}**, **{s2['best_device']}**, **{s2['best_city']}**")
    with c3.container(border=True):
        st.subheader("🧠 Social intelligence")
        st.caption(f"Source: {s3['source']}")
        st.metric("Comments clustered", fmt_int(s3["comments"]))
        st.write(f"Largest segment: **{s3['top_segment']}** ({fmt_pct(s3['top_segment_share'], 0)})")
        st.write(f"Best format: **{s3['best_format'] or 'n/a'}**")
        st.write(f"Posts flagged as unusual: {fmt_pct(s3['anomaly_share'])}")

    recs = []
    if s1["top_negative_theme"]:
        action = THEME_ACTIONS[s1["top_negative_theme"]]
        recs.append(f"**Fix the message first.** '{s1['top_negative_theme']}' is the most common complaint in the sentiment study: {action}.")
    recs.append(f"**Move budget towards efficiency.** {s2['best_platform']} returns {s2['best_platform_roas']:.2f}x versus "
                f"{s2['worst_platform_roas']:.2f}x on {s2['worst_platform']}. Test a gradual shift in the Budget optimizer before committing.")
    recs.append(f"**Target the right people.** Ad data favours {s2['best_age_group']} on {s2['best_device']}; the largest social "
                f"segment is '{s3['top_segment']}'. Check whether the two audiences overlap before assuming they do.")
    if s3["best_format"]:
        recs.append(f"**Choose the format.** {s3['best_format']} has the highest median engagement in the social data "
                    f"({fmt_pct(s3['best_format_er'], 2)}). Engagement is not ROAS, so confirm with a paid test.")
    recs.append("**Monitor.** Review flagged posts and negative-sentiment spikes weekly; both are prompts for investigation, not proof of a problem.")
    st.markdown("#### Combined recommendations")
    for r in recs:
        st.markdown(f"- {r}")
    synthetic = [n for n, s in [("Project 1", s1), ("Project 2", s2), ("Project 3", s3)] if "synthetic" in s["source"]]
    if synthetic:
        st.warning(f"{', '.join(synthetic)} currently use synthetic demo data, so these recommendations illustrate the "
                   "workflow and are not findings about real brands.")
    summary = {"generated": datetime.now().isoformat(timespec="seconds"), "project1_sentiment": s1,
               "project2_adspend": s2, "project3_social": s3, "recommendations": [re.sub(r"\*\*|'", "", r) for r in recs]}
    c1, c2 = st.columns(2)
    c1.download_button("Download combined summary (JSON)", json.dumps(summary, indent=2, default=str), "combined_summary.json", "application/json")
    c2.download_button("Download combined report (Markdown)", "# Combined insights\n\n" + "\n".join(f"- {r}" for r in recs), "combined_report.md")
    st.caption("To refresh these numbers with your own data, open each project page, choose your data source, then return here.")


# ===========================================================================
# NAVIGATION
# ===========================================================================
PAGES = {
    "Overview": [(page_home, "Home", "🏠"), (page_integrated, "Integrated insights", "🧩")],
    "Project 1 · Sentiment": [(page_sentiment, "Sentiment analysis", "💬")],
    "Project 2 · AdSpend IQ": [(page_adspend, "Performance & audience", "📈")],
    "Project 3 · Social intelligence": [(page_p3_clusters, "Audience clustering", "🧠"),
                                        (page_p3_content, "Content intelligence", "🎬"),
                                        (page_p3_anomaly, "Anomaly detection", "🚨"),
                                        (page_p3_marketing, "Marketing opportunities", "🎯")],
}


def main():
    if hasattr(st, "navigation"):  # Streamlit 1.36+: also disables the old pages/ folder
        nav = {sec: [st.Page(fn, title=title, icon=icon, default=(fn is page_home)) for fn, title, icon in items]
               for sec, items in PAGES.items()}
        st.navigation(nav).run()
    else:  # older Streamlit: simple sidebar selector (upgrade for the full experience)
        flat = {f"{icon} {title}": fn for items in PAGES.values() for fn, title, icon in items}
        st.sidebar.warning("Upgrade Streamlit (pip install -U streamlit) for the full navigation menu.")
        flat[st.sidebar.radio("Page", list(flat))]()


if __name__ == "__main__":
    main()