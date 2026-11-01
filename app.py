"""Streamlit front-end: evaluate the five SAVED classifiers on an uploaded test CSV.

No model is trained or re-fitted here -- the app only loads the .joblib files
produced by train_models.py and calls predict / predict_proba on the upload.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from metrics_utils import METRIC_ORDER, confusion, evaluate_model, report_dict

MODEL_DIR = Path("model")
SAMPLE_FILE = Path("test_data.csv")

INK, MUTED, PAPER, LINE, GOLD = "#1B2A41", "#5F6B7A", "#FAF8F4", "#E4DED2", "#B8893B"
CLASS_COLORS = {"Dropout": "#A63D40", "Enrolled": "#C9963B", "Graduate": "#2F6F5E"}
METRIC_HELP = {
    "Accuracy": "Share of students classified correctly.",
    "AUC": "One-vs-rest ROC AUC, macro-averaged over the classes.",
    "Precision": "Macro-averaged: every class counts equally.",
    "Recall": "Macro-averaged: every class counts equally.",
    "F1": "Macro-averaged harmonic mean of precision and recall.",
    "MCC": "Matthews correlation coefficient: -1 to 1, robust to class imbalance.",
}

st.set_page_config(page_title="Student Outcome Model Bench", page_icon="🎓", layout="wide")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
.block-container {{ padding-top: 3.4rem; max-width: 1180px; }}
#MainMenu, footer {{ visibility: hidden; }}
h1, h2, h3, .serif {{ font-family: 'Newsreader', Georgia, 'Times New Roman', serif !important;
                      letter-spacing: -0.01em; color: {INK}; }}
.eyebrow {{ font-size: .72rem; letter-spacing: .16em; text-transform: uppercase; color: {GOLD};
            font-weight: 600; margin-bottom: .6rem; line-height: 1.5; }}
.hero-title {{ font-family: 'Newsreader', Georgia, serif; font-size: 2.7rem; font-weight: 600;
               line-height: 1.2; color: {INK}; margin: 0 0 .7rem 0; padding-top: .15rem; }}
.hero-sub {{ color: {MUTED}; font-size: 1.02rem; max-width: 52rem; line-height: 1.55; margin-bottom: .4rem; }}
.rule {{ height: 1px; background: linear-gradient(90deg, {GOLD}, {LINE} 40%, transparent);
         border: 0; margin: 1.1rem 0 1.4rem 0; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}
.brand {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.35rem; font-weight: 600; color: {INK}; }}
.brand-sub {{ font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: {MUTED};
              margin: .1rem 0 1.2rem 0; }}
.step {{ font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: {GOLD};
         font-weight: 600; margin: 1.1rem 0 .2rem 0; }}
[data-testid="stMetric"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 12px;
                            padding: 14px 16px; box-shadow: 0 1px 0 rgba(27,42,65,.04); }}
[data-testid="stMetricLabel"] p {{ font-size: .7rem; letter-spacing: .1em; text-transform: uppercase;
                                   color: {MUTED}; font-weight: 600; }}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{
    font-family: 'Newsreader', Georgia, serif; font-size: 1.85rem; font-weight: 600; color: {INK}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: .5rem 0; font-weight: 500; }}
.chip {{ display: inline-flex; align-items: center; gap: .45rem; padding: .3rem .75rem;
         border: 1px solid {LINE}; border-radius: 999px; background: #fff; font-size: .86rem;
         margin: 0 .45rem .45rem 0; color: {INK}; }}
.chip b {{ font-variant-numeric: tabular-nums; }}
.dot {{ width: .62rem; height: .62rem; border-radius: 50%; display: inline-block; }}
.badge {{ background: #F5ECD9; border-color: #E3CFA0; color: #7A5A1C; font-weight: 600; }}
.label {{ font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: {MUTED};
          font-weight: 600; margin: .2rem 0 .5rem 0; }}
.empty {{ border: 1px dashed #CFC6B3; border-radius: 14px; padding: 1.5rem 1.8rem; background: #fff; }}
.steps {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem; margin-top: .8rem; }}
.steps div b {{ font-family: 'Newsreader', Georgia, serif; font-size: 1.6rem; color: {GOLD}; display: block; }}
.steps div span {{ color: {MUTED}; font-size: .92rem; line-height: 1.5; }}
.foot {{ margin-top: 2.4rem; padding-top: .9rem; border-top: 1px solid {LINE}; color: {MUTED};
         font-size: .8rem; }}
@media (max-width: 760px) {{ .steps {{ grid-template-columns: 1fr; }} .hero-title {{ font-size: 2rem; }} }}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ loading --
@st.cache_resource
def load_artifacts():
    meta = joblib.load(MODEL_DIR / "meta.joblib")
    models = {name: joblib.load(MODEL_DIR / f) for name, f in meta["models"].items()}
    return meta, models


@st.cache_data
def read_csv(source):
    df = pd.read_csv(source)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data(show_spinner=False)
def score_all(df: pd.DataFrame, _models, _meta):
    """Evaluate every saved model on the uploaded frame (inference only)."""
    class_names = _meta["class_names"]
    X = df[_meta["feature_names"]]
    y_true = df[_meta["target"]].map({c: i for i, c in enumerate(class_names)}).to_numpy()
    out = {}
    for name, model in _models.items():
        metrics, y_pred = evaluate_model(model, X, y_true, len(class_names))
        out[name] = {"metrics": metrics, "y_pred": y_pred}
    return y_true, out


def validate(df, meta):
    """Return a list of human-readable problems (empty list == file is usable)."""
    problems = []
    target = meta["target"]
    missing = [c for c in meta["feature_names"] + [target] if c not in df.columns]
    if missing:
        shown = ", ".join(f"`{c}`" for c in missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        problems.append(f"Missing required columns: {shown}{more}.")
        return problems
    if df[meta["feature_names"] + [target]].isna().any().any():
        problems.append("The file contains empty cells. Remove or fill them and upload again.")
    unknown = set(df[target].unique()) - set(meta["class_names"])
    if unknown:
        problems.append(
            f"`{target}` has unexpected labels {sorted(map(str, unknown))}; "
            f"allowed labels are {meta['class_names']}."
        )
    return problems


# --------------------------------------------------------------------- plots --
def _clean_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=MUTED, length=0)


def plot_confusion(cm, class_names, title):
    """Colour = share of each ACTUAL class (row-normalised); text = count and %."""
    row_sums = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)
    labels = np.array([[f"{cm[i, j]}\n{share[i, j]:.0%}" for j in range(cm.shape[1])]
                       for i in range(cm.shape[0])])
    cmap = LinearSegmentedColormap.from_list("ink", ["#F4F1EA", "#8FA6BE", INK])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    fig.patch.set_facecolor(PAPER)
    sns.heatmap(share, annot=labels, fmt="", cmap=cmap, vmin=0, vmax=1, cbar=False, square=True,
                linewidths=3, linecolor=PAPER, xticklabels=class_names, yticklabels=class_names,
                annot_kws={"fontsize": 11, "fontweight": "600"}, ax=ax)
    for text, (i, j) in zip(ax.texts, np.ndindex(cm.shape)):
        text.set_color("white" if share[i, j] > 0.5 else INK)
    ax.set_xlabel("Predicted", color=MUTED, labelpad=8)
    ax.set_ylabel("Actual", color=MUTED, labelpad=8)
    ax.tick_params(colors=INK, length=0)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="600", pad=12)
    fig.tight_layout()
    return fig


def plot_comparison(table):
    colors = [INK, "#41607F", "#8AA4BD", "#D8BE86", GOLD, "#7A5A1C"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    table.set_index("ML Model Name")[METRIC_ORDER].plot.bar(ax=ax, width=0.82, color=colors)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Score", color=MUTED)
    ax.yaxis.grid(True, color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False,
              fontsize=9, labelcolor=INK)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", color=INK)
    fig.tight_layout()
    return fig


def chip(label, value, color=None):
    dot = f'<span class="dot" style="background:{color}"></span>' if color else ""
    return f'<span class="chip">{dot}{label} <b>{value}</b></span>'


# ---------------------------------------------------------------------- page --
st.markdown(
    """
<div class="eyebrow">UCI dataset 697 · three-class classification</div>
<div class="hero-title">Student Outcome Model Bench</div>
<div class="hero-sub">Five pre-trained classifiers estimate whether a student will <b>drop out</b>,
stay <b>enrolled</b> or <b>graduate</b>. Upload a labelled test file and compare them side by side.
Nothing is retrained in this app: the saved models only predict.</div>
<hr class="rule">
""",
    unsafe_allow_html=True,
)

try:
    meta, models = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found in `model/`. Run `python train_models.py` and commit the folder.")
    st.stop()

if meta["sklearn_version"] != sklearn.__version__:
    st.warning(
        f"Models were trained with scikit-learn {meta['sklearn_version']} but this app runs "
        f"{sklearn.__version__}. Pin the version in requirements.txt if results look off."
    )

class_names = meta["class_names"]
color_of = lambda c: CLASS_COLORS.get(c, MUTED)

with st.sidebar:
    st.markdown('<div class="brand">Model Bench</div>'
                '<div class="brand-sub">Dropout &amp; academic success</div>', unsafe_allow_html=True)
    st.markdown('<div class="step">Step 1 · Test data</div>', unsafe_allow_html=True)
    upload = st.file_uploader("Upload test CSV", type=["csv"], label_visibility="collapsed",
                              help=f"Must include all {len(meta['feature_names'])} feature columns "
                                   f"and a `{meta['target']}` column.")
    use_sample = False
    if SAMPLE_FILE.exists():
        use_sample = st.checkbox("No file handy? Use the bundled hold-out set", value=False)
    st.markdown('<div class="step">Step 2 · Classifier</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Classifier", list(models.keys()), label_visibility="collapsed")
    with st.expander("Expected columns"):
        st.write(meta["feature_names"] + [meta["target"]])

source = upload if upload is not None else (SAMPLE_FILE if use_sample else None)
if source is None:
    st.markdown(
        """
<div class="empty"><div class="label">Get started</div>
<div class="steps">
<div><b>01</b><span>Upload a CSV in the sidebar. It needs every feature column plus the <code>Target</code> label.</span></div>
<div><b>02</b><span>Pick one of the five classifiers: Logistic Regression, Decision Tree, KNN, Naive Bayes or SVM.</span></div>
<div><b>03</b><span>Read the metrics, the confusion matrix, the model comparison and student-level predictions.</span></div>
</div></div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

df = read_csv(source)
problems = validate(df, meta)
if problems:
    for p in problems:
        st.error(p)
    st.stop()

y_true, results = score_all(df, models, meta)
table = pd.DataFrame([{"ML Model Name": n, **r["metrics"]} for n, r in results.items()])
best_model = table.loc[table["MCC"].idxmax(), "ML Model Name"]

present = [c for c in class_names if c in set(df[meta["target"]])]
st.markdown(
    chip("Students", f"{len(df):,}") + chip("Features", len(meta["feature_names"]))
    + "".join(chip(c, int((df[meta["target"]] == c).sum()), color_of(c)) for c in present),
    unsafe_allow_html=True,
)

tab_one, tab_all, tab_rows = st.tabs(["Selected model", "All five models", "Student-level predictions"])

# ---- tab 1: selected model --------------------------------------------------
with tab_one:
    res = results[model_name]
    st.markdown(f"## {model_name}")
    if model_name == best_model:
        st.markdown(chip("★", "Highest MCC of the five models on this file").replace(
            'class="chip"', 'class="chip badge"'), unsafe_allow_html=True)

    cols = st.columns(6)
    for col, key in zip(cols, METRIC_ORDER):
        val = res["metrics"][key]
        col.metric(key, "n/a" if pd.isna(val) else f"{val:.4f}", help=METRIC_HELP[key])
    st.caption("Precision, Recall and F1 are macro-averaged over the classes; AUC is one-vs-rest macro.")

    rep = report_dict(y_true, res["y_pred"], class_names)
    st.markdown('<div class="label">Recall by class: share of real cases this model catches</div>',
                unsafe_allow_html=True)
    st.markdown("".join(chip(c, f"{rep[c]['recall']:.0%}", color_of(c)) for c in class_names),
                unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")
    with left:
        cm = confusion(y_true, res["y_pred"], len(class_names))
        fig = plot_confusion(cm, class_names, "Confusion matrix")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown('<div class="label">Classification report</div>', unsafe_allow_html=True)
        rep_df = pd.DataFrame({k: v for k, v in rep.items() if k != "accuracy"}).T
        rep_df = rep_df.rename(columns={"f1-score": "f1"})
        rep_df["support"] = rep_df["support"].astype(int)
        st.dataframe(rep_df.round(3))

# ---- tab 2: comparison -------------------------------------------------------
with tab_all:
    st.markdown("## Side-by-side comparison")
    st.dataframe(
        table.style.format({m: "{:.4f}" for m in METRIC_ORDER})
        .highlight_max(subset=METRIC_ORDER, color="#EADFC4"),
        hide_index=True,
    )
    st.caption(f"Highest value per metric is shaded. Best MCC on this file: **{best_model}**.")
    fig = plot_comparison(table)
    st.pyplot(fig)
    plt.close(fig)

# ---- tab 3: row level --------------------------------------------------------
with tab_rows:
    X = df[meta["feature_names"]]
    proba = pd.DataFrame(models[model_name].predict_proba(X),
                         columns=[f"P({c})" for c in class_names], index=df.index)
    out = pd.DataFrame({
        "Actual": df[meta["target"]],
        "Predicted": [class_names[i] for i in results[model_name]["y_pred"]],
    }, index=df.index).join(proba.round(3))
    out["Correct"] = out["Actual"] == out["Predicted"]

    st.markdown(f"## Student-level predictions · {model_name}")
    flagged = int((out["P(Dropout)"] >= 0.5).sum()) if "P(Dropout)" in out else 0
    st.markdown(
        chip("Flagged as likely dropout (P ≥ 0.5)", flagged, color_of("Dropout"))
        + chip("Correct", int(out["Correct"].sum()))
        + chip("Misclassified", int((~out["Correct"]).sum())),
        unsafe_allow_html=True,
    )
    if "Dropout" in class_names:
        out = out.sort_values("P(Dropout)", ascending=False)
    only_wrong = st.checkbox("Show misclassified students only")
    shown = out[~out["Correct"]] if only_wrong else out

    view = shown.reset_index(names="Row")
    view["Result"] = np.where(view["Correct"], "✓", "✗")
    view = view.drop(columns="Correct")
    bars = {f"P({c})": st.column_config.ProgressColumn(f"P({c})", min_value=0.0, max_value=1.0,
                                                       format="%.2f") for c in class_names}
    st.dataframe(view, hide_index=True, column_config=bars, height=440)
    st.download_button("Download predictions (CSV)", shown.to_csv().encode("utf-8"),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                       mime="text/csv")

st.markdown(
    '<div class="foot">Data: Predict Students\' Dropout and Academic Success, UCI Machine Learning '
    'Repository (dataset 697, CC BY 4.0). Evaluation uses saved models; no training happens here.</div>',
    unsafe_allow_html=True,
)