"""
Spark Optimizer Dashboard  ·  Streamlit
Compatible with all Streamlit versions (no border=True container)
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd, joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = r"C:\Aravindh\data\ml_dataset.csv"
MODEL_PATH = r"C:\Aravindh\models\spark_optimizer.pkl"
MODEL_DIR  = r"C:\Aravindh\models"

# ── Palette ────────────────────────────────────────────────────────────────────
P = dict(
    bg="#0D0F14", panel="#161A22", border="#2A2F3D",
    a1="#5B8DEF", a2="#A78BFA", a3="#34D399", a4="#FB923C",
    text="#E2E8F0", sub="#64748B",
)
plt.rcParams.update({
    "figure.facecolor": P["bg"], "axes.facecolor": P["panel"],
    "axes.edgecolor": P["border"], "axes.labelcolor": P["text"],
    "axes.titlecolor": P["text"], "xtick.color": P["sub"],
    "ytick.color": P["sub"], "text.color": P["text"],
    "grid.color": P["border"], "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
})

def sax(ax, title="", xlabel="", ylabel="", grid=True):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8, color=P["text"])
    ax.set_xlabel(xlabel, fontsize=8, color=P["sub"])
    ax.set_ylabel(ylabel, fontsize=8, color=P["sub"])
    ax.tick_params(labelsize=7.5)
    if grid:
        ax.grid(axis="y", alpha=0.3, ls="--")
        ax.grid(axis="x", alpha=0.12, ls="--")
    for sp in ax.spines.values(): sp.set_edgecolor(P["border"])

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Spark Optimizer", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
/* ── backgrounds ── */
[data-testid="stAppViewContainer"] {{ background:{P["bg"]}; }}
[data-testid="stMain"]             {{ background:{P["bg"]}; }}
[data-testid="stSidebar"]          {{ background:{P["panel"]}; border-right:1px solid {P["border"]}; }}
.block-container                   {{ padding-top:1.8rem !important; max-width:100% !important; }}

/* ── global text ── */
body, p, span, div, label         {{ color:{P["text"]}; }}
h1,h2,h3,h4                       {{ color:{P["text"]} !important; }}
.stMarkdown p                      {{ color:{P["text"]} !important; }}

/* ── predict panel cards ── 
   Only columns inside .predict-wrap get the card border/bg */
.predict-wrap [data-testid="column"] > div:first-child {{
    background    : {P["panel"]};
    border        : 1px solid {P["border"]};
    border-radius : 12px;
    padding       : 18px 16px 20px !important;
    box-sizing    : border-box;
    overflow      : visible;
}}

/* ── metric stat cards (EDA / Model pages) ── */
.mk {{
    background:{P["panel"]}; border:1px solid {P["border"]};
    border-radius:10px; padding:18px 10px; text-align:center; margin-bottom:4px;
}}
.mv {{ font-size:2rem; font-weight:800; line-height:1; }}
.ml {{ font-size:0.8rem; color:{P["sub"]}; margin-top:6px; }}

/* ── section labels ── */
.sl {{
    font-size:0.68rem; font-weight:800; letter-spacing:0.13em;
    text-transform:uppercase; color:{P["sub"]}; margin-bottom:8px; display:block;
}}

/* ── inline chips ── */
.chip {{
    display:inline-block; background:#1A1F2E;
    border:1px solid {P["border"]}; border-radius:6px;
    padding:2px 9px; font-size:0.76rem; margin:2px 1px;
}}

/* ── result output box ── */
.rbox {{
    background:#0B0E16; border:1px solid {P["border"]};
    border-radius:12px; padding:24px 10px 18px;
    text-align:center; margin-bottom:10px;
}}
.rt {{ font-size:2.8rem; font-weight:900; letter-spacing:-2px; line-height:1; }}
.rs {{ font-size:0.78rem; color:{P["sub"]}; margin-top:5px; }}

/* ── run button ── */
div[data-testid="stButton"] > button {{
    background : linear-gradient(135deg,{P["a1"]},{P["a2"]}) !important;
    color:#fff !important; border:none !important;
    border-radius:8px !important; font-weight:700 !important;
    font-size:1rem !important; width:100% !important; padding:10px 0 !important;
}}

/* ── input fields ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {{
    background:{P["panel"]} !important; border-color:{P["border"]} !important;
    color:{P["text"]} !important; border-radius:7px !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Dataset size profiles ──────────────────────────────────────────────────────
DS = {
    "XS  ·  < 1 GB":      dict(gb=0.25,  parts=20,  lbl="< 1 GB",    col=P["a3"]),
    "S   ·  1 – 10 GB":   dict(gb=5.0,   parts=80,  lbl="1–10 GB",   col=P["a1"]),
    "M   ·  10 – 50 GB":  dict(gb=28.0,  parts=200, lbl="10–50 GB",  col=P["a2"]),
    "L   ·  50 – 200 GB": dict(gb=110.0, parts=350, lbl="50–200 GB", col=P["a4"]),
    "XL  ·  > 200 GB":    dict(gb=450.0, parts=500, lbl="> 200 GB",  col="#F87171"),
}
CX = {
    "low":    dict(em="🟢", hint="Simple scans / filters"),
    "medium": dict(em="🟡", hint="Joins, aggregations"),
    "high":   dict(em="🔴", hint="Multi-join, window functions"),
}

# ── Data & model ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH), False
    np.random.seed(42); n = 1200
    df = pd.DataFrame({
        "num_cores":          np.random.choice([4, 8, 16, 32], n),
        "memory_gb":          np.random.choice([8, 16, 32, 64, 128], n),
        "data_size_gb":       np.random.uniform(0.5, 200, n).round(2),
        "num_partitions":     np.random.randint(10, 500, n),
        "shuffle_enabled":    np.random.choice(["yes", "no"], n),
        "cluster_type":       np.random.choice(["on_prem", "cloud_small", "cloud_large"], n),
        "query_complexity":   np.random.choice(["low", "medium", "high"], n),
        "execution_time_sec": (
            np.random.uniform(5, 600, n)
            + np.random.choice([4, 8, 16, 32], n) * 0.8
            + np.random.uniform(0.5, 200, n) * 0.9
        ).round(2),
    })
    return df, True

@st.cache_resource(show_spinner=False)
def get_model(df):
    X = pd.get_dummies(df.drop(columns=["execution_time_sec"]), drop_first=True)
    y = df["execution_time_sec"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = None
    if os.path.exists(MODEL_PATH):
        try:   model = joblib.load(MODEL_PATH)
        except Exception: model = None
    if model is None:
        model = RandomForestRegressor(n_estimators=200, max_depth=8,
                                      random_state=42, n_jobs=-1)
        model.fit(Xtr, ytr)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
    pr  = model.predict(Xte)
    met = dict(mae=mean_absolute_error(yte, pr),
               rmse=float(np.sqrt(mean_squared_error(yte, pr))),
               r2=r2_score(yte, pr))
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, X, Xte, yte, pr, met, imp

df, synth    = load_data()
model, X, Xte, yte, preds, met, imp = get_model(df)
y = df["execution_time_sec"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h2 style='color:{P['a1']};margin-bottom:2px'>⚡ Spark Optimizer</h2>",
                unsafe_allow_html=True)
    st.divider()
    if synth: st.info("Using synthetic demo data")
    else:     st.success(f"ml_dataset.csv  ·  {df.shape[0]:,} rows")
    st.divider()
    page  = st.radio("Navigate", ["📊 EDA", "🤖 Model Results", "🔮 Predict"])
    st.divider()
    top_n = st.slider("Top N features", 5, min(20, len(imp)), 10)

# ══════════════════════════════════════════════════════════════════════════════
#  EDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    c1, c2, c3, c4 = st.columns(4, gap="small")
    for col, lbl, val, hx in zip([c1,c2,c3,c4],
            ["Rows","Columns","Target Mean","Target Std"],
            [f"{df.shape[0]:,}", str(df.shape[1]),
             f"{y.mean():.1f}s", f"{y.std():.1f}s"],
            [P["a1"],P["a2"],P["a3"],P["a4"]]):
        col.markdown(
            f'<div class="mk"><div class="mv" style="color:{hx}">{val}</div>'
            f'<div class="ml">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns(2, gap="medium")
    with ca:
        fig, ax = plt.subplots(figsize=(5,3.2), facecolor=P["bg"])
        ax.hist(y, bins=40, color=P["a1"], edgecolor=P["bg"], alpha=0.9)
        ax.axvline(y.mean(), color=P["a4"], lw=1.5, ls="--",
                   label=f"Mean {y.mean():.1f}s")
        ax.legend(fontsize=8, framealpha=0)
        sax(ax, "Target Distribution", "Execution Time (s)", "Count")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    with cb:
        num_df = df.select_dtypes("number"); corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(5,3.2), facecolor=P["bg"])
        sns.heatmap(corr, mask=np.triu(np.ones_like(corr,bool)), ax=ax,
                    cmap=sns.diverging_palette(220,20,as_cmap=True),
                    vmax=1, vmin=-1, annot=True, fmt=".2f",
                    annot_kws={"size":6.5}, linewidths=0.4, linecolor=P["bg"],
                    cbar_kws={"shrink":0.8})
        ax.set_title("Correlation Matrix", fontsize=10, fontweight="bold", color=P["text"])
        ax.tick_params(labelsize=7, colors=P["sub"])
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    cc, cd = st.columns(2, gap="medium")
    with cc:
        best = corr["execution_time_sec"].drop("execution_time_sec").abs().idxmax()
        fig, ax = plt.subplots(figsize=(5,3.2), facecolor=P["bg"])
        sc = ax.scatter(df[best], y, c=y, cmap="plasma", s=9, alpha=0.55, linewidths=0)
        plt.colorbar(sc, ax=ax, pad=0.02).ax.tick_params(labelsize=7)
        sax(ax, f"{best} vs Target", best, "Execution Time (s)")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    with cd:
        cats = df.select_dtypes("object").columns.tolist()
        if cats:
            chosen = st.selectbox("Category column", cats)
            fig, ax = plt.subplots(figsize=(5,3.2), facecolor=P["bg"])
            grps = [df.loc[df[chosen]==v,"execution_time_sec"].values
                    for v in df[chosen].unique()]
            bp = ax.boxplot(grps, patch_artist=True, notch=False,
                            medianprops=dict(color=P["a3"],lw=2),
                            whiskerprops=dict(color=P["sub"]),
                            capprops=dict(color=P["sub"]),
                            flierprops=dict(marker="o",ms=2,
                                            markerfacecolor=P["a4"],alpha=0.5))
            for patch, c in zip(bp["boxes"],
                    [P["a1"],P["a2"],P["a3"],P["a4"],P["sub"]]):
                patch.set_facecolor(c); patch.set_alpha(0.75)
            ax.set_xticklabels(df[chosen].unique(), fontsize=7.5, rotation=20)
            sax(ax, f"Target by {chosen}", chosen, "Execution Time (s)", grid=False)
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
        else:
            st.info("No categorical columns detected.")

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Results":
    st.title("🤖 Model Results")
    c1, c2, c3 = st.columns(3, gap="small")
    for col, lbl, val, unit, hx in zip([c1,c2,c3],
            ["MAE","RMSE","R²"],
            [f"{met['mae']:.3f}",f"{met['rmse']:.3f}",f"{met['r2']:.4f}"],
            ["sec","sec",""],
            [P["a1"],P["a2"],P["a3"]]):
        col.markdown(
            f'<div class="mk"><div class="mv" style="color:{hx}">{val}</div>'
            f'<div class="ml">{lbl}{"  ("+unit+")" if unit else ""}</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns(2, gap="medium")
    with ca:
        lims = [min(yte.min(),preds.min()), max(yte.max(),preds.max())]
        fig, ax = plt.subplots(figsize=(5,3.5), facecolor=P["bg"])
        ax.scatter(yte, preds, c=P["a1"], s=10, alpha=0.55, linewidths=0)
        ax.plot(lims, lims, "--", color=P["a4"], lw=1.5, label="Perfect fit")
        ax.fill_between(lims,
                        [l-met["mae"] for l in lims],
                        [l+met["mae"] for l in lims],
                        alpha=0.1, color=P["a4"], label="MAE band")
        ax.legend(fontsize=8, framealpha=0)
        sax(ax, "Actual vs Predicted", "Actual (s)", "Predicted (s)")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    with cb:
        res = yte.values - preds
        fig, ax = plt.subplots(figsize=(5,3.5), facecolor=P["bg"])
        ax.hist(res, bins=35, color=P["a2"], edgecolor=P["bg"], alpha=0.85)
        ax.axvline(0, color=P["a4"], lw=1.5, ls="--")
        sax(ax, "Residual Distribution", "Residual (s)", "Count")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown("### Feature Importance")
    ti   = imp.head(top_n)
    norm = ti.values / ti.values.max()
    bc   = [P["a1"] if v>0.6 else P["a2"] if v>0.3 else P["sub"] for v in norm]
    fig, ax = plt.subplots(figsize=(9, top_n*0.42+1), facecolor=P["bg"])
    bars = ax.barh(ti.index[::-1], ti.values[::-1],
                   color=bc[::-1], edgecolor=P["bg"], height=0.65)
    for bar in bars:
        ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                f"{bar.get_width():.4f}", va="center", fontsize=8, color=P["text"])
    sax(ax, f"Top {top_n} Feature Importances", "Importance Score", "", grid=False)
    fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT  — 3 equal columns, card style via .predict-wrap CSS class
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.title("🔮 Predict Execution Time")
    st.markdown(
        f"<p style='color:{P['sub']};margin-top:-8px;margin-bottom:16px;'>"
        "Select a dataset profile, configure the query, then run the prediction.</p>",
        unsafe_allow_html=True)

    # ── open the wrapper div that scopes the card CSS ──
    st.markdown("<div class='predict-wrap'>", unsafe_allow_html=True)
    col_L, col_M, col_R = st.columns(3, gap="medium")

    # ── LEFT: Dataset profile ──────────────────────────────────────────────
    with col_L:
        st.markdown(f"<span class='sl'>📦 Dataset Profile</span>", unsafe_allow_html=True)
        size_key = st.selectbox("size", list(DS.keys()), index=1,
                                label_visibility="collapsed")
        prof = DS[size_key]
        st.markdown(
            f"<span class='chip' style='color:{prof['col']};border-color:{prof['col']};'>"
            f"{prof['lbl']}</span>",
            unsafe_allow_html=True)
        st.divider()
        data_size = st.number_input("Data Size (GB)", 0.1, 10000.0,
                                    float(prof["gb"]), 0.5)
        num_parts = st.number_input("Num Partitions", 1, 2000,
                                    int(prof["parts"]), 10)
        st.divider()
        st.markdown(f"<span class='sl'>🖥 Cluster Config</span>", unsafe_allow_html=True)
        num_cores    = st.selectbox("CPU Cores",    [4, 8, 16, 32],                           index=1)
        memory_gb    = st.selectbox("Memory (GB)", [8, 16, 32, 64, 128],                      index=2)
        cluster_type = st.selectbox("Cluster Type",["on_prem","cloud_small","cloud_large"],   index=1)

    # ── MIDDLE: Query config ───────────────────────────────────────────────
    with col_M:
        st.markdown(f"<span class='sl'>🔍 Query Complexity</span>", unsafe_allow_html=True)
        query_cx = st.radio(
            "cx", list(CX.keys()), index=1, label_visibility="collapsed",
            format_func=lambda v: f"{CX[v]['em']}  {v.capitalize()}  —  {CX[v]['hint']}")
        st.divider()
        st.markdown(f"<span class='sl'>⚙️ Runtime Options</span>", unsafe_allow_html=True)
        shuffle = st.radio(
            "sh", ["yes","no"], horizontal=True, label_visibility="collapsed",
            format_func=lambda v: "✅ Shuffle On" if v=="yes" else "❌ Shuffle Off")
        st.divider()
        st.markdown(f"<span class='sl'>📋 Config Summary</span>", unsafe_allow_html=True)
        em = CX[query_cx]["em"]
        st.markdown(f"""
        <div style='line-height:2.2;font-size:0.82rem;'>
            <span style='color:{P["sub"]}'>Size</span>&nbsp;
            <span class='chip' style='color:{prof["col"]};border-color:{prof["col"]};'>
                {prof["lbl"]}</span>&nbsp;
            <span style='color:{P["sub"]}'>Complexity</span>&nbsp;
            <span class='chip'>{em} {query_cx.capitalize()}</span><br>
            <span style='color:{P["sub"]}'>Cluster</span>&nbsp;
            <span class='chip'>{cluster_type}</span>&nbsp;
            <span style='color:{P["sub"]}'>Shuffle</span>&nbsp;
            <span class='chip'>{shuffle}</span><br>
            <span style='color:{P["sub"]}'>Cores</span>&nbsp;
            <span class='chip'>{num_cores}</span>&nbsp;
            <span style='color:{P["sub"]}'>RAM</span>&nbsp;
            <span class='chip'>{memory_gb} GB</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("⚡  Run Prediction")

    # ── RIGHT: Output ──────────────────────────────────────────────────────
    with col_R:
        st.markdown(f"<span class='sl'>📈 Prediction Output</span>", unsafe_allow_html=True)

        if run:
            row = pd.DataFrame([{
                "num_cores": num_cores, "memory_gb": memory_gb,
                "data_size_gb": data_size, "num_partitions": num_parts,
                "shuffle_enabled": shuffle, "cluster_type": cluster_type,
                "query_complexity": query_cx,
            }])
            row_enc = pd.get_dummies(row, drop_first=True).reindex(
                columns=X.columns, fill_value=0)
            pred = model.predict(row_enc)[0]

            if   pred < 60:  tier, tc, ti = "Fast",      P["a3"], "🟢"
            elif pred < 300: tier, tc, ti = "Moderate",  P["a1"], "🟡"
            elif pred < 600: tier, tc, ti = "Slow",      P["a4"], "🟠"
            else:            tier, tc, ti = "Very Slow", P["a2"], "🔴"

            mins = int(pred // 60); secs = pred % 60
            tstr = f"{mins}m {secs:.0f}s" if mins else f"{secs:.1f}s"

            st.markdown(f"""
            <div class='rbox'>
                <div class='rt' style='color:{tc};'>{tstr}</div>
                <div class='rs'>estimated execution time</div>
                <div style='margin-top:12px;font-size:1rem;'>
                    {ti}&nbsp;<span style='color:{tc};font-weight:700;'>{tier}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            mae_v = met["mae"]
            lo, hi = max(0.0, pred - mae_v), pred + mae_v

            fig, ax = plt.subplots(figsize=(3.2, 0.85), facecolor=P["panel"])
            ax.barh([0], [hi-lo], left=lo, height=0.5,
                    color=tc, alpha=0.2, edgecolor="none")
            ax.axvline(pred, color=tc, lw=2.5)
            ax.axvline(lo,   color=P["sub"], lw=1, ls="--")
            ax.axvline(hi,   color=P["sub"], lw=1, ls="--")
            ax.set_yticks([])
            ax.set_xlabel("seconds", fontsize=7, color=P["sub"])
            ax.tick_params(labelsize=7, colors=P["sub"])
            for sp in ax.spines.values(): sp.set_edgecolor(P["border"])
            fig.patch.set_facecolor(P["panel"])
            fig.tight_layout(pad=0.3)
            st.pyplot(fig, use_container_width=True); plt.close(fig)

            st.divider()
            for lbl, val, color in [
                ("Low estimate",  f"{lo:.1f} s",     P["sub"]),
                ("Prediction",    f"{pred:.1f} s",   tc),
                ("High estimate", f"{hi:.1f} s",     P["sub"]),
                ("Model MAE",     f"+/-{mae_v:.1f} s", P["sub"]),
            ]:
                ra, rb = st.columns([1.6, 1])
                ra.markdown(
                    f"<span style='color:{P['sub']};font-size:0.8rem;'>{lbl}</span>",
                    unsafe_allow_html=True)
                rb.markdown(
                    f"<span style='color:{color};font-size:0.8rem;font-weight:700;'>{val}</span>",
                    unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='text-align:center;padding:70px 0 60px;'>
                <div style='font-size:2.8rem;color:{P["border"]};'>⚡</div>
                <div style='font-size:0.82rem;color:{P["sub"]};margin-top:14px;'>
                    Configure and click<br>
                    <strong style='color:{P["a1"]};'>Run Prediction</strong>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── close wrapper ──
    st.markdown("</div>", unsafe_allow_html=True)