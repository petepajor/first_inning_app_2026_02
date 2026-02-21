"""
app.py — First Inning Run Probability Tool
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

from model import (
    train_model, save_model, load_model,
    get_pitcher_rolling_stats, score_pitchers,
    prob_to_american, american_to_implied, edge,
    PITCHER_FEATURE_COLS, get_feature_cols,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="1st Inning Run Probability",
    page_icon="⚾",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');
  
  .metric-card {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 8px;
  }
  .prob-high { color: #ff4a6e; font-size: 2rem; font-weight: 700; }
  .prob-mid  { color: #ffa54a; font-size: 2rem; font-weight: 700; }
  .prob-low  { color: #4affa3; font-size: 2rem; font-weight: 700; }
  .edge-pos  { color: #4affa3; font-weight: 600; }
  .edge-neg  { color: #ff4a6e; font-weight: 600; }
  .stDataFrame { font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
  div[data-testid="metric-container"] { background: #111318; border-radius: 8px; padding: 12px; border: 1px solid #1e2330; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
REQUIRED_PIT_COLS = {"Date", "Name", "Tm", "R", "K%", "BB%", "K-BB%", "wOBA", "xFIP", "BABIP", "GB%", "LD%", "FB%", "WHIP"}
REQUIRED_OFF_COLS = {"Date", "Tm", "R"}

@st.cache_data(show_spinner=False)
def cached_train(file_hash: str, pit_df_json: str):
    pit_df = pd.read_json(pit_df_json)
    return train_model(pit_df)

def load_csv_or_xlsx(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Please upload a .csv or .xlsx file.")
        return None

def validate_df(df, required_cols, label):
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"**{label}** is missing columns: {', '.join(sorted(missing))}")
        return False
    return True

def prob_color_class(p):
    if p >= 0.40: return "prob-high"
    if p >= 0.28: return "prob-mid"
    return "prob-low"

def format_edge(e):
    sign = "+" if e >= 0 else ""
    cls = "edge-pos" if e >= 0 else "edge-neg"
    return f'<span class="{cls}">{sign}{e*100:.1f}pp</span>'

def get_file_hash(f):
    import hashlib
    f.seek(0)
    h = hashlib.md5(f.read()).hexdigest()
    f.seek(0)
    return h

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "model_meta" not in st.session_state:
    st.session_state.model_meta = None
if "pit_df" not in st.session_state:
    st.session_state.pit_df = None
if "off_df" not in st.session_state:
    st.session_state.off_df = None
if "matchups" not in st.session_state:
    st.session_state.matchups = []

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("## ⚾ First Inning Run Probability")
st.markdown("Upload your FanGraphs first-inning splits CSVs, then build matchups to get model probabilities and implied odds.")

st.divider()

# ─────────────────────────────────────────────
# SIDEBAR — DATA UPLOAD + MODEL TRAINING
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Data Upload")
    st.caption("Download from FanGraphs › Splits › First Inning. Upload both files each day.")

    pit_file = st.file_uploader(
        "Pitching — First Inning Splits",
        type=["csv", "xlsx"],
        key="pit_upload",
        help="FanGraphs pitching splits filtered to 1st inning"
    )
    off_file = st.file_uploader(
        "Offense — First Inning Splits",
        type=["csv", "xlsx"],
        key="off_upload",
        help="FanGraphs team offense splits filtered to 1st inning"
    )

    if pit_file:
        pit_df = load_csv_or_xlsx(pit_file)
        if pit_df is not None and validate_df(pit_df, REQUIRED_PIT_COLS, "Pitching file"):
            for c in PITCHER_FEATURE_COLS + ["R"]:
                if c in pit_df.columns:
                    pit_df[c] = pd.to_numeric(pit_df[c], errors="coerce")
            pit_df["Date"] = pd.to_datetime(pit_df["Date"], errors="coerce")
            st.session_state.pit_df = pit_df
            st.success(f"✓ Pitching: {len(pit_df):,} rows, {pit_df['Name'].nunique()} pitchers")

    if off_file:
        off_df = load_csv_or_xlsx(off_file)
        if off_df is not None and validate_df(off_df, REQUIRED_OFF_COLS, "Offense file"):
            off_df["Date"] = pd.to_datetime(off_df["Date"], errors="coerce")
            st.session_state.off_df = off_df
            st.success(f"✓ Offense: {len(off_df):,} rows, {off_df['Tm'].nunique()} teams")

    st.divider()

    if st.session_state.pit_df is not None:
        if st.button("🔄 Train / Retrain Model", type="primary", use_container_width=True):
            with st.spinner("Training GBM model on uploaded data..."):
                meta = train_model(st.session_state.pit_df)
                st.session_state.model_meta = meta
            st.success(f"Model trained on {meta['n_starts']:,} starts · {meta['n_pitchers']} pitchers")
            st.caption(f"League avg 1st-inn run rate: {meta['baseline']*100:.1f}%")
    else:
        st.info("Upload pitching splits above to enable training.")

    if st.session_state.model_meta:
        st.divider()
        st.markdown("### 📊 Feature Importance")
        imp = st.session_state.model_meta["feature_importances"]
        imp_clean = {
            k.replace("roll_", "").replace("run_rate", "Run Rate"): v
            for k, v in sorted(imp.items(), key=lambda x: -x[1])
        }
        max_imp = max(imp_clean.values())
        for feat, val in list(imp_clean.items())[:8]:
            pct = val / max_imp
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
                f"<span style='font-size:11px;color:#6b7591;width:70px;text-align:right;flex-shrink:0'>{feat}</span>"
                f"<div style='flex:1;background:#1e2330;height:6px;border-radius:3px'>"
                f"<div style='width:{pct*100:.0f}%;background:#e8ff4a;height:100%;border-radius:3px'></div></div>"
                f"<span style='font-size:11px;color:#4a5168;width:30px'>{val*100:.1f}%</span>"
                f"</div>",
                unsafe_allow_html=True
            )

# ─────────────────────────────────────────────
# MAIN — MATCHUP BUILDER
# ─────────────────────────────────────────────
if st.session_state.model_meta is None:
    st.info("👈 Upload your data files and train the model in the sidebar to get started.")
    st.stop()

meta = st.session_state.model_meta
pit_df = st.session_state.pit_df

# Get known pitchers for autocomplete
rolling = get_pitcher_rolling_stats(pit_df)
known_pitchers = sorted(rolling["Name"].dropna().unique().tolist())

st.markdown("### 🎯 Today's Matchups")
st.caption("Build each game's matchup. Enter the sportsbook's American odds to see your edge.")

# ─────────────────────────────────────────────
# ADD MATCHUP FORM
# ─────────────────────────────────────────────
with st.expander("➕ Add a Matchup", expanded=len(st.session_state.matchups) == 0):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Away Team Pitcher**")
        away_pitcher = st.selectbox(
            "Away Pitcher", options=[""] + known_pitchers, key="away_pitcher_sel",
            label_visibility="collapsed"
        )
        away_odds_yes = st.text_input(
            "Book odds — Yes (run scored)", value="", placeholder="e.g. +120 or -130",
            key="away_odds_yes",
            help="Sportsbook American odds for YES a run is scored in the 1st"
        )
        away_odds_no = st.text_input(
            "Book odds — No (run scored)", value="", placeholder="e.g. -150",
            key="away_odds_no",
        )

    with col2:
        st.markdown("**Home Team Pitcher**")
        home_pitcher = st.selectbox(
            "Home Pitcher", options=[""] + known_pitchers, key="home_pitcher_sel",
            label_visibility="collapsed"
        )
        home_odds_yes = st.text_input(
            "Book odds — Yes", value="", placeholder="e.g. +140",
            key="home_odds_yes",
        )
        home_odds_no = st.text_input(
            "Book odds — No", value="", placeholder="e.g. -170",
            key="home_odds_no",
        )

    game_label = st.text_input("Game label (optional)", placeholder="e.g. NYY @ BOS 7:10pm")

    if st.button("Add Matchup", type="primary"):
        if not away_pitcher and not home_pitcher:
            st.warning("Select at least one pitcher.")
        else:
            matchup = {
                "label": game_label or f"{away_pitcher} @ {home_pitcher}",
                "away": {"name": away_pitcher, "odds_yes": away_odds_yes, "odds_no": away_odds_no},
                "home": {"name": home_pitcher, "odds_yes": home_odds_yes, "odds_no": home_odds_no},
            }
            st.session_state.matchups.append(matchup)
            st.rerun()

# ─────────────────────────────────────────────
# DISPLAY MATCHUPS
# ─────────────────────────────────────────────
if not st.session_state.matchups:
    st.markdown("---")
    st.markdown("No matchups yet. Add one above.")
    st.stop()

col_reset, col_spacer = st.columns([1, 5])
with col_reset:
    if st.button("🗑 Clear All Matchups"):
        st.session_state.matchups = []
        st.rerun()

st.divider()

# Score all pitchers in batch
all_names = []
for m in st.session_state.matchups:
    if m["away"]["name"]: all_names.append(m["away"]["name"])
    if m["home"]["name"]: all_names.append(m["home"]["name"])
all_names = list(set(all_names))

scores_df = score_pitchers(meta, all_names, pit_df)
scores = {row["Name"]: row for _, row in scores_df.iterrows()} if not scores_df.empty else {}

def render_pitcher_card(side_label, pitcher_info, scores, meta, col):
    name = pitcher_info["name"]
    odds_yes_str = pitcher_info.get("odds_yes", "")
    odds_no_str = pitcher_info.get("odds_no", "")

    with col:
        st.markdown(f"**{side_label}**")

        if not name:
            st.markdown("*No pitcher selected*")
            return

        if name not in scores or not scores[name].get("found", False):
            reason = scores.get(name, {}).get("reason", "not found in data")
            st.warning(f"**{name}** — {reason}")
            return

        s = scores[name]
        prob = s["prob"]
        baseline = meta["baseline"]
        model_odds_yes = prob_to_american(prob)
        model_odds_no = prob_to_american(1 - prob)

        # Prob color
        color = "#ff4a6e" if prob >= 0.40 else ("#ffa54a" if prob >= 0.28 else "#4affa3")
        vs_baseline = prob - baseline
        vs_sign = "+" if vs_baseline >= 0 else ""
        vs_color = "#ff4a6e" if vs_baseline > 0.03 else ("#4affa3" if vs_baseline < -0.03 else "#aaa")

        st.markdown(
            f"<div style='background:#111318;border:1px solid #1e2330;border-radius:8px;padding:16px'>"
            f"<div style='font-size:13px;color:#aaa;margin-bottom:4px'>{name} &nbsp;"
            f"<span style='background:#1e2330;padding:2px 7px;border-radius:3px;font-size:11px;color:#6b7591'>{s.get('Team','')}</span></div>"
            f"<div style='font-size:36px;font-weight:700;color:{color};line-height:1.1'>{prob*100:.0f}%</div>"
            f"<div style='font-size:12px;color:#6b7591;margin-bottom:10px'>chance of 1st inning run</div>"
            f"<div style='font-size:12px;color:{vs_color};margin-bottom:12px'>"
            f"{vs_sign}{vs_baseline*100:.1f}pp vs league avg ({baseline*100:.1f}%)</div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px'>"
            f"<div style='background:#0a0c10;border-radius:5px;padding:10px'>"
            f"<div style='font-size:10px;color:#4a5168;text-transform:uppercase;letter-spacing:1px'>Model Odds (Yes)</div>"
            f"<div style='font-size:18px;font-weight:600;color:#e8ff4a'>{model_odds_yes}</div></div>"
            f"<div style='background:#0a0c10;border-radius:5px;padding:10px'>"
            f"<div style='font-size:10px;color:#4a5168;text-transform:uppercase;letter-spacing:1px'>Model Odds (No)</div>"
            f"<div style='font-size:18px;font-weight:600;color:#e8ff4a'>{model_odds_no}</div></div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Edge calculation
        edge_rows = []
        for side, odds_str, label in [("yes", odds_yes_str, "Yes"), ("no", odds_no_str, "No")]:
            model_p = prob if side == "yes" else (1 - prob)
            book_p = american_to_implied(odds_str) if odds_str.strip() else None
            if book_p:
                e = edge(model_p, book_p)
                sign = "+" if e >= 0 else ""
                e_color = "#4affa3" if e > 0.02 else ("#ff4a6e" if e < -0.02 else "#aaa")
                verdict = "✅ Value" if e > 0.02 else ("❌ Fade" if e < -0.02 else "⚪ Neutral")
                edge_rows.append(
                    f"<div style='display:flex;justify-content:space-between;align-items:center;"
                    f"background:#0a0c10;border-radius:5px;padding:8px 12px;margin-bottom:4px'>"
                    f"<span style='font-size:12px;color:#aaa'>{label} &nbsp;"
                    f"<span style='color:#6b7591'>Book: {odds_str.strip()}</span></span>"
                    f"<span style='font-size:12px;color:{e_color}'>{sign}{e*100:.1f}pp &nbsp; {verdict}</span>"
                    f"</div>"
                )

        if edge_rows:
            st.markdown(
                "<div style='margin-top:4px'>"
                "<div style='font-size:10px;color:#4a5168;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px'>Edge vs. Book</div>"
                + "".join(edge_rows) + "</div>",
                unsafe_allow_html=True
            )

        # Key stats
        feat_display = {
            "K%": ("K%", lambda v: f"{v*100:.1f}%"),
            "BB%": ("BB%", lambda v: f"{v*100:.1f}%"),
            "K-BB%": ("K-BB%", lambda v: f"{v*100:.1f}%"),
            "xFIP": ("xFIP", lambda v: f"{v:.2f}"),
            "wOBA": ("wOBA", lambda v: f"{v:.3f}"),
            "BABIP": ("BABIP", lambda v: f"{v:.3f}"),
        }
        stat_html = ""
        for raw_key, (label, fmt) in feat_display.items():
            roll_key = f"roll_{raw_key}"
            val = s.get(roll_key, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                stat_html += (
                    f"<div style='text-align:center;background:#0a0c10;border-radius:5px;padding:8px'>"
                    f"<div style='font-size:9px;color:#4a5168;text-transform:uppercase;letter-spacing:1px'>{label}</div>"
                    f"<div style='font-size:14px;color:#cdd4e8'>{fmt(float(val))}</div>"
                    f"</div>"
                )

        if stat_html:
            st.markdown(
                "<div style='margin-top:10px'>"
                "<div style='font-size:10px;color:#4a5168;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px'>"
                "Rolling 10-start averages</div>"
                f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:6px'>{stat_html}</div>"
                "</div></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown("</div>", unsafe_allow_html=True)


# Render each matchup
for i, matchup in enumerate(st.session_state.matchups):
    st.markdown(f"#### {matchup['label']}")
    col_away, col_mid, col_home = st.columns([5, 1, 5])

    render_pitcher_card("Away Pitcher", matchup["away"], scores, meta, col_away)

    with col_mid:
        st.markdown("<div style='text-align:center;padding-top:60px;font-size:24px;color:#4a5168'>vs</div>", unsafe_allow_html=True)

    render_pitcher_card("Home Pitcher", matchup["home"], scores, meta, col_home)

    col_del, _ = st.columns([1, 6])
    with col_del:
        if st.button(f"Remove", key=f"del_{i}"):
            st.session_state.matchups.pop(i)
            st.rerun()

    st.divider()

# ─────────────────────────────────────────────
# PITCHER LOOKUP TABLE
# ─────────────────────────────────────────────
st.markdown("### 🔍 All Pitcher Probabilities")
st.caption("Full ranked list from the model based on your uploaded data.")

rolling_full = get_pitcher_rolling_stats(pit_df)
feat_cols = get_feature_cols()
valid = rolling_full.dropna(subset=feat_cols)

if not valid.empty:
    X_all = valid[feat_cols].values.astype(float)
    probs = meta["model"].predict_proba(X_all)[:, 1]
    valid = valid.copy()
    valid["Model Prob"] = probs
    valid["Implied Odds (Yes)"] = [prob_to_american(p) for p in probs]
    valid["Implied Odds (No)"] = [prob_to_american(1 - p) for p in probs]
    valid["vs Baseline"] = probs - meta["baseline"]
    valid["Actual Rate"] = valid["allowed_run"]

    display_cols = ["Name", "Tm", "Model Prob", "Implied Odds (Yes)", "Implied Odds (No)", "vs Baseline", "Actual Rate"]
    disp = valid[display_cols].sort_values("Model Prob", ascending=False).reset_index(drop=True)
    disp.index += 1

    # Search filter
    search = st.text_input("Search pitcher", placeholder="Type a name...", label_visibility="collapsed")
    if search:
        disp = disp[disp["Name"].str.lower().str.contains(search.lower())]

    # Format for display
    disp_fmt = disp.copy()
    disp_fmt["Model Prob"] = disp_fmt["Model Prob"].apply(lambda x: f"{x*100:.1f}%")
    disp_fmt["vs Baseline"] = disp_fmt["vs Baseline"].apply(lambda x: f"{'+' if x>=0 else ''}{x*100:.1f}pp")
    disp_fmt["Actual Rate"] = disp_fmt["Actual Rate"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    disp_fmt = disp_fmt.rename(columns={"Tm": "Team"})

    st.dataframe(disp_fmt, use_container_width=True, height=400)
