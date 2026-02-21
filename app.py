"""
app.py — First Inning Run Probability Tool
Streamlit Community Cloud ready — upload CSVs, get probabilities + odds.
"""

import streamlit as st
import pandas as pd
import numpy as np

from model import (
    train_model,
    get_pitcher_rolling_stats,
    score_pitchers,
    prob_to_american,
    american_to_implied,
    edge,
    PITCHER_FEATURE_COLS,
    get_feature_cols,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="1st Inning Run Probability",
    page_icon="⚾",
    layout="wide",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');
  .block-container { padding-top: 2rem; }
  .prob-high { color: #ff4a6e; }
  .prob-mid  { color: #ffa54a; }
  .prob-low  { color: #4affa3; }
  div[data-testid="metric-container"] {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 12px 16px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
REQUIRED_PIT_COLS = {
    "Date", "Name", "Tm", "R", "K%", "BB%", "K-BB%",
    "wOBA", "xFIP", "BABIP", "GB%", "LD%", "FB%", "WHIP"
}

def load_upload(f):
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    return pd.read_excel(f)

def validate(df, required, label):
    missing = required - set(df.columns)
    if missing:
        st.error(f"**{label}** missing columns: {', '.join(sorted(missing))}")
        return False
    return True

def prob_color(p):
    if p >= 0.40: return "#ff4a6e"
    if p >= 0.28: return "#ffa54a"
    return "#4affa3"

def american_to_implied_safe(s):
    try:
        s = s.strip()
        if not s: return None
        v = int(s.replace("+", ""))
        return 100 / (v + 100) if v > 0 else abs(v) / (abs(v) + 100)
    except Exception:
        return None


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in [
    ("model_meta", None),
    ("pit_df", None),
    ("matchups", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("## ⚾ First Inning Run Probability")
st.caption(
    "Upload your FanGraphs first-inning pitching splits CSV each day. "
    "The model trains automatically, then build matchups to see probabilities and implied odds."
)
st.divider()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Upload Data")
    st.caption(
        "FanGraphs → Pitchers → Splits → Innings Pitched → First Inning → Export CSV"
    )

    pit_file = st.file_uploader(
        "Pitching — First Inning Splits",
        type=["csv", "xlsx"],
        key="pit_upload",
    )

    if pit_file:
        try:
            raw = load_upload(pit_file)
            if validate(raw, REQUIRED_PIT_COLS, "Pitching file"):
                for c in PITCHER_FEATURE_COLS + ["R"]:
                    if c in raw.columns:
                        raw[c] = pd.to_numeric(raw[c], errors="coerce")
                raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
                st.session_state.pit_df = raw

                # Auto-train on upload
                with st.spinner("Training model..."):
                    meta = train_model(raw)
                    st.session_state.model_meta = meta

                st.success(
                    f"✓ Ready — {raw['Name'].nunique()} pitchers, "
                    f"{meta['n_starts']:,} starts\n\n"
                    f"League avg: **{meta['baseline']*100:.1f}%**"
                )
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Feature importance
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
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px'>"
                f"<span style='font-size:11px;color:#6b7591;width:72px;text-align:right;flex-shrink:0'>{feat}</span>"
                f"<div style='flex:1;background:#1e2330;height:6px;border-radius:3px'>"
                f"<div style='width:{pct*100:.0f}%;background:#e8ff4a;height:100%;border-radius:3px'></div></div>"
                f"<span style='font-size:11px;color:#4a5168;width:32px'>{val*100:.1f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("### ℹ️ How to read edge")
        st.markdown("""
**Edge** = model probability minus book's implied probability.

- `+5pp` → model sees value, bet the Yes  
- `0pp` → model agrees with the book  
- `-5pp` → book is worse than fair value  

Rule of thumb: anything beyond **±3pp** is meaningful.
        """)


# ─────────────────────────────────────────────
# GATE: need data
# ─────────────────────────────────────────────
if st.session_state.model_meta is None:
    st.info("👈 Upload your FanGraphs pitching splits CSV in the sidebar to get started.")
    st.stop()

meta = st.session_state.model_meta
pit_df = st.session_state.pit_df
known_pitchers = sorted(
    get_pitcher_rolling_stats(pit_df)["Name"].dropna().unique().tolist()
)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_matchups, tab_rankings, tab_guide = st.tabs(["🎯 Matchups", "📋 All Pitchers", "📖 How to Use"])


# ══════════════════════════════════════════════
# TAB 1 — MATCHUPS
# ══════════════════════════════════════════════
with tab_matchups:

    # Add matchup form
    with st.expander("➕ Add a Matchup", expanded=len(st.session_state.matchups) == 0):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Away Pitcher**")
            away_p = st.selectbox("Away", [""] + known_pitchers, key="f_away", label_visibility="collapsed")
            st.markdown("Book odds for this pitcher's 1st inning:")
            a_yes = st.text_input("Yes (run scores)", placeholder="+120", key="f_a_yes")
            a_no  = st.text_input("No (run doesn't score)", placeholder="-150", key="f_a_no")

        with c2:
            st.markdown("**Home Pitcher**")
            home_p = st.selectbox("Home", [""] + known_pitchers, key="f_home", label_visibility="collapsed")
            st.markdown("Book odds for this pitcher's 1st inning:")
            h_yes = st.text_input("Yes (run scores)", placeholder="+140", key="f_h_yes")
            h_no  = st.text_input("No (run doesn't score)", placeholder="-170", key="f_h_no")

        label = st.text_input("Game label", placeholder="e.g.  NYY @ BOS  7:10 ET", key="f_label")

        if st.button("Add Matchup ➕", type="primary"):
            if not away_p and not home_p:
                st.warning("Select at least one pitcher.")
            else:
                st.session_state.matchups.append({
                    "label": label or f"{away_p or '?'} @ {home_p or '?'}",
                    "away": {"name": away_p, "yes": a_yes, "no": a_no},
                    "home": {"name": home_p, "yes": h_yes, "no": h_no},
                })
                st.rerun()

    if not st.session_state.matchups:
        st.markdown("No matchups yet — add one above.")
        st.stop()

    # Clear all
    if st.button("🗑 Clear all matchups"):
        st.session_state.matchups = []
        st.rerun()

    # Batch score all pitchers
    all_names = list({
        p["name"]
        for m in st.session_state.matchups
        for p in [m["away"], m["home"]]
        if p["name"]
    })
    scores_df = score_pitchers(meta, all_names, pit_df)
    scores = {r["Name"]: r for _, r in scores_df.iterrows()} if not scores_df.empty else {}

    st.divider()

    # ── Render each matchup ──
    for i, m in enumerate(st.session_state.matchups):
        st.markdown(f"#### {m['label']}")
        col_a, col_vs, col_h = st.columns([5, 1, 5])

        def render_card(col, side_label, pitcher_info):
            name = pitcher_info["name"]
            odds_yes = pitcher_info["yes"]
            odds_no  = pitcher_info["no"]

            with col:
                st.markdown(f"**{side_label}**")
                if not name:
                    st.markdown("*No pitcher selected*")
                    return

                s = scores.get(name, {})
                if not s.get("found"):
                    reason = s.get("reason", "not found in data")
                    st.warning(f"{name} — {reason}")
                    return

                prob = float(s["prob"])
                baseline = meta["baseline"]
                color = prob_color(prob)
                diff = prob - baseline
                diff_sign = "+" if diff >= 0 else ""
                diff_color = "#ff4a6e" if diff > 0.03 else ("#4affa3" if diff < -0.03 else "#aaa")

                model_yes = prob_to_american(prob)
                model_no  = prob_to_american(1 - prob)

                # Main card
                st.markdown(
                    f"<div style='background:#111318;border:1px solid #1e2330;border-radius:10px;padding:20px'>"

                    # Name + team
                    f"<div style='font-size:13px;color:#aaa;margin-bottom:6px'>"
                    f"{name}&nbsp;&nbsp;"
                    f"<span style='background:#1e2330;padding:2px 8px;border-radius:3px;"
                    f"font-size:11px;color:#6b7591'>{s.get('Team','')}</span></div>"

                    # Big probability
                    f"<div style='font-size:48px;font-weight:700;color:{color};line-height:1'>"
                    f"{prob*100:.0f}%</div>"
                    f"<div style='font-size:12px;color:#6b7591;margin-bottom:6px'>"
                    f"chance of allowing a 1st inning run</div>"

                    # vs baseline
                    f"<div style='font-size:12px;color:{diff_color};margin-bottom:16px'>"
                    f"{diff_sign}{diff*100:.1f}pp vs league avg ({baseline*100:.1f}%)</div>"

                    # Model odds row
                    f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px'>"
                    f"<div style='background:#0a0c10;border-radius:6px;padding:10px'>"
                    f"<div style='font-size:9px;color:#4a5168;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>Model — Yes</div>"
                    f"<div style='font-size:22px;font-weight:600;color:#e8ff4a'>{model_yes}</div></div>"
                    f"<div style='background:#0a0c10;border-radius:6px;padding:10px'>"
                    f"<div style='font-size:9px;color:#4a5168;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>Model — No</div>"
                    f"<div style='font-size:22px;font-weight:600;color:#e8ff4a'>{model_no}</div></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Edge rows
                edge_html = ""
                for side_key, book_str, label_str, model_p in [
                    ("yes", odds_yes, "Yes — run scores",       prob),
                    ("no",  odds_no,  "No — run doesn't score", 1 - prob),
                ]:
                    book_imp = american_to_implied_safe(book_str)
                    if book_imp:
                        e_val = model_p - book_imp
                        e_sign = "+" if e_val >= 0 else ""
                        e_color = "#4affa3" if e_val > 0.02 else ("#ff4a6e" if e_val < -0.02 else "#888")
                        verdict = "✅ Value" if e_val > 0.02 else ("❌ Avoid" if e_val < -0.02 else "⚪ Neutral")
                        edge_html += (
                            f"<div style='display:flex;justify-content:space-between;"
                            f"background:#0a0c10;border-radius:5px;padding:9px 12px;margin-bottom:4px'>"
                            f"<span style='font-size:12px;color:#aaa'>{label_str}"
                            f"&nbsp;<span style='color:#6b7591'>({book_str.strip()})</span></span>"
                            f"<span style='font-size:12px;color:{e_color}'>"
                            f"{e_sign}{e_val*100:.1f}pp &nbsp; {verdict}</span></div>"
                        )

                if edge_html:
                    st.markdown(
                        f"<div style='background:#111318;border-radius:10px;padding:0'>"
                        f"<div style='font-size:9px;color:#4a5168;text-transform:uppercase;"
                        f"letter-spacing:1px;margin-bottom:6px'>Edge vs. Book</div>"
                        f"{edge_html}</div>",
                        unsafe_allow_html=True,
                    )

                # Key rolling stats
                stat_map = {
                    "K%":    ("K%",    lambda v: f"{v*100:.1f}%"),
                    "BB%":   ("BB%",   lambda v: f"{v*100:.1f}%"),
                    "K-BB%": ("K-BB%", lambda v: f"{v*100:.1f}%"),
                    "xFIP":  ("xFIP",  lambda v: f"{v:.2f}"),
                    "wOBA":  ("wOBA",  lambda v: f"{v:.3f}"),
                    "BABIP": ("BABIP", lambda v: f"{v:.3f}"),
                }
                stat_html = ""
                for raw_k, (lbl, fmt) in stat_map.items():
                    val = s.get(f"roll_{raw_k}")
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        stat_html += (
                            f"<div style='text-align:center;background:#0a0c10;"
                            f"border-radius:5px;padding:8px'>"
                            f"<div style='font-size:9px;color:#4a5168;text-transform:uppercase;"
                            f"letter-spacing:1px'>{lbl}</div>"
                            f"<div style='font-size:14px;color:#cdd4e8'>{fmt(float(val))}</div></div>"
                        )

                if stat_html:
                    st.markdown(
                        f"<div style='margin-top:10px'>"
                        f"<div style='font-size:9px;color:#4a5168;text-transform:uppercase;"
                        f"letter-spacing:1px;margin-bottom:6px'>10-start rolling averages</div>"
                        f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:6px'>"
                        f"{stat_html}</div></div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

        render_card(col_a, "Away", m["away"])
        with col_vs:
            st.markdown(
                "<div style='text-align:center;padding-top:80px;font-size:20px;color:#4a5168'>vs</div>",
                unsafe_allow_html=True,
            )
        render_card(col_h, "Home", m["home"])

        col_del, _ = st.columns([1, 7])
        with col_del:
            if st.button("Remove", key=f"del_{i}"):
                st.session_state.matchups.pop(i)
                st.rerun()

        st.divider()


# ══════════════════════════════════════════════
# TAB 2 — ALL PITCHERS
# ══════════════════════════════════════════════
with tab_rankings:
    st.markdown("### All Pitcher Probabilities")
    st.caption("Full ranked list. Probabilities based on rolling 10-start averages from your uploaded data.")

    rolling = get_pitcher_rolling_stats(pit_df)
    feat_cols = get_feature_cols()
    valid = rolling.dropna(subset=feat_cols).copy()

    if valid.empty:
        st.warning("Not enough data to rank pitchers yet.")
    else:
        probs = meta["model"].predict_proba(valid[feat_cols].values.astype(float))[:, 1]
        valid["prob"] = probs
        valid["Model Prob"] = [f"{p*100:.1f}%" for p in probs]
        valid["Yes Odds"] = [prob_to_american(p) for p in probs]
        valid["No Odds"]  = [prob_to_american(1-p) for p in probs]
        valid["vs Avg"]   = [f"{'+'if p>=meta['baseline'] else ''}{(p-meta['baseline'])*100:.1f}pp" for p in probs]

        # Filters
        fc1, fc2, fc3 = st.columns([2, 2, 2])
        with fc1:
            search = st.text_input("Search pitcher", placeholder="Name...", label_visibility="visible")
        with fc2:
            teams = ["All"] + sorted(valid["Tm"].dropna().unique().tolist())
            team_filter = st.selectbox("Team", teams)
        with fc3:
            min_starts = st.slider("Min starts in data", 1, 30, 5)

        # Apply filters
        start_counts = pit_df.groupby("Name").size().reset_index(name="starts")
        valid = valid.merge(start_counts, on="Name", how="left")
        mask = valid["starts"] >= min_starts
        if search:
            mask &= valid["Name"].str.lower().str.contains(search.lower())
        if team_filter != "All":
            mask &= valid["Tm"] == team_filter
        filtered = valid[mask].sort_values("prob", ascending=False).reset_index(drop=True)
        filtered.index += 1

        st.caption(f"{len(filtered)} pitchers shown")
        st.dataframe(
            filtered[["Name", "Tm", "Model Prob", "Yes Odds", "No Odds", "vs Avg", "starts"]].rename(
                columns={"Tm": "Team", "starts": "Starts"}
            ),
            use_container_width=True,
            height=500,
        )


# ══════════════════════════════════════════════
# TAB 3 — HOW TO USE
# ══════════════════════════════════════════════
with tab_guide:
    st.markdown("""
### Daily Workflow

**Step 1 — Download from FanGraphs** (takes ~2 minutes)

1. Go to **fangraphs.com**
2. Click **Pitchers** at the top → **Splits**
3. Set Split Type: **Innings Pitched** → **First Inning**
4. Set the season to the current year, group by **Player**
5. Click the **Export Data** button (bottom of page)
6. Save the CSV somewhere easy to find

**Step 2 — Upload and train**

1. Come back to this app
2. Click **Browse files** in the left sidebar
3. Upload the CSV you just downloaded
4. The model trains automatically in a few seconds

**Step 3 — Build matchups**

1. Go to the **Matchups** tab
2. Click **Add a Matchup**
3. Select the away and home starting pitcher
4. Optionally type in the sportsbook's American odds (e.g. `+130`, `-150`)
5. Click **Add Matchup**

---

### Understanding the Output

| Term | What it means |
|---|---|
| **Model Prob** | The model's estimated % chance the pitcher allows ≥1 run in the 1st inning |
| **Model Odds** | That probability expressed as American odds |
| **vs League Avg** | How much higher or lower this pitcher is vs the ~29.5% league baseline |
| **Edge** | Your advantage vs the book. Positive = model sees value on that side |
| **Rolling stats** | The 10-start rolling averages the model used to make this prediction |

---

### What makes a good bet signal?

The model is most confident when:
- Edge is **+4pp or more** on one side
- The pitcher has **10+ starts** of data (rolling averages are more stable)
- Model probability is **meaningfully different** from 29.5% baseline — not just noise

Don't use single-game results to judge the model. Evaluate over 50+ bets.

---

### Required CSV columns

Your FanGraphs export must contain these columns:
`Date, Name, Tm, R, K%, BB%, K-BB%, wOBA, xFIP, BABIP, GB%, LD%, FB%, WHIP`

If you get a "missing columns" error, make sure you're on the **Pitchers → Splits → First Inning** view, not the standard leaderboard.
    """)
