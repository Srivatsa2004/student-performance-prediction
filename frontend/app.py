"""Streamlit frontend for the ANN + Fuzzy Student Performance Prediction system.

All intelligence lives in the FastAPI backend; this app only calls its REST API.
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

GRADE_COLORS = {"A": "#16a34a", "B": "#65a30d", "C": "#ca8a04", "D": "#ea580c", "F": "#dc2626"}
RISK_ICONS = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}


# --------------------------------------------------------------------------- API
def api(method: str, path: str, quiet: bool = False, timeout: int = 120, **kwargs):
    """Call the backend. Returns parsed JSON or None on failure."""
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.RequestException as exc:
        if not quiet:
            st.error(f"Cannot reach the backend at {API_URL}: {exc}")
        return None
    if r.status_code >= 400:
        if not quiet:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            st.error(f"API error {r.status_code}: {detail}")
        return None
    return r.json()


# ----------------------------------------------------------------------- helpers
def gauge(score: float, grade: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100", "font": {"size": 34}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": GRADE_COLORS.get(grade, "#2563eb"), "thickness": 0.35},
            "steps": [
                {"range": [0, 40], "color": "rgba(220,38,38,0.18)"},
                {"range": [40, 55], "color": "rgba(234,88,12,0.18)"},
                {"range": [55, 70], "color": "rgba(202,138,4,0.18)"},
                {"range": [70, 85], "color": "rgba(101,163,13,0.18)"},
                {"range": [85, 100], "color": "rgba(22,163,74,0.22)"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def pretty(s: str) -> str:
    return s.replace("_", " ").title()


def render_result(res: dict):
    name = res.get("student_name") or "Student"
    st.subheader(f"Result for {name}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final score", f"{res['final_score']:.1f}")
    c2.metric("Grade", f"{res['grade']} · {res['category']}")
    c3.metric("Risk level", f"{RISK_ICONS[res['risk_level']]} {res['risk_level']}")
    c4.metric("Pass prediction", "Pass ✅" if res["will_pass"] else "At risk ❌")

    st.plotly_chart(gauge(res["final_score"], res["grade"]))

    a, b, c = st.columns(3)
    a.metric("🧠 ANN prediction", f"{res['ann_score']:.1f}", help="Numeric estimate from the neural network")
    b.metric("🌫️ Fuzzy index", f"{res['fuzzy_score']:.1f}", f"{pretty(res['fuzzy_label'])}", delta_color="off",
             help="Linguistic rule-based assessment")
    c.metric("🤝 Model agreement", f"{res['model_agreement']:.0f}%",
             help="How close the ANN and fuzzy scores are")

    st.markdown("#### 💡 Recommendations")
    for tip in res["recommendations"]:
        st.markdown(f"- {tip}")

    with st.expander("🔍 Why? Fuzzy reasoning details"):
        fd = res["fuzzy_details"]
        rows = [(v, t, d) for v, terms in fd["input_memberships"].items() for t, d in terms.items()]
        mdf = pd.DataFrame(rows, columns=["variable", "term", "membership"])
        mdf["variable"] = mdf["variable"].map(pretty)
        fig = px.bar(mdf, x="variable", y="membership", color="term", barmode="group",
                     title="How strongly each input belongs to each fuzzy set", range_y=[0, 1])
        fig.update_layout(height=340, xaxis_title=None)
        st.plotly_chart(fig)

        agg = fd["aggregated"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=agg["x"], y=agg["y"], fill="tozeroy", name="Aggregated output"))
        fig2.add_vline(x=res["fuzzy_score"], line_dash="dash", annotation_text=f"centroid = {res['fuzzy_score']:.1f}")
        fig2.update_layout(title="Aggregated output fuzzy set → centroid defuzzification", height=300,
                           xaxis_title="Performance index", yaxis_title="Membership", yaxis_range=[0, 1.05])
        st.plotly_chart(fig2)

        st.markdown(f"**Rules fired ({len(fd['fired_rules'])})**")
        rdf = pd.DataFrame(fd["fired_rules"])[["rule", "strength"]].head(12)
        st.dataframe(rdf, hide_index=True)


# ------------------------------------------------------------------------- pages
def page_predict(info: dict):
    st.header("🎯 Predict student performance")
    meta = info["feature_meta"]
    left, right = st.columns([1, 1.5], gap="large")

    with left:
        with st.form("predict_form"):
            name = st.text_input("Student name (optional)")
            vals = {}
            for f in info["features"]:
                m = meta[f]
                vals[f] = st.slider(m["label"], float(m["min"]), float(m["max"]),
                                    float(m["default"]), float(m["step"]))
            submitted = st.form_submit_button("🚀 Predict", type="primary")

    if submitted:
        res = api("POST", "/predict", json={"student_name": name or None, **vals})
        if res:
            st.session_state["last_result"] = res

    with right:
        res = st.session_state.get("last_result")
        if res:
            render_result(res)
        else:
            st.info("Set the student's details on the left and press **Predict**.")


def page_batch(info: dict):
    st.header("📂 Batch prediction")
    meta, features = info["feature_meta"], info["features"]
    st.write("Upload a CSV with columns: " + ", ".join(f"`{f}`" for f in features) +
             " (optional: `student_name`).")

    c1, c2 = st.columns(2)
    sample = api("GET", "/sample-data", params={"n": 20}, quiet=True)
    if sample:
        tpl = pd.DataFrame(sample["records"]).drop(columns=["final_score"], errors="ignore")
        c1.download_button("⬇️ Download sample CSV", tpl.to_csv(index=False), "students_sample.csv", "text/csv")
    use_sample = c2.checkbox("Use generated sample data instead of uploading")

    df = None
    if use_sample and sample:
        df = pd.DataFrame(sample["records"]).drop(columns=["final_score"], errors="ignore")
    else:
        up = st.file_uploader("CSV file", type=["csv"])
        if up is not None:
            try:
                df = pd.read_csv(up)
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")

    if df is None:
        return
    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return

    for f in features:  # coerce + clip to allowed ranges
        df[f] = pd.to_numeric(df[f], errors="coerce").clip(meta[f]["min"], meta[f]["max"])
    before = len(df)
    df = df.dropna(subset=features).reset_index(drop=True)
    if len(df) < before:
        st.warning(f"Dropped {before - len(df)} rows with missing/non-numeric values.")
    if "student_name" not in df.columns:
        df.insert(0, "student_name", [f"Student {i + 1}" for i in range(len(df))])
    st.dataframe(df.head(10), hide_index=True)

    if st.button("▶️ Run batch prediction", type="primary"):
        results = []
        with st.spinner("Predicting..."):
            for i in range(0, len(df), 1000):
                chunk = df.iloc[i:i + 1000]
                payload = [{"student_name": str(r["student_name"]),
                            **{f: float(r[f]) for f in features}} for _, r in chunk.iterrows()]
                out = api("POST", "/predict/batch", json={"students": payload})
                if out is None:
                    return
                results += out["results"]
        st.session_state["batch_df"] = pd.DataFrame([
            {"student_name": r["student_name"], **r["inputs"], "ann_score": r["ann_score"],
             "fuzzy_score": r["fuzzy_score"], "final_score": r["final_score"], "grade": r["grade"],
             "category": r["category"], "risk_level": r["risk_level"]} for r in results])

    out = st.session_state.get("batch_df")
    if out is not None:
        st.success(f"Predicted {len(out)} students.")
        k1, k2, k3 = st.columns(3)
        k1.metric("Average final score", f"{out['final_score'].mean():.1f}")
        k2.metric("High-risk students", int((out["risk_level"] == "High").sum()))
        k3.metric("Predicted pass rate", f"{(out['final_score'] >= 40).mean() * 100:.0f}%")
        g1, g2 = st.columns(2)
        g1.plotly_chart(px.histogram(out, x="final_score", nbins=20, title="Final score distribution"))
        counts = out["category"].value_counts().reindex(
            ["Excellent", "Good", "Average", "Below Average", "Poor"]).fillna(0).reset_index()
        counts.columns = ["category", "students"]
        g2.plotly_chart(px.bar(counts, x="category", y="students", title="Performance categories"))
        st.dataframe(out.sort_values("final_score"), hide_index=True)
        st.download_button("⬇️ Download results CSV", out.to_csv(index=False), "predictions.csv", "text/csv")


def page_accuracy(info: dict):
    st.header("📊 Model accuracy")
    model = info["model"]
    mt = model["metrics"]
    st.write("These numbers come from testing the neural network on students it had **never seen** "
             "during training.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model fit (R²)", f"{mt['r2'] * 100:.1f}%",
              help="Share of the differences between students' scores that the model explains. 100% is perfect.")
    c2.metric("Average error (MAE)", f"± {mt['mae']:.2f} points",
              help="On average, how far the prediction is from the real score (out of 100).")
    c3.metric("Large-error measure (RMSE)", f"{mt['rmse']:.2f} points",
              help="Like the average error, but penalises big misses more.")

    st.info(f"On average, the model's score prediction is within about **{mt['mae']:.1f} points** "
            f"of the real score (out of 100).")
    st.caption(f"Trained on {model['n_samples']} students ({model['n_train']} for training, "
               f"{model['n_test']} for testing) · last trained {model['trained_at']}.")


def page_train(info: dict):
    st.header("⚙️ Train / retrain the ANN")
    features = info["features"]
    st.write("Retrain the neural network on synthetic data or on your own CSV. "
             f"Required columns: {', '.join(f'`{f}`' for f in features)}, `final_score` (0-100).")

    source = st.radio("Training data", ["Synthetic (generated)", "My own CSV"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    n_samples = c1.number_input("Synthetic samples", 200, 50000, 3000, 500,
                                disabled=source != "Synthetic (generated)")
    layers_txt = c2.text_input("Hidden layers (comma separated)", "64,32")
    max_iter = c3.number_input("Max epochs", 50, 3000, 600, 50)

    records = None
    if source == "My own CSV":
        up = st.file_uploader("Training CSV", type=["csv"])
        if up is not None:
            try:
                df = pd.read_csv(up)
                st.write(f"Loaded {len(df)} rows.")
                st.dataframe(df.head(5), hide_index=True)
                records = df.to_dict(orient="records")
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")

    if st.button("🏋️ Train model", type="primary"):
        try:
            layers = [int(x) for x in layers_txt.split(",") if x.strip()]
            assert layers
        except Exception:
            st.error("Hidden layers must be integers like `64,32`.")
            return
        if source == "My own CSV" and not records:
            st.error("Please upload a CSV first.")
            return
        payload = {"source": "custom" if records else "synthetic", "n_samples": int(n_samples),
                   "hidden_layers": layers, "max_iter": int(max_iter), "records": records}
        with st.spinner("Training... this can take a little while."):
            out = api("POST", "/train", json=payload, timeout=900)
        if out:
            m = out["model"]["metrics"]
            st.success(f"{out['message']}  R² = {m['r2']:.3f}, MAE = {m['mae']:.2f}, RMSE = {m['rmse']:.2f}")
            st.session_state.pop("last_result", None)
            st.balloons()


def page_history(info: dict):
    st.header("🕘 Prediction history")
    out = api("GET", "/history", params={"limit": 500})
    if not out:
        return
    df = pd.DataFrame(out["records"])
    if df.empty:
        st.info("No predictions recorded yet. Make one on the **Predict** page.")
        return
    st.dataframe(df, hide_index=True)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download CSV", df.to_csv(index=False), "history.csv", "text/csv")
    if c2.button("🗑️ Clear history"):
        d = api("DELETE", "/history")
        if d:
            st.success(f"Deleted {d['deleted']} records.")
            st.rerun()


# -------------------------------------------------------------------------- main
PAGES = {
    "🎯 Predict": page_predict,
    "📂 Batch prediction": page_batch,
    "📊 Model accuracy": page_accuracy,
    "⚙️ Train model": page_train,
    "🕘 History": page_history,
}


def main():
    st.sidebar.title("🎓 Student Performance")
    st.sidebar.caption("ANN + Fuzzy Logic hybrid predictor")

    info = api("GET", "/model/info", quiet=True)
    if info is None:
        st.sidebar.error("Backend offline")
        st.error(f"The backend is not reachable at **{API_URL}**.")
        st.markdown("Start it with:\n```bash\nuvicorn backend.main:app --port 8000\n```\n"
                    "or run everything at once with `python run.py`.")
        st.stop()

    page = st.sidebar.radio("Navigate", list(PAGES))
    st.sidebar.success("Backend connected")
    if info["model_ready"] and info["model"]:
        st.sidebar.caption(f"Model R² = {info['model']['metrics']['r2']:.3f} · "
                           f"MAE = {info['model']['metrics']['mae']:.2f}")
    st.sidebar.caption(f"API: {API_URL}")

    PAGES[page](info)


main()
