"""Visualization layer (stands in for Power BI / Tableau).  Run:  streamlit run dashboard.py"""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

OUT = Path(__file__).parent / "output"
st.set_page_config(page_title="Clinical Decision Support - Big Data Analytics", layout="wide")
st.title("Enhanced Clinical Decision-Making in Healthcare")
st.caption("Big Data Analytics on the Diabetes 130-US Hospitals dataset (101,766 encounters) - Spark + Parquet + MLlib + decision-support rules")

if not (OUT / "analytics" / "kpi.json").exists():
    st.error("No results found. Run `python run_pipeline.py` first.")
    st.stop()

kpi = json.load(open(OUT / "analytics" / "kpi.json"))
mt = json.load(open(OUT / "model" / "metrics.json"))
c = st.columns(5)
c[0].metric("Encounters", f"{int(kpi['encounters']):,}")
c[1].metric("Patients", f"{int(kpi['patients']):,}")
c[2].metric("30-day readmission", f"{kpi['readmission_rate_pct']}%")
c[3].metric("Avg length of stay", f"{kpi['avg_los']} days")
c[4].metric("Avg medications", f"{kpi['avg_meds']}")

def bar(df, x, y, title, **kw):
    return px.bar(df, x=x, y=y, title=title, labels={y: "readmission %"}, **kw)

def hbar(df, x, y, title):
    return px.bar(df, x=x, y=y, orientation="h", title=title).update_yaxes(autorange="reversed")

A = lambda n: pd.read_csv(OUT / "analytics" / f"{n}.csv")
t1, t2, t3, t4 = st.tabs(["Descriptive Analytics", "Predictive Model", "Clinical Decision Support", "Medications (MapReduce) & Fairness"])

with t1:
    a, b = st.columns(2)
    a.plotly_chart(bar(A("by_diag_group"), "diag_group", "readmission_pct", "Readmission by primary diagnosis group"), width="stretch")
    b.plotly_chart(bar(A("by_prior_inpatient"), "prior_inpatient_visits", "readmission_pct", "Readmission vs prior inpatient stays (past year)"), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(bar(A("by_a1c"), "a1c_grp", "readmission_pct", "Readmission by HbA1c result"), width="stretch")
    b.plotly_chart(bar(A("by_discharge"), "discharge_grp", "readmission_pct", "Readmission by discharge destination"), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(bar(A("by_age_group"), "age_group", "readmission_pct", "Readmission by age group"), width="stretch")
    b.plotly_chart(bar(A("by_los"), "length_of_stay", "readmission_pct", "Readmission by length of stay"), width="stretch")
    st.subheader("Correlation of numeric features with readmission")
    st.plotly_chart(hbar(A("correlations"), "correlation", "feature", ""), width="stretch")

with t2:
    st.write(f"**Best model:** {mt['best_model']}  |  train {mt['train_rows']:,} / test {mt['test_rows']:,} encounters  |  "
             f"patients in both sets: **{mt['patients_in_both_sets']}** (split by patient, so no leakage)")
    st.dataframe(pd.DataFrame(mt["models"]).T, width="stretch")
    a, b = st.columns(2)
    cm = mt["confusion_matrix"]
    cmdf = pd.DataFrame([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]], index=["Actual: no readmit", "Actual: readmit"], columns=["Flagged: no", "Flagged: yes"])
    a.plotly_chart(px.imshow(cmdf, text_auto=True, title=f"Confusion matrix ({mt['operating_point']})"), width="stretch")
    b.plotly_chart(hbar(pd.read_csv(OUT / "model" / "feature_importance.csv").head(12), "importance", "feature", "Top drivers of risk"), width="stretch")
    st.write(f"Flagging the top 20% of encounters catches **{mt['recall']*100:.0f}%** of readmissions (precision {mt['precision']*100:.0f}%). "
             f"The top 10% highest-risk encounters are readmitted **{mt['top_decile_lift']}x** more often than average "
             f"({mt['top_decile_rate']*100:.1f}% vs {mt['base_rate']*100:.1f}%).")

with t3:
    rl = pd.read_csv(OUT / "decision_support" / "risk_levels.csv")
    st.plotly_chart(bar(rl, "risk_level", "actual_readmission_pct", "Observed readmission rate per risk level (held-out patients)",
                        category_orders={"risk_level": ["LOW", "MEDIUM", "HIGH"]}), width="stretch")
    pr = pd.read_csv(OUT / "decision_support" / "priority_patients.csv")
    f1, f2 = st.columns(2)
    dx = f1.multiselect("Primary diagnosis group", sorted(pr.diag_group.unique()))
    minr = f2.slider("Minimum risk score", float(pr.risk_score.min()), float(pr.risk_score.max()), float(pr.risk_score.quantile(0.5)), 0.01)
    view = pr[(pr.risk_score >= minr) & (pr.diag_group.isin(dx) if dx else True)]
    st.write(f"{len(view)} priority encounters")
    st.dataframe(view[["patient_nbr", "diag_group", "age_mid", "risk_score", "risk_level", "n_alerts", "alerts_text", "recommended_action"]], width="stretch")
    st.info("Decision *support* only. Rule thresholds are illustrative and need clinician validation. Scores assist, never replace, clinical judgement.")

with t4:
    dr = A("drug_usage_mapreduce")
    tot = dr.groupby("drug")["count"].sum().sort_values(ascending=False).head(10).index
    st.plotly_chart(px.bar(dr[dr.drug.isin(tot)], x="count", y="drug", color="status", orientation="h",
                           title="Drug usage by dose status (MapReduce over medication columns)").update_yaxes(autorange="reversed"), width="stretch")
    st.subheader("Fairness check: is any group flagged far more or less than its actual rate suggests? (race is NOT a model input)")
    a, b = st.columns(2)
    for col, box in [("race", a), ("gender", b)]:
        f = pd.read_csv(OUT / "decision_support" / f"fairness_by_{col}.csv").melt(id_vars=[col, "encounters"], var_name="metric", value_name="percent")
        box.plotly_chart(px.bar(f, x=col, y="percent", color="metric", barmode="group", title=f"Share flagged by model vs observed readmission, by {col}"), width="stretch")
