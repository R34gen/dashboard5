import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =========================================================
# Olist Customer Intelligence & Business Performance Dashboard
# Scope: BI, RFM-based customer prioritization, delivery experience,
# and evidence-based business recommendation.
# =========================================================

st.set_page_config(
    page_title="Olist Customer Intelligence",
    page_icon="◈",
    layout="wide",
)

DATA_DIRS = [Path("olist_output"), Path(".")]

FILES = {
    "audit": ["audit_df.csv", "audit_df(1).csv"],
    "kpi": ["kpi.csv", "kpi(1).csv"],
    "monthly": ["monthly_kpi.csv", "monthly_kpi(1).csv"],
    "category": ["category_kpi.csv", "category_kpi(1).csv"],
    "state": ["state_kpi.csv"],
    "customer_profile": ["customer_profile.csv", "customer_profile(1).csv"],
    "rfm_diag": ["rfm_diagnostic.csv", "rfm_diagnostic(1).csv"],
    "segment": ["segment_priority.csv", "segment_priority(1).csv"],
    "delivery": ["delivery_experience.csv", "delivery_experience(1).csv"],
    "insight": ["insight_table.csv", "insight_table(1).csv"],
    "storytelling": ["storytelling_map.csv"],
}

REQUIRED = ["kpi", "monthly", "category", "state", "segment", "delivery"]


@st.cache_data(show_spinner=False)
def read_first_available(names):
    for folder in DATA_DIRS:
        for name in names:
            path = folder / name
            if path.exists():
                return pd.read_csv(path), str(path)
    return pd.DataFrame(), None


@st.cache_data(show_spinner=False)
def load_data():
    data, used, missing = {}, {}, []
    for key, names in FILES.items():
        df, path = read_first_available(names)
        data[key], used[key] = df, path
        if path is None and key in REQUIRED:
            missing.append(" / ".join(names))
    return data, used, missing


data, used_files, missing_files = load_data()

audit = data["audit"]
kpi = data["kpi"]
monthly = data["monthly"]
category = data["category"]
state = data["state"]
customer_profile = data["customer_profile"]
rfm_diag = data["rfm_diag"]
segment = data["segment"]
delivery = data["delivery"]
insight = data["insight"]
storytelling = data["storytelling"]


def to_num(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def clean_label(x):
    return str(x).replace("_", " ").replace("-", " ").strip().title()


def fmt_num(x):
    x = to_num(x)
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:,.0f}"


def fmt_money(x):
    x = to_num(x)
    if abs(x) >= 1_000_000:
        return f"R$ {x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"R$ {x/1_000:.1f}K"
    return f"R$ {x:,.0f}"


def fmt_pct(x):
    return f"{to_num(x) * 100:.1f}%"


def fmt_float(x):
    return f"{to_num(x):.2f}"


def metric_from(df, metric, default=0.0):
    if df.empty or not {"metric", "value"}.issubset(df.columns):
        return default
    hit = df.loc[df["metric"].astype(str).eq(metric), "value"]
    return to_num(hit.iloc[0], default) if not hit.empty else default


def metric_any(metric, default=0.0):
    for df in [kpi, rfm_diag, customer_profile]:
        val = metric_from(df, metric, None)
        if val is not None:
            return val
    return default


def ensure_numeric(df, cols):
    if df.empty:
        return df
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def style_fig(fig, height=390):
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB", family="Inter, Arial, sans-serif"),
        margin=dict(l=18, r=18, t=48, b=24),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.14)", zerolinecolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.14)", zerolinecolor="rgba(148,163,184,0.18)")
    return fig


def metric_card(label, value, note, variant="primary"):
    return f"""
    <div class="metric-card {variant}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def rank_card(no, name, caption, value=""):
    return f"""
    <div class="rank-card">
        <div class="rank-left">
            <div class="rank-no">{no}</div>
            <div>
                <div class="rank-name">{name}</div>
                <div class="rank-caption">{caption}</div>
            </div>
        </div>
        <div class="rank-value">{value}</div>
    </div>
    """


def insight_card(title, tag, evidence, interpretation, action, kpi_monitoring):
    return f"""
    <div class="insight-card">
        <span class="tag">{tag}</span>
        <h3>{title}</h3>
        <p><b>Evidence:</b> {evidence}</p>
        <p><b>Interpretation:</b> {interpretation}</p>
        <p class="green"><b>Action:</b> {action}</p>
        <p class="muted"><b>KPI monitoring:</b> {kpi_monitoring}</p>
    </div>
    """


def panel(title, subtitle=""):
    st.markdown(
        f"""
        <div class="panel-head">
            <div class="panel-title">{title}</div>
            <div class="panel-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- Data cleaning for dashboard use ----------
monthly = ensure_numeric(monthly, ["orders", "gmv", "avg_review", "late_rate"])
category = ensure_numeric(category, ["orders", "gmv", "avg_review", "late_rate"])
state = ensure_numeric(state, ["orders", "gmv", "avg_delivery_days", "late_rate"])
segment = ensure_numeric(
    segment,
    [
        "customers",
        "avg_recency",
        "avg_frequency",
        "avg_monetary",
        "total_monetary",
        "customer_share",
        "revenue_share",
        "priority_score",
    ],
)
delivery = ensure_numeric(delivery, ["orders", "avg_review", "low_review_rate", "avg_delivery_days"])

if not monthly.empty and "order_month" in monthly.columns:
    monthly["order_month"] = pd.to_datetime(monthly["order_month"].astype(str), errors="coerce")
    monthly = monthly.dropna(subset=["order_month"]).sort_values("order_month")

if not category.empty and "main_category" in category.columns:
    category["category_label"] = category["main_category"].apply(clean_label)
    category = category.sort_values("gmv", ascending=False)

if not state.empty and "customer_state" in state.columns:
    state["state_label"] = state["customer_state"].astype(str).str.upper()
    state = state.sort_values("gmv", ascending=False)

if not segment.empty and "segment" in segment.columns:
    segment["segment_label"] = segment["segment"].apply(clean_label)

if not delivery.empty and "delivery_status" in delivery.columns:
    delivery["delivery_status"] = delivery["delivery_status"].astype(str)

# ---------- CSS ----------
st.markdown(
    """
<style>
html, body, [class*="css"] {font-family: Inter, Arial, sans-serif;}
.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(124,58,237,0.25), transparent 31%),
        radial-gradient(circle at 100% 7%, rgba(6,182,212,0.20), transparent 32%),
        linear-gradient(135deg, #060817 0%, #091123 50%, #0B1728 100%);
    color:#F8FAFC;
}
.block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1320px;}
section[data-testid="stSidebar"] {background: rgba(7,10,24,0.98); border-right:1px solid rgba(148,163,184,0.12);}
.hero {
    padding:26px 32px; border-radius:28px;
    background:linear-gradient(135deg,rgba(17,24,39,0.96),rgba(30,41,59,0.74));
    border:1px solid rgba(148,163,184,0.14); margin-bottom:18px;
    box-shadow:0 24px 70px rgba(0,0,0,0.25);
}
.hero h1 {font-size:38px; line-height:1.08; margin:0; letter-spacing:-1.2px;}
.hero p {color:#A5B4FC; font-size:15px; margin:12px 0 0 0;}
.context-box {
    padding:15px 18px; border-radius:20px;
    background:linear-gradient(135deg,rgba(6,182,212,0.12),rgba(124,58,237,0.12));
    border:1px solid rgba(148,163,184,0.16); color:#CBD5E1;
    line-height:1.58; margin-bottom:16px;
}
.context-box span {color:#A7F3D0; font-weight:900;}
.metric-card {
    min-height:124px; border-radius:24px; padding:20px 22px;
    border:1px solid rgba(255,255,255,0.12);
    box-shadow:0 24px 60px rgba(0,0,0,0.32); margin-bottom:14px;
}
.metric-card.primary {background:linear-gradient(135deg,rgba(124,58,237,0.96),rgba(37,99,235,0.90));}
.metric-card.cyan {background:linear-gradient(135deg,rgba(6,182,212,0.96),rgba(37,99,235,0.90));}
.metric-card.dark {background:linear-gradient(135deg,rgba(30,41,59,0.92),rgba(15,23,42,0.98));}
.metric-label {font-size:13px; font-weight:900; color:rgba(255,255,255,0.76); margin-bottom:9px;}
.metric-value {font-size:34px; font-weight:900; color:#FFFFFF; letter-spacing:-0.7px;}
.metric-note {font-size:12px; color:rgba(255,255,255,0.68); margin-top:8px;}
.panel-head {
    background:rgba(15,23,42,0.72); border:1px solid rgba(148,163,184,0.13);
    border-radius:24px; padding:18px 21px; margin:16px 0 12px 0;
    box-shadow:0 20px 52px rgba(0,0,0,0.20);
}
.panel-title {font-size:19px; font-weight:900; color:#F8FAFC; margin-bottom:7px;}
.panel-subtitle {color:#94A3B8; font-size:13px; line-height:1.45;}
.rank-card {
    display:flex; align-items:center; justify-content:space-between; gap:16px;
    padding:14px 15px; border-radius:18px;
    background:linear-gradient(135deg, rgba(255,255,255,0.045), rgba(255,255,255,0.025));
    border:1px solid rgba(255,255,255,0.08); margin-bottom:9px;
}
.rank-left {display:flex; align-items:center; gap:12px;}
.rank-no {
    min-width:34px; height:34px; border-radius:12px; display:flex; align-items:center; justify-content:center;
    background:linear-gradient(135deg,#8B5CF6,#06B6D4); font-weight:900;
}
.rank-name {font-weight:900; color:#F8FAFC;}
.rank-caption {color:#94A3B8; font-size:12px; margin-top:2px;}
.rank-value {font-weight:900; color:#A7F3D0; white-space:nowrap;}
.insight-card {
    padding:20px 22px; border-radius:22px;
    background:linear-gradient(135deg,rgba(15,23,42,0.96),rgba(30,41,59,0.76));
    border:1px solid rgba(148,163,184,0.14);
    box-shadow:0 18px 46px rgba(0,0,0,0.22);
    height:100%; margin-bottom:14px;
}
.insight-card h3 {font-size:20px; line-height:1.25; margin:5px 0 12px 0;}
.insight-card p {font-size:14px; color:#CBD5E1; line-height:1.52;}
.insight-card .green {color:#A7F3D0;}
.insight-card .muted {color:#94A3B8;}
.tag {
    display:inline-block; padding:7px 11px; border-radius:999px;
    background:rgba(124,58,237,0.18); border:1px solid rgba(124,58,237,0.45);
    color:#DDD6FE; font-size:12px; font-weight:900; margin-bottom:8px;
}
.small-note {font-size:13px; color:#94A3B8; line-height:1.52;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
st.sidebar.markdown("## ◈ Olist Dashboard")
st.sidebar.caption("BI • RFM • Delivery Experience • Business Recommendation")

page = st.sidebar.radio(
    "PAGE",
    [
        "Executive Overview",
        "Data & Methodology",
        "Customer Segmentation",
        "Category & Market",
        "Delivery Experience",
        "Business Insight",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Visual Controls")
top_n = st.sidebar.slider("Top N ranking", 3, 12, 6)

month_min = month_max = None
if not monthly.empty:
    month_min = monthly["order_month"].min().to_pydatetime()
    month_max = monthly["order_month"].max().to_pydatetime()
    month_range = st.sidebar.slider(
        "Month range",
        min_value=month_min,
        max_value=month_max,
        value=(month_min, month_max),
        format="YYYY-MM",
    )
else:
    month_range = None

if not category.empty:
    min_orders = st.sidebar.slider("Minimum category orders", 0, int(category["orders"].max()), 0, step=250)
    default_categories = category["category_label"].head(15).tolist()
    selected_categories = st.sidebar.multiselect(
        "Category focus",
        category["category_label"].tolist(),
        default=default_categories,
    )
else:
    min_orders, selected_categories = 0, []

if not state.empty:
    selected_states = st.sidebar.multiselect(
        "State focus",
        state["state_label"].tolist(),
        default=state["state_label"].head(10).tolist(),
    )
else:
    selected_states = []

if not delivery.empty:
    selected_delivery = st.sidebar.multiselect(
        "Delivery status",
        delivery["delivery_status"].tolist(),
        default=delivery["delivery_status"].tolist(),
    )
else:
    selected_delivery = []

st.sidebar.markdown("---")
st.sidebar.caption(
    "Catatan: KPI utama memakai agregasi full dataset. Filter hanya memengaruhi visual yang memiliki dimensi terkait. Jangan membuat klaim global dari filter parsial."
)

# ---------- Filtered display data ----------
monthly_view = monthly.copy()
if month_range is not None and not monthly_view.empty:
    start, end = pd.to_datetime(month_range[0]), pd.to_datetime(month_range[1])
    monthly_view = monthly_view[(monthly_view["order_month"] >= start) & (monthly_view["order_month"] <= end)]

category_view = category.copy()
if not category_view.empty:
    category_view = category_view[category_view["orders"] >= min_orders]
    if selected_categories:
        category_view = category_view[category_view["category_label"].isin(selected_categories)]

state_view = state.copy()
if not state_view.empty and selected_states:
    state_view = state_view[state_view["state_label"].isin(selected_states)]

delivery_view = delivery.copy()
if not delivery_view.empty and selected_delivery:
    delivery_view = delivery_view[delivery_view["delivery_status"].isin(selected_delivery)]

# ---------- Global derived values ----------
total_orders = metric_any("Total Orders")
delivered_orders = metric_any("Delivered Orders")
unique_customers = metric_any("Unique Customers")
total_gmv = metric_any("Total GMV")
aov = metric_any("Average Order Value")
avg_review = metric_any("Average Review Score")
avg_delivery_days = metric_any("Average Delivery Days")
late_rate = metric_any("Late Delivery Rate")
one_time_rate = metric_any("One-Time Customer Rate")
avg_frequency = metric_any("Average Frequency")
total_rfm_customers = metric_any("Total Customers")

if not category.empty:
    top_category = category.iloc[0]
else:
    top_category = pd.Series(dtype=object)

if not state.empty:
    top_state = state.iloc[0]
else:
    top_state = pd.Series(dtype=object)

if not segment.empty:
    top_segment = segment.sort_values("priority_score", ascending=False).iloc[0]
else:
    top_segment = pd.Series(dtype=object)

on_time = delivery[delivery["delivery_status"].eq("On Time")]
late = delivery[delivery["delivery_status"].eq("Late")]
on_review = to_num(on_time["avg_review"].iloc[0]) if not on_time.empty else 0
late_review = to_num(late["avg_review"].iloc[0]) if not late.empty else 0
review_gap = on_review - late_review
late_low_review = to_num(late["low_review_rate"].iloc[0]) if not late.empty else 0
on_low_review = to_num(on_time["low_review_rate"].iloc[0]) if not on_time.empty else 0

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>Olist Customer Intelligence & Business Performance Dashboard</h1>
        <p>Business Intelligence • RFM-Based Customer Prioritization • Category/Market Portfolio • Delivery Experience • Data-Driven Recommendation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if missing_files:
    st.warning("File output belum lengkap: " + ", ".join(missing_files))

# =========================================================
# Pages
# =========================================================

if page == "Executive Overview":
    st.markdown(
        """
        <div class="context-box">
            <b>Executive Overview</b><br>
            <span>Mata kuliah yang didukung:</span> Kecerdasan Bisnis, Pemodelan Data Bisnis, Intuisi dan Wawasan Data.<br>
            <span>Fungsi:</span> merangkum performa bisnis melalui GMV, order, AOV, review, tren bulanan, kategori produk, dan wilayah pasar.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Total GMV", fmt_money(total_gmv), "Delivered transaction value"), unsafe_allow_html=True)
    c2.markdown(metric_card("Delivered Orders", fmt_num(delivered_orders), "Completed orders", "cyan"), unsafe_allow_html=True)
    c3.markdown(metric_card("Average Order Value", fmt_money(aov), "Revenue per order", "dark"), unsafe_allow_html=True)
    c4.markdown(metric_card("Avg Review Score", fmt_float(avg_review), "Customer experience proxy", "dark"), unsafe_allow_html=True)

    left, right = st.columns([1.45, 1])
    with left:
        panel("GMV and Orders Momentum", "Monthly revenue movement and order volume. This is descriptive trend monitoring, not forecasting.")
        if not monthly_view.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_view["order_month"], y=monthly_view["gmv"], name="GMV", mode="lines+markers", fill="tozeroy"))
            fig.add_trace(go.Scatter(x=monthly_view["order_month"], y=monthly_view["orders"], name="Orders", mode="lines+markers", yaxis="y2"))
            fig.update_layout(yaxis=dict(title="GMV"), yaxis2=dict(title="Orders", overlaying="y", side="right"))
            st.plotly_chart(style_fig(fig, 430), use_container_width=True)
        else:
            st.info("Monthly KPI belum tersedia untuk rentang/filter ini.")

    with right:
        panel("Top Market Contribution", "Customer states contributing the most GMV.")
        if not state_view.empty:
            for i, row in state_view.head(top_n).reset_index(drop=True).iterrows():
                st.markdown(
                    rank_card(i + 1, row.get("state_label", "-"), f"Orders: {fmt_num(row.get('orders', 0))} • Late: {fmt_pct(row.get('late_rate', 0))}", fmt_money(row.get("gmv", 0))),
                    unsafe_allow_html=True,
                )
        else:
            st.info("State KPI kosong setelah filter.")

    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card("Late Delivery Rate", fmt_pct(late_rate), "Delivered orders arriving after estimate", "dark"), unsafe_allow_html=True)
    c2.markdown(metric_card("Avg Delivery Days", fmt_float(avg_delivery_days), "Operational service proxy", "dark"), unsafe_allow_html=True)
    c3.markdown(metric_card("Unique Customers", fmt_num(unique_customers), "Distinct customer IDs", "dark"), unsafe_allow_html=True)

elif page == "Data & Methodology":
    st.markdown(
        """
        <div class="context-box">
            <b>Data & Methodology Summary</b><br>
            <span>Fungsi:</span> membuat dashboard ini bisa dipertanggungjawabkan secara akademik. Bagian ini penting karena tanpa definisi data, dashboard hanya menjadi visualisasi cantik.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Raw Orders", fmt_num(total_orders), "Order table coverage", "dark"), unsafe_allow_html=True)
    c2.markdown(metric_card("Delivered Orders", fmt_num(delivered_orders), "Main analytical scope", "cyan"), unsafe_allow_html=True)
    delivered_ratio = delivered_orders / total_orders if total_orders else 0
    c3.markdown(metric_card("Delivered Ratio", fmt_pct(delivered_ratio), "Order status retained", "primary"), unsafe_allow_html=True)
    c4.markdown(metric_card("RFM Customers", fmt_num(total_rfm_customers), "Customer-level analysis", "dark"), unsafe_allow_html=True)

    panel("Operational Definitions", "Definitions used to prevent vague analytical claims.")
    st.markdown(
        """
        - **GMV**: total transaction value used for business performance monitoring.
        - **Delivered orders**: completed orders retained as the main scope for performance, delivery, review, category, and market analysis.
        - **On Time**: delivered date is earlier than or equal to the estimated delivery date.
        - **Late**: delivered date is later than the estimated delivery date.
        - **Low Review Rate**: share of orders with review score ≤ 2.
        - **RFM segmentation**: rule-based customer value prioritization using recency, frequency, and monetary value.
        """
    )

    panel("Methodological Position", "This is the part a critical lecturer will check.")
    st.markdown(
        """
        **This dashboard is not a predictive model.** It is a descriptive business intelligence and prioritization dashboard.  
        RFM is used to prioritize customer value, not to prove customer loyalty. This matters because the dataset has a very high one-time purchase pattern.

        **Causal caution:** the delivery page shows an association between late delivery and lower review score. It does not prove that late delivery is the only cause of low reviews because review score may also be affected by product quality, seller reliability, price, and customer expectation.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card("One-Time Rate", fmt_pct(one_time_rate), "Main RFM limitation", "primary"), unsafe_allow_html=True)
    c2.markdown(metric_card("Avg Frequency", fmt_float(avg_frequency), "Repeat-purchase weakness", "cyan"), unsafe_allow_html=True)
    c3.markdown(metric_card("Review Gap", fmt_float(review_gap), "On-time review minus late review", "dark"), unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])
    with left:
        panel("Input Data Audit", "Rows, columns, duplicates, and missingness from source tables.")
        if not audit.empty:
            st.dataframe(audit, use_container_width=True, hide_index=True)
        else:
            st.info("audit_df.csv belum tersedia.")

    with right:
        panel("Evidence Mapping", "Do not treat course labels as decoration; link each label to actual analytical output.")
        evidence_rows = [
            ("Kecerdasan Bisnis", "KPI, GMV trend, order volume, top market, category portfolio"),
            ("Analisis Segmentasi Pelanggan", "RFM score, segment priority, one-time buyer limitation"),
            ("Pemodelan Data Bisnis", "Rule-based prioritization using value, volume, review, and delivery-risk indicators"),
            ("Intuisi dan Wawasan Data", "Evidence → interpretation → action → KPI monitoring recommendations"),
        ]
        for no, (course, evidence_text) in enumerate(evidence_rows, 1):
            st.markdown(rank_card(no, course, evidence_text, "Evidence"), unsafe_allow_html=True)

elif page == "Customer Segmentation":
    st.markdown(
        """
        <div class="context-box">
            <b>Customer Segmentation</b><br>
            <span>Mata kuliah yang didukung:</span> Analisis Segmentasi Pelanggan dan Pemodelan Data Bisnis.<br>
            <span>Fungsi:</span> menerapkan RFM untuk customer value prioritization. Interpretasi sengaja dibatasi karena repeat purchase rendah.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card("RFM Customers", fmt_num(total_rfm_customers), "Customer-level analysis"), unsafe_allow_html=True)
    c2.markdown(metric_card("One-Time Rate", fmt_pct(one_time_rate), "Critical limitation", "cyan"), unsafe_allow_html=True)
    c3.markdown(metric_card("Avg Frequency", fmt_float(avg_frequency), "Repeat behavior indicator", "dark"), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="context-box">
            <b>Reading guide</b><br>
            <span>Big Spenders</span> = high monetary value; <span>Champions</span> = strongest RFM score but very small count; <span>New Customers</span> = recent buyers; <span>At Risk/Hibernating</span> = candidates for selective retention/reactivation. This is RFM rule-based prioritization, not clustering.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 1])
    with left:
        panel("Revenue Share by Segment", "Monetary contribution by customer segment.")
        if not segment.empty:
            fig = px.bar(
                segment.sort_values("revenue_share"),
                x="revenue_share",
                y="segment_label",
                orientation="h",
                text="revenue_share",
                labels={"revenue_share": "Revenue Share", "segment_label": "Segment"},
            )
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            st.plotly_chart(style_fig(fig, 430), use_container_width=True)
        else:
            st.info("segment_priority.csv belum tersedia.")

    with right:
        panel("Segment Action Priority", "Priority score is a business prioritization aid, not a predictive probability.")
        if not segment.empty:
            for i, row in segment.sort_values("priority_score", ascending=False).reset_index(drop=True).iterrows():
                caption = f"Customers: {fmt_num(row.get('customers', 0))} • Revenue share: {fmt_pct(row.get('revenue_share', 0))}"
                st.markdown(rank_card(i + 1, row.get("segment_label", "-"), caption, fmt_float(row.get("priority_score", 0))), unsafe_allow_html=True)
        else:
            st.info("Segment data kosong.")

    panel("RFM Limitation That Must Be Stated in the Report", "This is not weakness if you state it honestly.")
    st.markdown(
        f"""
        The dataset has **{fmt_pct(one_time_rate)} one-time customers** and **{fmt_float(avg_frequency)} average frequency**.  
        Therefore, the correct claim is: **RFM helps prioritize current customer value**, not: "RFM proves long-term loyalty".  
        Overclaiming this point is the easiest way for a lecturer to attack the project.
        """
    )

elif page == "Category & Market":
    st.markdown(
        """
        <div class="context-box">
            <b>Category & Market Portfolio</b><br>
            <span>Mata kuliah yang didukung:</span> Kecerdasan Bisnis dan Pemodelan Data Bisnis.<br>
            <span>Fungsi:</span> menggabungkan nilai transaksi, volume order, review, dan risiko keterlambatan untuk membaca fokus komersial dan wilayah.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 1])
    with left:
        panel("Category Performance Portfolio", "Bubble size = GMV; color = average review; filtered by sidebar controls.")
        if not category_view.empty:
            fig = px.scatter(
                category_view.head(25),
                x="orders",
                y="gmv",
                size="gmv",
                color="avg_review",
                hover_name="category_label",
                hover_data={"late_rate": ":.1%", "orders": True, "gmv": ":,.0f", "avg_review": ":.2f"},
                labels={"orders": "Orders", "gmv": "GMV", "avg_review": "Avg Review"},
            )
            st.plotly_chart(style_fig(fig, 460), use_container_width=True)
        else:
            st.info("Category KPI kosong setelah filter.")

    with right:
        panel("Top Category Focus", "Highest GMV categories after filter.")
        if not category_view.empty:
            for i, row in category_view.head(top_n).reset_index(drop=True).iterrows():
                caption = f"Orders: {fmt_num(row.get('orders', 0))} • Review: {fmt_float(row.get('avg_review', 0))} • Late: {fmt_pct(row.get('late_rate', 0))}"
                st.markdown(rank_card(i + 1, row.get("category_label", "-"), caption, fmt_money(row.get("gmv", 0))), unsafe_allow_html=True)
        else:
            st.info("Tidak ada kategori sesuai filter.")

    left, right = st.columns([1.15, 1])
    with left:
        panel("Market Delivery Risk", "States with high GMV still need delivery-risk monitoring.")
        if not state_view.empty:
            fig = px.scatter(
                state_view,
                x="orders",
                y="late_rate",
                size="gmv",
                color="avg_delivery_days",
                hover_name="state_label",
                hover_data={"gmv": ":,.0f", "orders": True, "avg_delivery_days": ":.2f", "late_rate": ":.1%"},
                labels={"orders": "Orders", "late_rate": "Late Rate", "avg_delivery_days": "Avg Delivery Days"},
            )
            st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        else:
            st.info("State KPI kosong setelah filter.")

    with right:
        panel("Top Market Focus", "Highest GMV states after filter.")
        if not state_view.empty:
            for i, row in state_view.head(top_n).reset_index(drop=True).iterrows():
                caption = f"Orders: {fmt_num(row.get('orders', 0))} • Delivery days: {fmt_float(row.get('avg_delivery_days', 0))} • Late: {fmt_pct(row.get('late_rate', 0))}"
                st.markdown(rank_card(i + 1, row.get("state_label", "-"), caption, fmt_money(row.get("gmv", 0))), unsafe_allow_html=True)
        else:
            st.info("Tidak ada wilayah sesuai filter.")

elif page == "Delivery Experience":
    st.markdown(
        """
        <div class="context-box">
            <b>Delivery Experience</b><br>
            <span>Mata kuliah yang didukung:</span> Kecerdasan Bisnis, Pemodelan Data Bisnis, Intuisi dan Wawasan Data.<br>
            <span>Fungsi:</span> membaca asosiasi antara status pengiriman dan review pelanggan secara deskriptif. Ini bukan causal inference dan bukan predictive modeling.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("On-Time Review", fmt_float(on_review), "Average review score"), unsafe_allow_html=True)
    c2.markdown(metric_card("Late Review", fmt_float(late_review), "Average review score", "cyan"), unsafe_allow_html=True)
    c3.markdown(metric_card("Review Gap", fmt_float(review_gap), "On-time minus late", "dark"), unsafe_allow_html=True)
    c4.markdown(metric_card("Late Low Review", fmt_pct(late_low_review), "Review score ≤ 2", "dark"), unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        panel("Association: Delivery Status vs Average Review", "Do not write this as a causal claim.")
        if not delivery_view.empty:
            fig = px.bar(delivery_view, x="delivery_status", y="avg_review", text="avg_review", labels={"delivery_status": "Delivery Status", "avg_review": "Average Review"})
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(style_fig(fig, 400), use_container_width=True)
        else:
            st.info("Delivery data kosong setelah filter.")

    with right:
        panel("Low Review Rate", "Share of orders with review score ≤ 2 by delivery status.")
        if not delivery_view.empty:
            fig = px.bar(delivery_view, x="delivery_status", y="low_review_rate", text="low_review_rate", labels={"delivery_status": "Delivery Status", "low_review_rate": "Low Review Rate"})
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            st.plotly_chart(style_fig(fig, 400), use_container_width=True)
        else:
            st.info("Delivery data kosong setelah filter.")

    panel("Operational Profile", "Orders and average delivery days by status.")
    if not delivery_view.empty:
        for _, row in delivery_view.iterrows():
            caption = f"Orders: {fmt_num(row.get('orders', 0))} • Avg Delivery Days: {fmt_float(row.get('avg_delivery_days', 0))} • Low Review: {fmt_pct(row.get('low_review_rate', 0))}"
            st.markdown(rank_card("•", row.get("delivery_status", "-"), caption, f"Avg Review {fmt_float(row.get('avg_review', 0))}"), unsafe_allow_html=True)

    panel("Correct Interpretation for Report", "This protects the analysis from overclaiming.")
    st.markdown(
        f"""
        Orders categorized as **Late** have lower average review (**{fmt_float(late_review)}**) than **On Time** orders (**{fmt_float(on_review)}**).  
        The low-review rate is also much higher for late orders (**{fmt_pct(late_low_review)}**) than on-time orders (**{fmt_pct(on_low_review)}**).  
        The safe conclusion is: **late delivery is strongly associated with worse customer experience**, not that delivery delay is the only cause of poor reviews.
        """
    )

else:
    st.markdown(
        """
        <div class="context-box">
            <b>Business Insight</b><br>
            <span>Mata kuliah yang didukung:</span> Pemodelan Data Bisnis dan Intuisi/Wawasan Data.<br>
            <span>Fungsi:</span> mengubah KPI, segmentasi, kategori, wilayah, dan delivery experience menjadi rekomendasi bisnis yang bisa dimonitor.
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_category_name = top_category.get("category_label", "Top category") if not top_category.empty else "Top category"
    top_state_name = top_state.get("state_label", "Top market") if not top_state.empty else "Top market"
    top_segment_name = top_segment.get("segment_label", "Top segment") if not top_segment.empty else "Top segment"

    cards = [
        {
            "title": "Protect the Main Revenue Category",
            "tag": "Commercial Focus",
            "evidence": f"{top_category_name} contributes the highest GMV at {fmt_money(top_category.get('gmv', 0))}, with {fmt_num(top_category.get('orders', 0))} orders and average review {fmt_float(top_category.get('avg_review', 0))}.",
            "interpretation": "This category is a revenue engine. The priority is not random promotion, but availability control and selective campaign support.",
            "action": "Maintain stock reliability, bundle related products, and prioritize campaigns for high-GMV categories with acceptable review performance.",
            "kpi": "GMV, order volume, AOV, review score, late delivery rate.",
        },
        {
            "title": "Focus Market Execution on the Largest State",
            "tag": "Market Focus",
            "evidence": f"{top_state_name} is the largest market with {fmt_money(top_state.get('gmv', 0))}, {fmt_num(top_state.get('orders', 0))} orders, and late rate {fmt_pct(top_state.get('late_rate', 0))}.",
            "interpretation": "The largest market should receive separate commercial and logistics monitoring because small rate changes affect many orders.",
            "action": "Use state-level tracking for campaign allocation, seller performance review, and delivery monitoring in high-volume states.",
            "kpi": "State GMV, orders, late rate, average delivery days.",
        },
        {
            "title": "Prioritize Customer Value, Not Fake Loyalty Claims",
            "tag": "Customer Focus",
            "evidence": f"{top_segment_name} has priority score {fmt_float(top_segment.get('priority_score', 0))}. However, one-time customer rate is {fmt_pct(one_time_rate)} and average frequency is {fmt_float(avg_frequency)}.",
            "interpretation": "RFM is useful for prioritizing customer value, but weak for claiming loyalty because repeat purchase is rare.",
            "action": "Apply high-value offers to Big Spenders and second-purchase campaigns to New Customers, while keeping reactivation selective for Hibernating customers.",
            "kpi": "Repeat rate, customer share, revenue share, monetary value, campaign response.",
        },
        {
            "title": "Treat Late Delivery as a Customer Experience Risk",
            "tag": "Service Focus",
            "evidence": f"On-time orders average {fmt_float(on_review)} review score, while late orders average {fmt_float(late_review)}. Late low-review rate reaches {fmt_pct(late_low_review)}.",
            "interpretation": "Late delivery is strongly associated with lower customer review, so service monitoring must be part of business performance review.",
            "action": "Monitor delivery-risk states/categories and prioritize operational follow-up before late delivery becomes a review problem.",
            "kpi": "Late rate, delivery days, low review rate, average review score.",
        },
    ]

    for i in range(0, len(cards), 2):
        cols = st.columns(2)
        for col, card in zip(cols, cards[i : i + 2]):
            with col:
                st.markdown(
                    insight_card(card["title"], card["tag"], card["evidence"], card["interpretation"], card["action"], card["kpi"]),
                    unsafe_allow_html=True,
                )

    panel("Dashboard Storyline", "How the pages support the project narrative.")
    storyline = [
        ("Executive Overview", "Summarizes business performance through KPI and monthly momentum.", "Kecerdasan Bisnis"),
        ("Data & Methodology", "Defines dataset scope, metric definitions, limitations, and academic positioning.", "All evidence"),
        ("Customer Segmentation", "Uses RFM for rule-based customer prioritization with explicit limitation.", "Analisis Segmentasi Pelanggan"),
        ("Category & Market", "Reads commercial focus and regional delivery risk from GMV, orders, review, and late rate.", "Pemodelan Data Bisnis"),
        ("Delivery Experience", "Shows association between delivery status and customer review.", "Intuisi dan Wawasan Data"),
        ("Business Insight", "Converts patterns into actions and monitoring KPIs.", "Intuisi dan Wawasan Data"),
    ]
    for no, (page_name, content, course) in enumerate(storyline, 1):
        st.markdown(rank_card(no, page_name, content, course), unsafe_allow_html=True)

    panel("Next Improvement", "Do not claim these as completed unless you actually build them.")
    st.markdown(
        """
        - Build an **order-level fact table** so date, category, state, and delivery filters can update every KPI globally.
        - Add a simple statistical test for review difference between on-time and late orders.
        - Only add a predictive model later if you build a real target variable, train/test split, evaluation metric, and baseline comparison.
        """
    )
