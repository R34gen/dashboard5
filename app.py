import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Olist Customer Intelligence",
    page_icon="◈",
    layout="wide"
)

DATA = Path("olist_output")

FILES = {
    "kpi": ["kpi.csv"],
    "monthly": ["monthly_kpi.csv"],
    "category": ["category_kpi.csv"],
    "state": ["state_kpi.csv"],
    "customer_profile": ["customer_profile.csv", "customer_profile_summary.csv"],
    "segment": ["segment_priority.csv"],
    "rfm_diag": ["rfm_diagnostic.csv"],
    "delivery": ["delivery_experience.csv", "delivery_experience_summary.csv", "late_review_impact.csv"],
    "insight": ["insight_table.csv"],
    "storytelling": ["storytelling_map.csv"],
}

@st.cache_data(show_spinner=False)
def read_first_available(names):
    for name in names:
        path = DATA / name
        if path.exists():
            return pd.read_csv(path), name
    return pd.DataFrame(), None

@st.cache_data(show_spinner=False)
def load_data():
    data = {}
    missing = []
    used = {}

    for key, names in FILES.items():
        df, filename = read_first_available(names)
        data[key] = df
        used[key] = filename

        if filename is None and key not in ["customer_profile", "storytelling"]:
            missing.append(" / ".join(names))

    return data, missing, used

data, missing, used_files = load_data()

kpi = data["kpi"]
monthly = data["monthly"]
category = data["category"]
state = data["state"]
customer_profile = data["customer_profile"]
segment = data["segment"]
rfm_diag = data["rfm_diag"]
delivery = data["delivery"]
insight = data["insight"]
storytelling = data["storytelling"]

def to_num(x, default=0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def clean_label(x):
    return str(x).replace("_", " ").replace("-", " ").title()

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

def kpi_value(metric, default=0):
    if kpi.empty or not {"metric", "value"}.issubset(kpi.columns):
        return default

    hit = kpi.loc[kpi["metric"].astype(str).eq(metric), "value"]
    return to_num(hit.iloc[0], default) if not hit.empty else default

def metric_value(df, metric, default=0):
    if df.empty or not {"metric", "value"}.issubset(df.columns):
        return default

    hit = df.loc[df["metric"].astype(str).eq(metric), "value"]
    return to_num(hit.iloc[0], default) if not hit.empty else default

def ensure_numeric(df, cols):
    if df.empty:
        return df

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
        font=dict(color="#E5E7EB", family="Inter, Arial"),
        margin=dict(l=18, r=18, t=42, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )

    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.14)",
        zerolinecolor="rgba(148,163,184,0.18)"
    )

    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.14)",
        zerolinecolor="rgba(148,163,184,0.18)"
    )

    return fig

def open_panel(title, subtitle=""):
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">{title}</div>
            <div class="panel-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )

def close_panel():
    st.markdown("</div>", unsafe_allow_html=True)

def metric_card(label, value, note, variant=""):
    return f"""
    <div class="metric-card {variant}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """

def rank_card(no, name, caption, value):
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
rfm_diag = ensure_numeric(rfm_diag, ["value"])
delivery = ensure_numeric(delivery, ["orders", "avg_review", "low_review_rate", "avg_delivery_days"])

if not category.empty and "main_category" in category.columns:
    category["category_label"] = category["main_category"].apply(clean_label)

if not state.empty and "customer_state" in state.columns:
    state["state_label"] = state["customer_state"].astype(str).str.upper()

if not segment.empty and "segment" in segment.columns:
    segment["segment_label"] = segment["segment"].apply(clean_label)

if not delivery.empty:
    if "delivery_status" not in delivery.columns and "is_late" in delivery.columns:
        delivery["delivery_status"] = (
            delivery["is_late"]
            .map({False: "On Time", True: "Late", "False": "On Time", "True": "Late"})
            .fillna(delivery["is_late"].astype(str))
        )

    delivery["delivery_status"] = delivery["delivery_status"].astype(str)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(124,58,237,0.23), transparent 31%),
        radial-gradient(circle at 100% 7%, rgba(6,182,212,0.18), transparent 32%),
        linear-gradient(135deg, #060817 0%, #091123 50%, #0B1728 100%);
    color:#F8FAFC;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}

section[data-testid="stSidebar"] {
    background: rgba(7,10,24,0.98);
    border-right:1px solid rgba(148,163,184,0.12);
}

.hero {
    padding:24px 30px;
    border-radius:28px;
    background:linear-gradient(135deg,rgba(17,24,39,0.96),rgba(30,41,59,0.72));
    border:1px solid rgba(148,163,184,0.14);
    margin-bottom:18px;
    box-shadow:0 24px 70px rgba(0,0,0,0.25);
}

.hero h1 {
    font-size:36px;
    line-height:1.08;
    margin:0;
    letter-spacing:-1.2px;
}

.hero p {
    color:#A5B4FC;
    font-size:15px;
    margin:11px 0 0 0;
}

.context-box {
    padding:14px 17px;
    border-radius:20px;
    background:linear-gradient(135deg,rgba(6,182,212,0.10),rgba(124,58,237,0.10));
    border:1px solid rgba(148,163,184,0.16);
    color:#CBD5E1;
    line-height:1.54;
    margin-bottom:16px;
}

.context-box span {
    color:#A7F3D0;
    font-weight:900;
}

.metric-card {
    min-height:123px;
    border-radius:24px;
    padding:20px 22px;
    border:1px solid rgba(255,255,255,0.12);
    background:linear-gradient(135deg,rgba(124,58,237,0.96),rgba(37,99,235,0.90));
    box-shadow:0 24px 60px rgba(0,0,0,0.32);
}

.metric-card.cyan {
    background:linear-gradient(135deg,rgba(6,182,212,0.96),rgba(37,99,235,0.90));
}

.metric-card.dark {
    background:linear-gradient(135deg,rgba(30,41,59,0.92),rgba(15,23,42,0.98));
}

.metric-label {
    font-size:13px;
    font-weight:900;
    color:rgba(255,255,255,0.76);
    margin-bottom:9px;
}

.metric-value {
    font-size:32px;
    font-weight:900;
    color:#FFFFFF;
    letter-spacing:-0.7px;
}

.metric-note {
    font-size:12px;
    color:rgba(255,255,255,0.68);
    margin-top:8px;
}

.panel {
    background:rgba(15,23,42,0.74);
    border:1px solid rgba(148,163,184,0.13);
    border-radius:26px;
    padding:20px 22px 14px 22px;
    box-shadow:0 24px 60px rgba(0,0,0,0.25);
    margin-bottom:16px;
}

.panel-title {
    font-size:18px;
    font-weight:900;
    color:#F8FAFC;
    margin-bottom:7px;
}

.panel-subtitle {
    color:#94A3B8;
    font-size:13px;
    margin-bottom:10px;
    line-height:1.45;
}

.rank-card {
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:14px 15px;
    border-radius:18px;
    background:linear-gradient(135deg, rgba(255,255,255,0.045), rgba(255,255,255,0.025));
    border:1px solid rgba(255,255,255,0.08);
    margin-bottom:9px;
}

.rank-left {
    display:flex;
    align-items:center;
    gap:12px;
}

.rank-no {
    width:34px;
    height:34px;
    border-radius:12px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:linear-gradient(135deg,#8B5CF6,#06B6D4);
    font-weight:900;
}

.rank-name {
    font-weight:900;
    color:#F8FAFC;
}

.rank-caption {
    color:#94A3B8;
    font-size:12px;
    margin-top:2px;
}

.rank-value {
    font-weight:900;
    color:#A7F3D0;
}

.insight-card {
    padding:20px 22px;
    border-radius:22px;
    background:linear-gradient(135deg,rgba(15,23,42,0.96),rgba(30,41,59,0.74));
    border:1px solid rgba(148,163,184,0.14);
    box-shadow:0 18px 46px rgba(0,0,0,0.22);
    height:100%;
    margin-bottom:14px;
}

.insight-card h3 {
    font-size:19px;
    line-height:1.25;
    margin-top:4px;
}

.insight-card p {
    font-size:14px;
    color:#CBD5E1;
    line-height:1.5;
}

.tag {
    display:inline-block;
    padding:7px 11px;
    border-radius:999px;
    background:rgba(124,58,237,0.18);
    border:1px solid rgba(124,58,237,0.45);
    color:#DDD6FE;
    font-size:12px;
    font-weight:900;
    margin-bottom:10px;
}

hr {
    border:0;
    height:1px;
    background:rgba(148,163,184,0.15);
    margin:18px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("## ◈ Olist Dashboard")
st.sidebar.caption("BI • RFM • Data Storytelling")

page = st.sidebar.radio(
    "REPORT",
    [
        "Executive Overview",
        "Customer Segmentation",
        "Delivery Experience",
        "Business Insight",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Scope")
st.sidebar.write("Kecerdasan Bisnis")
st.sidebar.write("Analisis Segmentasi Pelanggan")
st.sidebar.write("Intuisi dan Wawasan Data")
st.sidebar.markdown("---")
st.sidebar.caption("Tidak memuat machine learning, risk modeling, atau predictive modeling.")

st.markdown(
    '<div class="hero"><h1>Olist Customer Intelligence Dashboard</h1><p>Business Intelligence • RFM Customer Prioritization • Delivery Experience • Business Insight</p></div>',
    unsafe_allow_html=True,
)

if missing:
    st.warning("File output belum lengkap: " + ", ".join(missing))

if page == "Executive Overview":
    st.markdown(
        '<div class="context-box"><b>Executive Overview</b><br><span>Mewakili:</span> Kecerdasan Bisnis — CPL02 dan CPL08.<br><span>Fungsi:</span> merangkum performa bisnis melalui KPI, tren transaksi, kategori produk, dan wilayah pasar.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(
        metric_card("Total GMV", fmt_money(kpi_value("Total GMV")), "Delivered transaction value"),
        unsafe_allow_html=True,
    )

    c2.markdown(
        metric_card("Delivered Orders", fmt_num(kpi_value("Delivered Orders")), "Completed orders", "cyan"),
        unsafe_allow_html=True,
    )

    c3.markdown(
        metric_card("Average Order Value", fmt_money(kpi_value("Average Order Value")), "Revenue per order", "dark"),
        unsafe_allow_html=True,
    )

    c4.markdown(
        metric_card("Avg Review Score", fmt_float(kpi_value("Average Review Score")), "Customer experience proxy", "dark"),
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.45, 1])

    with left:
        open_panel("GMV and Orders Momentum", "Monthly revenue movement and order volume.")

        if not monthly.empty:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=monthly["order_month"],
                    y=monthly["gmv"],
                    name="GMV",
                    mode="lines+markers",
                    fill="tozeroy",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=monthly["order_month"],
                    y=monthly["orders"],
                    name="Orders",
                    mode="lines+markers",
                    yaxis="y2",
                )
            )

            fig.update_layout(
                yaxis=dict(title="GMV"),
                yaxis2=dict(title="Orders", overlaying="y", side="right"),
            )

            st.plotly_chart(style_fig(fig, 420), use_container_width=True)

        close_panel()

    with right:
        open_panel("Top Markets", "Customer states contributing the most GMV.")

        if not state.empty:
            for i, row in state.head(7).reset_index(drop=True).iterrows():
                st.markdown(
                    rank_card(
                        i + 1,
                        row.get("state_label", row.get("customer_state", "-")),
                        f"Orders: {fmt_num(row.get('orders', 0))}",
                        fmt_money(row.get("gmv", 0)),
                    ),
                    unsafe_allow_html=True,
                )

        close_panel()

    left, right = st.columns([1.3, 1])

    with left:
        open_panel(
            "Category Performance Portfolio",
            "Product categories by GMV, order volume, review, and late delivery rate.",
        )

        if not category.empty:
            fig = px.scatter(
                category.head(15),
                x="orders",
                y="gmv",
                size="gmv",
                color="avg_review",
                hover_name="category_label",
                hover_data={
                    "late_rate": ":.1%",
                    "orders": True,
                    "gmv": ":,.0f",
                    "avg_review": ":.2f",
                },
                labels={
                    "orders": "Orders",
                    "gmv": "GMV",
                    "avg_review": "Avg Review",
                },
            )

            st.plotly_chart(style_fig(fig, 430), use_container_width=True)

        close_panel()

    with right:
        open_panel("Top Category Focus", "Highest GMV categories for commercial focus.")

        if not category.empty:
            for i, row in category.head(7).reset_index(drop=True).iterrows():
                st.markdown(
                    rank_card(
                        i + 1,
                        row.get("category_label", row.get("main_category", "-")),
                        f"Avg Review: {fmt_float(row.get('avg_review', 0))}",
                        fmt_money(row.get("gmv", 0)),
                    ),
                    unsafe_allow_html=True,
                )

        close_panel()

elif page == "Customer Segmentation":
    st.markdown(
        '<div class="context-box"><b>Customer Segmentation</b><br><span>Mewakili:</span> Analisis Segmentasi Pelanggan — CPL02 dan CPL08.<br><span>Fungsi:</span> menerapkan RFM untuk customer value prioritization dan strategi CRM.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.markdown(
        metric_card("RFM Customers", fmt_num(metric_value(rfm_diag, "Total Customers")), "Customer-level analysis"),
        unsafe_allow_html=True,
    )

    c2.markdown(
        metric_card("One-Time Rate", fmt_pct(metric_value(rfm_diag, "One-Time Customer Rate")), "RFM limitation to explain", "cyan"),
        unsafe_allow_html=True,
    )

    c3.markdown(
        metric_card("Avg Frequency", fmt_float(metric_value(rfm_diag, "Average Frequency")), "Repeat behavior indicator", "dark"),
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="context-box"><b>Segment reading guide</b><br><span>Big Spenders</span> = nilai transaksi tinggi; <span>Champions</span> = skor RFM terbaik; <span>New Customers</span> = baru membeli; <span>At Risk/Hibernating</span> = perlu retensi atau reaktivasi selektif. Segmentasi ini adalah RFM rule-based, bukan clustering.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 1])

    with left:
        open_panel("Revenue Share by Segment", "Monetary contribution by customer segment.")

        if not segment.empty:
            fig = px.bar(
                segment.sort_values("revenue_share"),
                x="revenue_share",
                y="segment_label",
                orientation="h",
                text="revenue_share",
                labels={
                    "revenue_share": "Revenue Share",
                    "segment_label": "Segment",
                },
            )

            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            st.plotly_chart(style_fig(fig, 420), use_container_width=True)

        close_panel()

    with right:
        open_panel("Action Priority", "Recommended CRM action by segment priority.")

        if not segment.empty:
            for i, row in segment.sort_values("priority_score", ascending=False).reset_index(drop=True).iterrows():
                st.markdown(
                    rank_card(
                        i + 1,
                        row.get("segment_label", row.get("segment", "-")),
                        row.get("recommended_action", ""),
                        fmt_float(row.get("priority_score", 0)),
                    ),
                    unsafe_allow_html=True,
                )

        close_panel()

    open_panel("RFM Boundary", "Important methodological limitation.")

    st.markdown(
        """
        RFM digunakan sebagai **customer value prioritization**, bukan klaim loyalitas mutlak.  
        Alasannya: banyak customer hanya melakukan satu kali transaksi, sehingga dimensi Frequency terbatas.
        """
    )

    close_panel()

elif page == "Delivery Experience":
    st.markdown(
        '<div class="context-box"><b>Delivery Experience</b><br><span>Mewakili:</span> Kecerdasan Bisnis + Intuisi dan Wawasan Data — CPL02 dan CPL08.<br><span>Fungsi:</span> membaca pengalaman layanan pengiriman secara deskriptif. Ini bukan risk modeling dan bukan predictive model.</div>',
        unsafe_allow_html=True,
    )

    if not delivery.empty:
        on = delivery[delivery["delivery_status"].eq("On Time")]
        late = delivery[delivery["delivery_status"].eq("Late")]

        on_review = to_num(on["avg_review"].iloc[0]) if not on.empty else 0
        late_review = to_num(late["avg_review"].iloc[0]) if not late.empty else 0
        gap = on_review - late_review
    else:
        on_review = late_review = gap = 0

    c1, c2, c3 = st.columns(3)

    c1.markdown(
        metric_card("On-Time Review", fmt_float(on_review), "Average review score"),
        unsafe_allow_html=True,
    )

    c2.markdown(
        metric_card("Late Review", fmt_float(late_review), "Average review score", "cyan"),
        unsafe_allow_html=True,
    )

    c3.markdown(
        metric_card("Review Gap", fmt_float(gap), "On-time minus late", "dark"),
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        open_panel("Review Impact of Late Delivery", "Comparison of average review by delivery status.")

        if not delivery.empty:
            fig = px.bar(
                delivery,
                x="delivery_status",
                y="avg_review",
                text="avg_review",
                labels={
                    "delivery_status": "Delivery Status",
                    "avg_review": "Average Review",
                },
            )

            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(style_fig(fig, 390), use_container_width=True)

        close_panel()

    with right:
        open_panel("Low Review Rate", "Share of orders with review score ≤ 2.")

        if not delivery.empty:
            fig = px.bar(
                delivery,
                x="delivery_status",
                y="low_review_rate",
                text="low_review_rate",
                labels={
                    "delivery_status": "Delivery Status",
                    "low_review_rate": "Low Review Rate",
                },
            )

            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            st.plotly_chart(style_fig(fig, 390), use_container_width=True)

        close_panel()

    open_panel("Experience Profile", "Operational profile by delivery status.")

    if not delivery.empty:
        for _, row in delivery.iterrows():
            st.markdown(
                rank_card(
                    "•",
                    row.get("delivery_status", "-"),
                    f"Orders: {fmt_num(row.get('orders', 0))} • Avg Delivery Days: {fmt_float(row.get('avg_delivery_days', 0))}",
                    f"Avg Review {fmt_float(row.get('avg_review', 0))}",
                ),
                unsafe_allow_html=True,
            )

    close_panel()

else:
    st.markdown(
        '<div class="context-box"><b>Business Insight</b><br><span>Mewakili:</span> Intuisi dan Wawasan Data — CPL02 dan CPL08.<br><span>Fungsi:</span> mengubah hasil KPI, segmentasi, dan delivery experience menjadi rekomendasi bisnis.</div>',
        unsafe_allow_html=True,
    )

    if not insight.empty:
        label_col = "focus" if "focus" in insight.columns else ("finding" if "finding" in insight.columns else insight.columns[0])
        rows = insight.head(4).to_dict("records")

        for i in range(0, len(rows), 2):
            cols = st.columns(2)

            for col, row in zip(cols, rows[i : i + 2]):
                with col:
                    st.markdown(
                        f"""
                        <div class="insight-card">
                            <span class="tag">{row.get(label_col, '')}</span>
                            <h3>{row.get('evidence', '')}</h3>
                            <p><b>Interpretation:</b> {row.get('interpretation', '')}</p>
                            <p style="color:#A7F3D0"><b>Recommendation:</b> {row.get('recommendation', '')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    else:
        st.info("Insight file belum tersedia.")

    if not storytelling.empty:
        open_panel("Dashboard Storyline", "How each page supports the project narrative.")

        for i, row in storytelling.head(4).iterrows():
            page_name = row.get("dashboard_page", "-")
            content = row.get("content", row.get("business_question", ""))
            course = row.get("mata_kuliah", row.get("decision_use", ""))

            st.markdown(
                rank_card(i + 1, page_name, content, course),
                unsafe_allow_html=True,
            )

        close_panel()
