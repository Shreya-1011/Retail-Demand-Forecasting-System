import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Retail Demand Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - DARK THEME
# ============================================================
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    h1, h2, h3, h4 {
        color: #FAFAFA;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #00D4FF;
        text-align: center;
        padding: 0.5rem 0 0 0;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.35);
    }

    .sub-title {
        text-align: center;
        color: #FFFFFF;
        font-size: 1.05rem;
        font-weight:500;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.45);
    }

    div[data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
        font-weight:600;
    }

    div[data-testid="stMetricValue"] {
        color: #00D4FF;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #161B22;
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        color: #FFFFFF !important;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        background-color: #00D4FF;
        color: #0E1117 !important;
    }

    .custom-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.45);
    }

    .prediction-card {
        background: linear-gradient(135deg, #161B22 0%, #1E2530 100%);
        border: 2px solid #00D4FF;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.2);
        margin-bottom: 1.5rem;
    }

    .prediction-value {
        font-size: 3.8rem;
        font-weight: 900;
        color: #00D4FF;
        line-height: 1.1;
    }

    .prediction-label {
        font-size: 1.1rem;
        color: #FFFFFF;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-top: 0.5rem;
    }

    .info-card {
        background-color: #161B22;
        border-left: 4px solid #00D4FF;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin-bottom: 1rem;
    }

    .info-card h4 {
        color: #00D4FF;
        margin-bottom: 0.4rem;
    }

    .info-card p {
        color: #FFFFFF;
        margin: 0;
    }

    .footer {
        text-align: center;
        color: #FFFFFF;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #30363D;
        margin-top: 2rem;
        font-size: 0.95rem;
    }

    .footer span {
        color: #00D4FF;
        font-weight: 700;
    }

    .stButton button {
        background-color: #00D4FF;
        color: #0E1117;
        border-radius: 8px;
        font-weight: 700;
        border: none;
        width: 100%;
    }

    /* ===================== SIDEBAR INPUT TEXT - BRIGHT WHITE ===================== */

    /* Widget labels (titles above inputs) and section headers */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: #FFFFFF !important;
        font-weight: 600;
    }

    /* Captions / helper text */
    section[data-testid="stSidebar"] small,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #E6EDF3 !important;
    }

    /* Number input & text input fields */
    section[data-testid="stSidebar"] input {
        color: #FFFFFF !important;
        background-color: #21262D !important;
        border: 1px solid #30363D !important;
        caret-color: #FFFFFF !important;
    }

    /* Selectbox (collapsed view + selected value) */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #21262D !important;
        border: 1px solid #30363D !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #FFFFFF !important;
    }

    /* Selectbox dropdown menu options */
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] li * {
        color: #FFFFFF !important;
        background-color: #21262D !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: #00D4FF !important;
        color: #0E1117 !important;
    }
    div[data-baseweb="popover"] li:hover * {
        color: #0E1117 !important;
    }

    /* Radio button labels */
    section[data-testid="stSidebar"] div[role="radiogroup"] label,
    section[data-testid="stSidebar"] div[role="radiogroup"] p {
        color: #FFFFFF !important;
    }

    /* Slider current value, min/max tick labels */
    section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
    section[data-testid="stSidebar"] [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] [data-testid="stTickBarMax"],
    section[data-testid="stSidebar"] [data-testid="stThumbValue"] {
        color: #FFFFFF !important;
        font-weight: 700;
    }

    /* Global bright white text */
    p, span, label, small, li, .stMarkdown, .stText, .stCaption {
        color:#FFFFFF !important;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("Models/Linear_Regression.pkl")
        scaler = joblib.load("Models/scaler.pkl")
        columns = joblib.load("Models/columns.pkl")
        return model, scaler, columns, True
    except Exception:
        return None, None, None, False


model, scaler, model_columns, artifacts_loaded = load_artifacts()

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">📈 Retail Demand Forecasting System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered demand prediction using Linear Regression | Smart Inventory & Promotion Insights</div>', unsafe_allow_html=True)

if not artifacts_loaded:
    st.warning("⚠️ Model artifacts not found in `Models/` folder. Running in **Demo Mode** with a fallback formula so the dashboard remains fully interactive.")

# ============================================================
# SIDEBAR INPUTS
# ============================================================
st.sidebar.markdown("## ⚙️ Input Parameters")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🏷️ Product Information")
category = st.sidebar.selectbox(
    "Product Category",
    ["Clothing", "Electronics", "Grocery", "Home", "Personal Care"]
)

product_freq = st.sidebar.number_input(
    "Product Frequency",
    min_value=0.0, max_value=1000.0, value=25.0, step=1.0,
    help="How frequently this product appears in historical sales records"
)

st.sidebar.markdown("### 📊 Demand History")
units_sold_lag_1 = st.sidebar.number_input(
    "Units Sold Yesterday (lag_1)",
    min_value=0.0, max_value=10000.0, value=50.0, step=1.0
)

units_sold_roll_mean_7 = st.sidebar.number_input(
    "7-Day Average Units Sold",
    min_value=0.0, max_value=10000.0, value=48.0, step=1.0
)

st.sidebar.markdown("### 💰 Pricing & Promotion")
discount_percent = st.sidebar.slider(
    "Discount Percent (%)",
    min_value=0.0, max_value=100.0, value=10.0, step=0.5
)

effective_price = st.sidebar.number_input(
    "Effective Price (₹)",
    min_value=0.0, max_value=100000.0, value=499.0, step=1.0
)

st.sidebar.markdown("### 📦 Inventory")
inventory_level = st.sidebar.number_input(
    "Inventory Level",
    min_value=0.0, max_value=100000.0, value=200.0, step=1.0
)

st.sidebar.markdown("### 📅 Date & Seasonality")
day_of_week = st.sidebar.selectbox(
    "Day of Week",
    ["Saturday", "Sunday", "Weekday"]
)

month = st.sidebar.slider(
    "Month (1-12)",
    min_value=1, max_value=12, value=6, step=1
)

festival_window = st.sidebar.radio(
    "Festival Window",
    ["Yes", "No"],
    horizontal=True,
    index=1
)

st.sidebar.markdown("---")
st.sidebar.caption("🔄 Predictions update automatically as inputs change.")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

category_encoding = {
    "Clothing":      [0, 0, 0, 0],
    "Electronics":   [1, 0, 0, 0],
    "Grocery":       [0, 1, 0, 0],
    "Home":          [0, 0, 1, 0],
    "Personal Care": [0, 0, 0, 1],
}
cat_elec, cat_groc, cat_home, cat_pcare = category_encoding[category]

day_sat = 1 if day_of_week == "Saturday" else 0
day_sun = 1 if day_of_week == "Sunday" else 0

festival_flag = 1 if festival_window == "Yes" else 0

# ============================================================
# BUILD INPUT DATAFRAME
# ============================================================
input_dict = {
    "units_sold_lag_1": units_sold_lag_1,
    "units_sold_roll_mean_7": units_sold_roll_mean_7,
    "discount_percent": discount_percent,
    "effective_price": effective_price,
    "inventory_level": inventory_level,
    "product_freq": product_freq,
    "category_Electronics": cat_elec,
    "category_Grocery": cat_groc,
    "category_Home": cat_home,
    "category_Personal Care": cat_pcare,
    "day_of_week_Saturday": day_sat,
    "day_of_week_Sunday": day_sun,
    "month_sin": month_sin,
    "month_cos": month_cos,
    "is_festival_window": festival_flag,
}

input_df = pd.DataFrame([input_dict])

# ============================================================
# PREDICTION PIPELINE
# ============================================================
if artifacts_loaded:
    try:
        input_df = input_df[model_columns]
        scaled_input = scaler.transform(input_df)
        prediction = float(model.predict(scaled_input)[0])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        prediction = 0.0
        scaled_input = np.zeros((1, len(input_df.columns)))
else:
    # Fallback heuristic formula (Demo Mode)
    prediction = (
        units_sold_lag_1 * 0.55
        + units_sold_roll_mean_7 * 0.35
        + inventory_level * 0.01
        - effective_price * 0.01
        + discount_percent * 0.30
        + festival_flag * 8
        + (cat_elec * 3 + cat_groc * 5 + cat_home * 2 + cat_pcare * 1)
    )
    prediction = max(prediction, 0.0)
    scaled_input = np.zeros((1, len(input_df.columns)))

prediction = round(prediction, 2)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📈 Prediction", "📊 Analytics", "🤖 Model Insights"])

# ============================================================
# TAB 1 - PREDICTION DASHBOARD
# ============================================================
with tab1:
    st.markdown("## 📈 Prediction Dashboard")
    st.markdown("")

    # KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Predicted Demand", f"{prediction:.2f} units")
    with k2:
        st.metric("Inventory Level", f"{inventory_level:.0f} units")
    with k3:
        st.metric("Discount", f"{discount_percent:.1f}%")
    with k4:
        st.metric("Effective Price", f"₹{effective_price:.2f}")

    st.markdown("")

    col_left, col_right = st.columns([1, 1])

    # Large Prediction Card
    with col_left:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">📦 Predicted Demand</div>
            <div class="prediction-value">{prediction:.2f}</div>
            <div class="prediction-label">units expected</div>
        </div>
        """, unsafe_allow_html=True)

        # Inventory Alert
        if prediction > inventory_level:
            st.error("⚠️ Predicted demand exceeds available inventory.")
        else:
            st.success("✅ Inventory level is sufficient.")

    # Gauge Chart
    with col_right:
        gauge_max = max(prediction * 1.5, inventory_level * 1.2, 10)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            number={'suffix': " units", 'font': {'color': '#00D4FF', 'size': 36}},
            delta={
                'reference': inventory_level,
                'increasing': {'color': "#FF4B4B"},
                'decreasing': {'color': "#3FB950"}
            },
            title={'text': "Predicted Demand", 'font': {'color': '#FAFAFA', 'size': 22}},
            gauge={
                'axis': {'range': [0, gauge_max], 'tickcolor': '#8B949E'},
                'bar': {'color': "#00D4FF"},
                'bgcolor': "#161B22",
                'borderwidth': 1,
                'bordercolor': "#30363D",
                'steps': [
                    {'range': [0, inventory_level * 0.5], 'color': '#1E2530'},
                    {'range': [inventory_level * 0.5, inventory_level], 'color': '#2D3748'},
                ],
                'threshold': {
                    'line': {'color': "#FF4B4B", 'width': 4},
                    'thickness': 0.8,
                    'value': inventory_level
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font={'color': "#FAFAFA"},
            height=380,
            margin=dict(t=60, b=20, l=30, r=30)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

# ============================================================
# TAB 2 - VISUAL ANALYTICS
# ============================================================
with tab2:
    st.markdown("## Visual Analytics")
    st.markdown("")

    col1, col2 = st.columns(2)

    # 1. Feature Input Bar Chart
    with col1:
        feature_names = ["Lag Demand", "7-Day Avg", "Inventory", "Price", "Discount"]
        feature_values = [
            units_sold_lag_1,
            units_sold_roll_mean_7,
            inventory_level,
            effective_price,
            discount_percent
        ]

        fig_bar = px.bar(
            x=feature_names,
            y=feature_values,
            color=feature_names,
            text=[f"{v:.1f}" for v in feature_values],
            color_discrete_sequence=px.colors.sequential.Teal,
            title="📊 Feature Input Overview"
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#161B22",
            font_color="#FFFFFF",
            showlegend=False,
            xaxis_title="",
            yaxis_title="Value",
            height=420
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ============================================================
# TAB 3 - MODEL INSIGHTS
# ============================================================
with tab3:
    st.markdown("## Model Insights")
    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model", "Linear Regression")

    with col2:
        st.metric("Scaler", "StandardScaler")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("R² Score", "0.999")

    with col4:
        st.metric("MAE", "0.13")

    st.markdown("---")
    st.markdown("### Why Demand Prediction Matters")

    st.markdown("""
    <div class="info-card">
        <h4>Inventory Optimization</h4>
        <p>Accurate demand forecasts help maintain optimal stock levels — reducing both
        overstock costs and stockout risks, ensuring products are available when customers need them.</p>
    </div>

    <div class="info-card">
        <h4>Promotion Planning</h4>
        <p>Understanding how discounts and festival windows influence demand allows
        businesses to design promotions that maximize sales without eroding margins.</p>
    </div>

    <div class="info-card">
        <h4>Business Decision Making</h4>
        <p>Reliable forecasts support smarter procurement, staffing, and supply chain
        decisions — turning historical sales data into forward-looking strategy.</p>
    </div>

    <div class="info-card">
        <h4>Benefits of Forecasting</h4>
        <p>Forecasting reduces waste, improves cash flow, strengthens supplier negotiations,
        and increases customer satisfaction through consistent product availability.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    Built with <span>Streamlit</span>, <span>Plotly</span> and <span>Machine Learning</span> 🚀
</div>
""", unsafe_allow_html=True)
