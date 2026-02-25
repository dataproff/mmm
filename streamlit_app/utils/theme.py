"""
Shared theme module for the DataProf MMM Streamlit app.
Provides unified dark-mode CSS, Plotly dark theme, and header rendering.
"""
import streamlit as st

from utils.i18n import t


# Brand colors
INDIGO_400 = "#818cf8"
INDIGO_500 = "#6366f1"
INDIGO_600 = "#4f46e5"
EMERALD_400 = "#34d399"
EMERALD_500 = "#10b981"
SLATE_200 = "#e2e8f0"
SLATE_400 = "#94a3b8"
SLATE_500 = "#64748b"
SLATE_800 = "#1e293b"
SLATE_900 = "#0f172a"
SLATE_950 = "#020617"

# Chart color sequence
DARK_COLORWAY = [
    INDIGO_500,   # indigo
    EMERALD_400,  # emerald
    "#f59e0b",    # amber
    "#ec4899",    # pink
    "#06b6d4",    # cyan
    "#a78bfa",    # violet
    "#fb923c",    # orange
    INDIGO_400,   # indigo-bright
]


def render_header(title_key=None, subtitle_key=None):
    """Render the DataProf header bar with gradient logo and brand name."""
    st.markdown("""
    <div class="dataprof-header">
        <div class="dataprof-header-content">
            <div class="dataprof-brand">
                <div class="dataprof-logo-mark">D</div>
                <span class="dataprof-brand-text">
                    <span style="color:#ffffff">data</span><span style="color:#818cf8">prof</span><span style="color:#64748b">.io</span>
                </span>
            </div>
            <p class="dataprof-subtitle">Data Analytics &amp; BI Consultancy</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def inject_css():
    """Inject the full DataProf dark-mode CSS."""
    st.markdown(DATAPROF_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Custom sidebar navigation with Material Icons (via st.page_link)
# ---------------------------------------------------------------------------

_NAV_PAGES = [
    ("Home.py", "Home", ":material/dashboard:"),
    ("pages/1_Calculator.py", "Budget Optimizer", ":material/calculate:"),
    ("pages/2_Context_Calendar.py", "Context Calendar", ":material/calendar_month:"),
    ("pages/3_Saturation_Curves.py", "Saturation Curves", ":material/show_chart:"),
    ("pages/4_Spend_vs_Effect.py", "Spend vs Effect", ":material/bubble_chart:"),
]

_NAV_CSS = """
<style>
    /* Hide default Streamlit page nav */
    [data-testid="stSidebarNav"],
    nav[data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* Style page_link elements in sidebar */
    [data-testid="stSidebar"] .stPageLink a {
        padding: 0.55rem 0.85rem !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important;
        text-decoration: none !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }

    [data-testid="stSidebar"] .stPageLink a:hover {
        color: #e2e8f0 !important;
        background-color: rgba(255,255,255,0.05) !important;
    }

    /* Active page link */
    [data-testid="stSidebar"] .stPageLink a[aria-current="page"] {
        color: #ffffff !important;
        background-color: rgba(99, 102, 241, 0.15) !important;
    }

    /* Style the Material Icon inside page_link */
    [data-testid="stSidebar"] .stPageLink a span[data-testid="stIconMaterial"] {
        color: #94a3b8 !important;
        font-size: 1.2rem !important;
    }

    [data-testid="stSidebar"] .stPageLink a[aria-current="page"] span[data-testid="stIconMaterial"] {
        color: #818cf8 !important;
    }

    [data-testid="stSidebar"] .stPageLink a:hover span[data-testid="stIconMaterial"] {
        color: #e2e8f0 !important;
    }
</style>
"""


_NAV_PAGES_EMOJI_FALLBACK = [
    ("Home.py", "Home", "🏠"),
    ("pages/1_Calculator.py", "Budget Optimizer", "🔢"),
    ("pages/2_Context_Calendar.py", "Context Calendar", "📅"),
    ("pages/3_Saturation_Curves.py", "Saturation Curves", "📈"),
    ("pages/4_Spend_vs_Effect.py", "Spend vs Effect", "🔵"),
]


def render_sidebar_nav(current_page: str = "Home"):
    """Render custom sidebar navigation with Material Icons.

    Uses st.page_link with icon parameter for clean outlined icons.
    Falls back to emoji icons if Material Icons are not supported.
    """
    st.sidebar.markdown(_NAV_CSS, unsafe_allow_html=True)
    try:
        for page_path, label, icon in _NAV_PAGES:
            st.sidebar.page_link(page_path, label=label, icon=icon)
    except Exception:
        for page_path, label, icon in _NAV_PAGES_EMOJI_FALLBACK:
            st.sidebar.page_link(page_path, label=label, icon=icon)


# ---------------------------------------------------------------------------
# Architecture diagram (HTML)
# ---------------------------------------------------------------------------


def render_architecture_diagram() -> str:
    """Return an HTML/CSS architecture diagram matching the dark theme."""
    return '''
<style>
.arch-wrap{font-family:'Inter',sans-serif;display:grid;grid-template-columns:1fr 280px;gap:1rem;padding:0}
.arch-flow{display:flex;flex-direction:column;gap:0}
.arch-insights{grid-row:1/6;grid-column:2}

/* Layer row */
.arch-layer{padding:0.75rem 0}
.arch-label{font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;padding-left:0.75rem;border-left:3px solid}
.arch-cards{display:flex;gap:0.5rem;flex-wrap:wrap}

/* Card */
.arch-card{background:rgba(255,255,255,0.04);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:0.65rem 0.8rem;flex:1;min-width:120px}
.arch-card b{display:block;color:#e2e8f0;font-size:0.8rem;margin-bottom:0.2rem}
.arch-card span{color:#94a3b8;font-size:0.7rem;line-height:1.35}

/* Arrow */
.arch-arrow{text-align:center;color:#64748b;font-size:1.1rem;line-height:1;padding:0.15rem 0;user-select:none}

/* Insights panel */
.arch-insights-box{background:rgba(255,255,255,0.03);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:1rem;height:100%}
.arch-insights-box h4{color:#f43f5e;font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin:0 0 0.75rem}
.arch-item{margin-bottom:0.65rem}
.arch-item b{color:#e2e8f0;font-size:0.8rem}
.arch-item span{display:block;color:#94a3b8;font-size:0.7rem;margin-top:0.1rem}

/* Accent colors */
.c-cyan{border-color:#06b6d4;color:#06b6d4}
.c-amber{border-color:#f59e0b;color:#f59e0b}
.c-emerald{border-color:#10b981;color:#10b981}
.c-violet{border-color:#a78bfa;color:#a78bfa}

/* Responsive */
@media(max-width:800px){.arch-wrap{grid-template-columns:1fr}.arch-insights{grid-row:auto;grid-column:1}}
</style>

<div class="arch-wrap">
<div class="arch-flow">

  <!-- Layer 1: Data Sources -->
  <div class="arch-layer">
    <div class="arch-label c-cyan">Data Sources</div>
    <div class="arch-cards">
      <div class="arch-card"><b>Marketing Spend</b><span>Google Ads, Facebook<br>Yandex Direct, TV, Radio</span></div>
      <div class="arch-card"><b>Revenue Data</b><span>Sales transactions<br>E-commerce data</span></div>
      <div class="arch-card"><b>Calendar Events</b><span>Holidays, Promos<br>Seasonality</span></div>
      <div class="arch-card"><b>Macro Factors</b><span>Economic indicators<br>Market trends</span></div>
      <div class="arch-card"><b>Competitor Data</b><span>Market trends<br>Pressure index</span></div>
    </div>
  </div>
  <div class="arch-arrow">▼</div>

  <!-- Layer 2: Data Warehouse -->
  <div class="arch-layer">
    <div class="arch-label c-amber">Data Warehouse</div>
    <div class="arch-cards">
      <div class="arch-card"><b>Data Collection</b><span>Ingestion from sources<br>API connections</span></div>
      <div class="arch-card"><b>Data Cleaning</b><span>Remove duplicates<br>Handle missing values</span></div>
      <div class="arch-card"><b>Data Integration</b><span>Join data sources<br>Align time periods</span></div>
      <div class="arch-card"><b>Data Transformation</b><span>Feature engineering<br>Normalization</span></div>
    </div>
  </div>
  <div class="arch-arrow">▼</div>

  <!-- Layer 3: ML Engine -->
  <div class="arch-layer">
    <div class="arch-label c-emerald">MMM Machine Learning Engine</div>
    <div class="arch-cards">
      <div class="arch-card"><b>Feature Selection</b><span>Correlation analysis<br>Dimensionality reduction</span></div>
      <div class="arch-card"><b>Model Training</b><span>Saturation curves<br>Adstock · Bayesian opt.</span></div>
      <div class="arch-card"><b>Model Validation</b><span>Cross-validation<br>R² score, MAPE</span></div>
    </div>
  </div>
  <div class="arch-arrow">▼</div>

  <!-- Layer 4: Dashboard -->
  <div class="arch-layer">
    <div class="arch-label c-violet">Interactive Dashboard</div>
    <div class="arch-cards">
      <div class="arch-card"><b>Budget Planning</b><span>Optimal allocation<br>Target optimization</span></div>
      <div class="arch-card"><b>Saturation Curves</b><span>Channel efficiency<br>Diminishing returns</span></div>
      <div class="arch-card"><b>Context Calendar</b><span>Event impact<br>Seasonality effects</span></div>
    </div>
  </div>

</div>

<!-- Insights panel (right side) -->
<div class="arch-insights">
  <div class="arch-insights-box">
    <h4>Business Insights</h4>
    <div class="arch-item"><b>Optimal Budget</b><span>Channel-specific allocation</span></div>
    <div class="arch-item"><b>Revenue Forecast</b><span>Predicted returns per scenario</span></div>
    <div class="arch-item"><b>Channel ROI</b><span>Performance ranking</span></div>
    <div class="arch-item"><b>What-If Scenarios</b><span>Budget simulations</span></div>
  </div>
</div>

</div>
'''


# ---------------------------------------------------------------------------
# Plotly dark theme
# ---------------------------------------------------------------------------

PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Inter, sans-serif", color=SLATE_400, size=12),
    title=dict(font=dict(color=SLATE_200, size=16, family="Inter, sans-serif")),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color=SLATE_400),
        title_font=dict(color=SLATE_400),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color=SLATE_400),
        title_font=dict(color=SLATE_400),
    ),
    legend=dict(font=dict(color=SLATE_400), bgcolor="rgba(0,0,0,0)"),
    colorway=DARK_COLORWAY,
    hoverlabel=dict(
        bgcolor=SLATE_800,
        bordercolor="rgba(255,255,255,0.1)",
        font=dict(color=SLATE_200, family="Inter, sans-serif"),
    ),
)


def apply_dark_theme(fig):
    """Apply the DataProf dark theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_DARK_LAYOUT)
    # Also update any secondary axes (subplots)
    for attr in dir(fig.layout):
        if attr.startswith("xaxis") or attr.startswith("yaxis"):
            axis = getattr(fig.layout, attr, None)
            if axis is not None:
                axis.update(
                    gridcolor="rgba(255,255,255,0.05)",
                    zerolinecolor="rgba(255,255,255,0.08)",
                    tickfont=dict(color=SLATE_400),
                    title_font=dict(color=SLATE_400),
                )
    # Update subplot annotation titles to white
    if fig.layout.annotations:
        for ann in fig.layout.annotations:
            ann.update(font=dict(color=SLATE_200, size=14))
    return fig


# ---------------------------------------------------------------------------
# Full CSS
# ---------------------------------------------------------------------------

DATAPROF_CSS = """
<style>
    /* ===== Fonts ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ===== Global ===== */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
        color: #94a3b8;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hide header decoration but keep sidebar toggle button accessible */
    [data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
    }
    [data-testid="stHeader"]::after {
        display: none !important;
    }

    .main .block-container {
        padding-top: 0;
    }

    /* ===== Custom Header ===== */
    .dataprof-header {
        background: rgba(2, 6, 23, 0.8);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1rem 2rem;
        margin: -6rem -1rem 1rem -1rem;
    }

    .dataprof-header-content {
        display: flex;
        align-items: center;
        justify-content: space-between;
        max-width: 1400px;
        margin: 0 auto;
    }

    .dataprof-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .dataprof-logo-mark {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        background: linear-gradient(135deg, #6366f1 0%, #10b981 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.15rem;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.35);
    }

    .dataprof-brand-text {
        font-size: 1.3rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
    }

    .dataprof-subtitle {
        color: #94a3b8;
        font-size: 0.85rem;
        margin: 0;
        font-family: 'Inter', sans-serif;
    }

    /* ===== Typography ===== */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600;
    }

    p, .stMarkdown p, .stMarkdown li {
        color: #94a3b8;
    }

    hr {
        border-color: rgba(255, 255, 255, 0.08);
    }

    /* ===== Sidebar ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
    }

    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] a span,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] span,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a,
    [data-testid="stSidebar"] .stPageLink span,
    [data-testid="stSidebar"] nav span,
    [data-testid="stSidebar"] nav a {
        color: #94a3b8 !important;
        transition: all 0.2s ease;
    }

    /* Active / hovered nav link */
    [data-testid="stSidebar"] a[aria-current="page"],
    [data-testid="stSidebar"] a[aria-current="page"] span,
    [data-testid="stSidebar"] li[data-testid="stSidebarNavLink"][aria-selected="true"] a,
    [data-testid="stSidebar"] li[data-testid="stSidebarNavLink"][aria-selected="true"] span {
        color: #ffffff !important;
        background: rgba(99, 102, 241, 0.12);
        border-radius: 8px;
    }

    [data-testid="stSidebar"] a:hover,
    [data-testid="stSidebar"] a:hover span {
        color: #e2e8f0 !important;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08);
    }

    /* Sidebar selectbox — dark */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] input {
        color: #e2e8f0 !important;
    }

    /* Sidebar metrics */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        padding: 0.8rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stMetricDelta"] {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricDelta"] svg {
        fill: #34d399 !important;
    }

    /* ===== Metric Cards (glass) ===== */
    .stMetric,
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
        max-width: none !important;
    }

    [data-testid="stMetricValue"] > div {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-family: 'JetBrains Mono', monospace;
        white-space: nowrap !important;
        overflow: visible !important;
        max-width: none !important;
    }

    [data-testid="stMetricLabel"] > div {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }

    [data-testid="stMetricDelta"] {
        color: #34d399 !important;
    }

    [data-testid="stMetric"] > div,
    [data-testid="stMetric"] > div > div {
        overflow: visible !important;
        text-overflow: unset !important;
    }

    /* ===== Buttons ===== */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.8rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5);
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }

    /* ===== Inputs ===== */
    .stNumberInput input,
    .stTextInput input,
    [data-baseweb="input"] input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px;
        color: #e2e8f0 !important;
    }

    .stNumberInput input:focus,
    .stTextInput input:focus,
    [data-baseweb="input"] input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }

    .stNumberInput button {
        color: #e2e8f0 !important;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #6366f1;
    }

    /* Radio / Checkbox */
    .stRadio label, .stCheckbox label {
        color: #e2e8f0 !important;
    }

    /* ===== Alerts ===== */
    .stAlert, [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }

    /* ===== Expander ===== */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        color: #e2e8f0 !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.02);
    }

    /* ===== Tabs ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: #94a3b8;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8 !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #6366f1 !important;
    }

    /* ===== DataFrames ===== */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* ===== Charts ===== */
    .js-plotly-plot {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    /* ===== Context banner ===== */
    .context-banner {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #10b981 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }

    /* ===== Caption ===== */
    .stCaption, [data-testid="stCaption"] {
        color: #64748b !important;
    }

    /* ===== Spinner ===== */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }
</style>
"""
