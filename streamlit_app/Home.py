import streamlit as st
from pathlib import Path
import base64
from utils.i18n import t, render_language_selector, get_language

# Page config
st.set_page_config(
    page_title="About - MMM Platform",
    page_icon="📖",
    layout="wide"
)


def get_logo_base64():
    """Load logo and convert to base64 for embedding"""
    logo_path = Path(__file__).parent / "web_resources" / "dataprof.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


# Custom CSS - DataProf style
DATAPROF_CSS = """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Main styling */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide default Streamlit header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom header at top */
    .dataprof-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1rem 2rem;
        margin: -6rem -1rem 1rem -1rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }

    /* Remove top padding from main content */
    .main .block-container {
        padding-top: 0;
    }

    .dataprof-header-content {
        display: flex;
        align-items: center;
        justify-content: space-between;
        max-width: 1400px;
        margin: 0 auto;
    }

    .dataprof-logo-container {
        background: white;
        padding: 8px 16px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .dataprof-logo {
        height: 40px;
        display: block;
    }

    .dataprof-title {
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
    }

    .dataprof-subtitle {
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
        margin: 0;
    }

    /* Card styling */
    .stMetric {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Navigation links in sidebar - make text white */
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] a span,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] span,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a,
    [data-testid="stSidebar"] .stPageLink span,
    [data-testid="stSidebar"] nav span,
    [data-testid="stSidebar"] nav a {
        color: white !important;
    }

    /* Sidebar text - white for labels and markdown */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
    }

    /* Keep selectbox/dropdown text dark on white background */
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] input {
        color: #1a1a2e !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #e94560 0%, #c73e54 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.3);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #c73e54 0%, #a83245 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.4);
    }

    /* Section headers */
    h1, h2, h3 {
        color: #1a1a2e;
        font-weight: 600;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border: none;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
    }

    /* Chart containers */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
</style>
"""

st.markdown(DATAPROF_CSS, unsafe_allow_html=True)


def render_header():
    """Render custom DataProf header"""
    logo_b64 = get_logo_base64()

    if logo_b64:
        logo_html = f'<div class="dataprof-logo-container"><img src="data:image/png;base64,{logo_b64}" class="dataprof-logo" alt="DataProf"></div>'
    else:
        logo_html = '<span style="color: white; font-size: 1.5rem; font-weight: bold;">DataProf</span>'

    st.markdown(f"""
    <div class="dataprof-header">
        <div class="dataprof-header-content">
            {logo_html}
            <div style="text-align: right;">
                <p class="dataprof-title">{t('app.title')}</p>
                <p class="dataprof-subtitle">{t('app.subtitle')}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render DataProf header
render_header()

# Language selector
render_language_selector()
lang = get_language()

# Title
st.title(t('about.title'))
st.markdown(t('about.subtitle'))

# Display architecture diagram and video side by side
col_diagram, col_video = st.columns([3, 1])

with col_diagram:
    diagram_path = Path(__file__).parent.parent / "docs" / "mmm_in_onde_diagram.svg"
    if diagram_path.exists():
        with open(diagram_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        # Scale down SVG significantly to fit in column
        st.markdown(
            f'''<div style="width: 100%; max-width: 100%; overflow: hidden; height: 600px;">
                <div style="transform: scale(0.6); transform-origin: top left; width: 166.67%; height: 100%;">
                    {svg_content}
                </div>
            </div>''',
            unsafe_allow_html=True
        )
    else:
        st.warning(t('about.diagram_not_found'))

with col_video:
    video_path = Path(__file__).parent.parent / "docs" / "video_demo_branded.mp4"
    if video_path.exists():
        st.video(str(video_path))
        st.caption(t('about.video_demo_caption'))
    else:
        st.info(t('about.video_not_found'))

st.markdown("---")

# What is MMM section
st.header(t('about.what_is_mmm'))
st.markdown(t('about.what_is_mmm_description'))

col1, col2 = st.columns(2)

with col1:
    st.subheader(t('about.key_benefits'))
    st.markdown(t('about.key_benefits_list'))

with col2:
    st.subheader(t('about.use_cases'))
    st.markdown(t('about.use_cases_list'))

st.markdown("---")

# Platform capabilities
st.header(t('about.platform_capabilities'))

tab1, tab2, tab3, tab4 = st.tabs([
    t('about.tab_data_sources'),
    t('about.tab_ml_engine'),
    t('about.tab_optimizer'),
    t('about.tab_dashboard')
])

with tab1:
    st.markdown(t('about.data_sources_description'))
    st.markdown("""
    - **💰 Marketing Spend**: Google Ads, Facebook, Yandex Direct, TV, Radio
    - **💵 Revenue Data**: Sales transactions, E-commerce data
    - **📅 Calendar Events**: Holidays, Promotions, Seasonality
    - **📈 Macro Factors**: Economic indicators, Market trends
    - **👥 CRM Data**: Customer base, Retention metrics
    """)

with tab2:
    st.markdown(t('about.ml_engine_description'))
    st.markdown("""
    - **🎯 Feature Selection**: Automatic selection of relevant factors
    - **🔬 Model Training**: Saturation curves, Adstock effects, Bayesian optimization
    - **✅ Model Validation**: Cross-validation, R² score, MAPE, Residual analysis
    """)

with tab3:
    st.markdown(t('about.optimizer_description'))
    st.markdown("""
    - **Budget for Target Revenue**: Calculate optimal budget to achieve revenue goals
    - **Maximize Revenue**: Find best channel allocation for given budget
    - **Scenario Analysis**: Compare multiple budget allocation scenarios
    - **Context-Aware**: Automatically adjusts for holidays, seasonality, and market conditions
    """)

with tab4:
    st.markdown(t('about.dashboard_description'))
    st.markdown("""
    - **💼 Budget Planning**: Optimal allocation, Scenario comparison, Target optimization
    - **📊 Saturation Curves**: Channel efficiency, Diminishing returns, Inflection points
    - **📅 Context Calendar**: Event impact, Multipliers, Seasonality effects
    """)

st.markdown("---")

# How to use
st.header(t('about.how_to_use'))

step_col1, step_col2, step_col3 = st.columns(3)

with step_col1:
    st.markdown(f"""
    ### 1️⃣ {t('about.step1_title')}
    {t('about.step1_description')}
    """)

with step_col2:
    st.markdown(f"""
    ### 2️⃣ {t('about.step2_title')}
    {t('about.step2_description')}
    """)

with step_col3:
    st.markdown(f"""
    ### 3️⃣ {t('about.step3_title')}
    {t('about.step3_description')}
    """)

st.markdown("---")

# Business insights
st.header(t('about.business_insights'))
st.markdown(t('about.business_insights_description'))

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.info(f"""
    **✓ {t('about.insight1_title')}**

    {t('about.insight1_description')}
    """)

    st.info(f"""
    **✓ {t('about.insight2_title')}**

    {t('about.insight2_description')}
    """)

with insight_col2:
    st.info(f"""
    **✓ {t('about.insight3_title')}**

    {t('about.insight3_description')}
    """)

    st.info(f"""
    **✓ {t('about.insight4_title')}**

    {t('about.insight4_description')}
    """)

st.markdown("---")

# Technical details
with st.expander(t('about.technical_details')):
    st.markdown(t('about.technical_details_content'))
