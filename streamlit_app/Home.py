import streamlit as st
from utils.i18n import t, render_language_selector, get_language
from utils.theme import inject_css, render_header, render_sidebar_nav, render_architecture_diagram

# Page config
st.set_page_config(
    page_title="About - MMM Platform",
    page_icon=":material/menu_book:",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_css()
render_sidebar_nav("Home")
render_header()

# Language selector
render_language_selector()
lang = get_language()

# Title
st.title(t('about.title'))
st.markdown(t('about.subtitle'))

# Architecture diagram
st.markdown(render_architecture_diagram(), unsafe_allow_html=True)

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
    - **Marketing Spend** — Google Ads, Facebook, Yandex Direct, TV, Radio
    - **Revenue Data** — Sales transactions, E-commerce data
    - **Calendar Events** — Holidays, Promotions, Seasonality
    - **Macro Factors** — Economic indicators, Market trends
    - **Competitor Data** — Market trends, Competitor pressure index
    """)

with tab2:
    st.markdown(t('about.ml_engine_description'))
    st.markdown("""
    - **Feature Selection** — Automatic selection of relevant factors
    - **Model Training** — Saturation curves, Adstock effects, Bayesian optimization
    - **Model Validation** — Cross-validation, R² score, MAPE, Residual analysis
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
    - **Budget Planning** — Optimal allocation, Scenario comparison, Target optimization
    - **Saturation Curves** — Channel efficiency, Diminishing returns, Inflection points
    - **Context Calendar** — Event impact, Multipliers, Seasonality effects
    """)

st.markdown("---")

# How to use
st.header(t('about.how_to_use'))

step_col1, step_col2, step_col3 = st.columns(3)

with step_col1:
    st.markdown(f"""
    ### Step 1 · {t('about.step1_title')}
    {t('about.step1_description')}
    """)

with step_col2:
    st.markdown(f"""
    ### Step 2 · {t('about.step2_title')}
    {t('about.step2_description')}
    """)

with step_col3:
    st.markdown(f"""
    ### Step 3 · {t('about.step3_title')}
    {t('about.step3_description')}
    """)

st.markdown("---")

# Business insights
st.header(t('about.business_insights'))
st.markdown(t('about.business_insights_description'))

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.info(f"""
    ** {t('about.insight1_title')}**

    {t('about.insight1_description')}
    """)

    st.info(f"""
    ** {t('about.insight2_title')}**

    {t('about.insight2_description')}
    """)

with insight_col2:
    st.info(f"""
    ** {t('about.insight3_title')}**

    {t('about.insight3_description')}
    """)

    st.info(f"""
    ** {t('about.insight4_title')}**

    {t('about.insight4_description')}
    """)

st.markdown("---")

# Technical details
with st.expander(t('about.technical_details')):
    st.markdown(t('about.technical_details_content'))
