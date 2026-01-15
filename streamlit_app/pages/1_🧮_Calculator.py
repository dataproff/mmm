"""
MMM Budget Optimization Streamlit App

Three main use cases:
1. Optimal budget split for target leads
2. Maximum leads for given budget
3. Scenario analysis with budget constraints
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict
import calendar
import base64
from pathlib import Path

from utils.model_loader import ModelLoader
from utils.robyn_optimizer import RobynOptimizer
from utils.context_calendar import ContextCalendar
from utils.i18n import t, render_language_selector, get_language, get_currency, fmt_currency

# Page configuration
st.set_page_config(
    page_title="MMM Budget Optimizer | DataProf",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_logo_base64():
    """Load logo and convert to base64 for embedding"""
    logo_path = Path(__file__).parent.parent / "web_resources" / "dataprof.png"
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

    /* Prevent metric values from being truncated */
    [data-testid="stMetricValue"] {
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
        font-size: 0.8rem !important;
        white-space: nowrap !important;
        overflow: visible !important;
        max-width: none !important;
    }

    [data-testid="stMetricLabel"] > div {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }

    /* Make metric container not truncate content */
    [data-testid="stMetric"] > div {
        overflow: visible !important;
    }

    [data-testid="stMetric"] > div > div {
        overflow: visible !important;
        text-overflow: unset !important;
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

    /* Sidebar metrics - keep dark background, white text */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: rgba(255,255,255,0.1);
        padding: 0.8rem;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.2);
    }

    [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stMetricDelta"] {
        color: white !important;
    }

    [data-testid="stSidebar"] [data-testid="stMetricDelta"] svg {
        fill: white !important;
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

    /* Metric value */
    [data-testid="stMetricValue"] {
        color: #1a1a2e;
        font-weight: 700;
    }

    /* Success/Info messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-radius: 10px;
    }

    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-radius: 10px;
    }

    /* Chart containers */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }

    /* DataFrame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Input fields */
    .stNumberInput input,
    .stSelectbox select {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }

    /* Context info banner */
    .context-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
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


def get_model_file_hash():
    """Get hash of model file for cache invalidation"""
    import hashlib
    from pathlib import Path
    # Use absolute path relative to this file
    app_dir = Path(__file__).parent.parent
    model_file = app_dir / ".." / "robyn_training" / "models" / "robyn_results.json"
    model_file = model_file.resolve()
    if model_file.exists():
        # Include file modification time for faster invalidation
        mtime = model_file.stat().st_mtime
        content_hash = hashlib.md5(model_file.read_bytes()).hexdigest()
        return f"{content_hash}_{mtime}"
    return None


@st.cache_resource
def load_model(_file_hash=None):
    """Load MMM model with caching (invalidates when file changes)"""
    # Use absolute path relative to this file
    app_dir = Path(__file__).parent.parent
    model_path = app_dir / ".." / "robyn_training" / "models"
    model_path = model_path.resolve()

    loader = ModelLoader(model_path=str(model_path))
    try:
        results = loader.load_model_results()
        channel_params = loader.get_channel_parameters(results)
        return results, channel_params
    except FileNotFoundError as e:
        st.error(f"{t('app.model_not_found')}: {e}")
        st.error(f"Tried to load from: {model_path}")
        st.stop()


@st.cache_resource
def load_context_calendar():
    """Load context calendar with caching"""
    return ContextCalendar()


def create_allocation_chart(allocation: Dict[str, float], title: str = None):
    """Create pie chart for budget allocation"""
    if title is None:
        title = t('charts.budget_allocation')

    # Filter out zero/negative values for pie chart
    filtered_allocation = {k: v for k, v in allocation.items() if v > 0}

    if not filtered_allocation or sum(filtered_allocation.values()) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text=t('charts.no_budget'),
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=title, height=400)
        return fig

    fig = go.Figure(data=[go.Pie(
        labels=list(filtered_allocation.keys()),
        values=list(filtered_allocation.values()),
        hole=0.3,
        textposition='inside',
        textinfo='label+percent',
    )])

    fig.update_layout(
        title=title,
        showlegend=True,
        height=400,
    )

    return fig


def create_budget_table(allocation: Dict[str, float]) -> pd.DataFrame:
    """Create formatted budget allocation table"""
    total = sum(allocation.values())
    if total == 0:
        total = 1  # Avoid division by zero

    df = pd.DataFrame([
        {
            t('charts.channel'): channel,
            t('charts.budget'): fmt_currency(spend, 2),
            t('charts.percentage'): f"{(spend / total * 100):.1f}%"
        }
        for channel, spend in allocation.items()
    ])
    return df


def main():
    """Main application"""

    # Check for cache clear request
    if st.query_params.get("clear_cache"):
        st.cache_resource.clear()
        st.query_params.clear()
        st.rerun()

    # Render DataProf header
    render_header()

    # Language selector at top of sidebar
    render_language_selector()
    st.sidebar.markdown("---")

    # Load model (with cache invalidation when file changes)
    with st.spinner(t('app.loading_model')):
        file_hash = get_model_file_hash()
        results, channel_params = load_model(_file_hash=file_hash)

    # Load context calendar
    context_calendar = load_context_calendar()

    # Initialize optimizer (uses Robyn model if available, otherwise fallback)
    # Use absolute path relative to this file
    app_dir = Path(__file__).parent.parent
    model_path = app_dir / ".." / "robyn_training" / "models"
    model_path = model_path.resolve()

    optimizer = RobynOptimizer(
        model_path=str(model_path),
        channel_params=channel_params
    )

    # Show optimizer method in sidebar (silently use best available method)

    # Sidebar - Month Selection
    st.sidebar.title(f"📅 {t('sidebar.planning_period')}")
    available_months = context_calendar.get_available_months()

    if available_months:
        month_options = []
        for m in available_months:
            year, month_num = m.split('-')
            month_name = calendar.month_name[int(month_num)]
            month_options.append(f"{month_name} {year}")

        selected_month_str = st.sidebar.selectbox(
            t('sidebar.select_month'),
            month_options,
            key="main_month_selector"
        )

        # Parse selected month
        parts = selected_month_str.split()
        month_name = parts[0]
        selected_year = int(parts[1])
        selected_month = list(calendar.month_name).index(month_name)

        # Store in session state
        st.session_state['selected_year'] = selected_year
        st.session_state['selected_month'] = selected_month
        st.session_state['selected_month_name'] = selected_month_str

        # Get context multipliers for selected month
        context_multipliers = context_calendar.calculate_context_multipliers(selected_year, selected_month)
        month_summary = context_calendar.get_month_summary(selected_year, selected_month)

        # Show context summary
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### {t('sidebar.context_for_month')}")
        st.sidebar.markdown(f"""
        - **{t('sidebar.days')}:** {month_summary['n_days']}
        - **{t('sidebar.holidays')}:** {month_summary['n_holidays']}
        - **{t('sidebar.promotion_days')}:** {month_summary['n_promotion_days']}
        - **{t('sidebar.avg_refi_rate')}:** {month_summary['avg_refinancing_rate']:.2f}%
        """)

        # Show impact multiplier
        combined_mult = context_multipliers['combined_multiplier']
        delta_pct = (combined_mult - 1) * 100
        st.sidebar.metric(
            t('sidebar.context_impact'),
            f"{combined_mult:.1%}",
            delta=f"{delta_pct:+.1f}% {t('sidebar.vs_average')}"
        )
    else:
        # Default values if no calendar
        selected_year = 2026
        selected_month = 1
        context_multipliers = {'combined_multiplier': 1.0, 'crm_contribution': 0}
        month_summary = {'n_days': 31}
        st.sidebar.warning(t('sidebar.no_calendar'))

    st.sidebar.markdown("---")

    # Sidebar - Use Case Selection
    st.sidebar.title(t('sidebar.use_case'))
    use_case = st.sidebar.radio(
        t('sidebar.select_scenario'),
        [
            f"1️⃣ {t('use_cases.budget_for_target')}",
            f"2️⃣ {t('use_cases.maximize_revenue')}",
            f"3️⃣ {t('use_cases.scenario_analysis')}"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {t('sidebar.model_info')}")

    # Get baseline info
    baseline_info = results.get('baseline', {})
    baseline_monthly = baseline_info.get('intercept', 0) / 12
    baseline_pct = baseline_info.get('percentage', 0)

    # Debug: Show if baseline is missing
    if not baseline_info:
        st.sidebar.warning("⚠️ Baseline data not found in model. Please retrain the model or check robyn_results.json")

    st.sidebar.info(f"""
    **{t('sidebar.training_date')}:** {results.get('training_date', 'N/A')[:10]}
    **{t('sidebar.channels')}:** {len(channel_params)}
    **{t('sidebar.date_range')}:** {results.get('date_range', {}).get('start', 'N/A')} {t('common.to')} {results.get('date_range', {}).get('end', 'N/A')}
    """)

    # Show baseline revenue
    if baseline_info:
        st.sidebar.metric(
            "📊 Baseline (organic)",
            fmt_currency(baseline_monthly),
            delta=f"{baseline_pct:.1f}% of total",
            help="Monthly revenue without paid advertising (organic traffic, brand, CRM)"
        )

    # Control variables info
    st.sidebar.markdown(f"### {t('sidebar.control_variables')}")
    control_vars = results.get('config', {}).get('model', {}).get('context_vars', [])
    if control_vars:
        st.sidebar.markdown(f"**{t('sidebar.included_in_model')}**")
        for var in control_vars:
            var_display = var.replace('_', ' ').title()
            if 'refinancing' in var.lower():
                st.sidebar.markdown(f"- {var_display} ({t('sidebar.macro')})")
            elif 'email' in var.lower() or 'push' in var.lower():
                st.sidebar.markdown(f"- {var_display} ({t('sidebar.crm')})")
            else:
                st.sidebar.markdown(f"- {var_display}")

    # Get daily context data for optimizer
    context_df = context_calendar.get_daily_context_for_robyn(selected_year, selected_month)

    # Main content area - pass context to optimizers
    context_info = {
        'year': selected_year,
        'month': selected_month,
        'month_name': st.session_state.get('selected_month_name', f"{calendar.month_name[selected_month]} {selected_year}"),
        'multipliers': context_multipliers,
        'summary': month_summary,
        'n_days': month_summary['n_days'],
        'daily_data': context_df,  # Daily context DataFrame for optimizer
        'baseline_monthly': baseline_monthly  # Baseline revenue per month
    }

    if t('use_cases.budget_for_target') in use_case:
        render_target_leads_optimizer(optimizer, channel_params, context_info)

    elif t('use_cases.maximize_revenue') in use_case:
        render_budget_optimizer(optimizer, channel_params, context_info)

    elif t('use_cases.scenario_analysis') in use_case:
        render_scenario_analysis(optimizer, channel_params, context_info)


def render_target_leads_optimizer(optimizer: RobynOptimizer, channel_params: Dict, context_info: Dict):
    """Use Case 1: Find optimal budget to achieve target leads"""

    st.header(f"1️⃣ {t('use_cases.budget_for_target')} - {context_info['month_name']}")

    # Calculate historical monthly contribution for reference
    hist_monthly_revenue = sum(p.get('contribution', 0) for p in channel_params.values()) / 12

    # Get daily context data
    context_df = context_info.get('daily_data')

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(t('optimizer.inputs'))

        st.info(f"📊 {t('optimizer.historical_monthly_revenue')}: **{fmt_currency(hist_monthly_revenue)}**")

        # Show context info
        n_holidays = context_info['summary'].get('n_holidays', 0)
        n_promos = context_info['summary'].get('n_promotion_days', 0)
        if n_holidays > 0 or n_promos > 0:
            st.success(f"📅 {context_info['month_name']}: {n_holidays} {t('sidebar.holidays').lower()}, {n_promos} {t('sidebar.promotion_days').lower()}")

        # Target leads input
        target_leads = st.number_input(
            t('optimizer.target_revenue_for', month=context_info['month_name'], currency=get_currency()),
            min_value=0.0,
            value=min(100000.0, hist_monthly_revenue),
            step=10000.0
        )

        # Channel constraints (optional)
        st.markdown(f"### {t('optimizer.channel_constraints', days=context_info['n_days'])}")
        st.caption(f"💡 {t('optimizer.constraints_hint')}")
        budget_constraints = {}

        # Default constraint values (only applied when enabled)
        default_min_target = 5000.0  # Minimum monthly spend per channel
        default_max_target = 500000.0  # Maximum monthly spend per channel

        for channel in channel_params.keys():
            with st.expander(f"📺 {channel.replace('_', ' ').title()}"):
                # Checkbox to enable/disable constraint for this channel
                enabled = st.checkbox(
                    t('optimizer.enable_constraint'),
                    value=False,
                    key=f"constraint_enabled_{channel}"
                )

                if enabled:
                    col_min, col_max = st.columns(2)
                    min_budget = col_min.number_input(
                        t('optimizer.min', currency=get_currency()),
                        min_value=0.0,
                        value=default_min_target,
                        step=100.0,
                        key=f"min_{channel}"
                    )
                    max_budget = col_max.number_input(
                        t('optimizer.max', currency=get_currency()),
                        min_value=0.0,
                        value=default_max_target,
                        step=1000.0,
                        key=f"max_{channel}"
                    )
                    budget_constraints[channel] = (min_budget, max_budget)
                else:
                    st.caption(t('optimizer.no_constraints'))

        # Optimize button
        if st.button(f"🚀 {t('optimizer.optimize')}", type="primary", use_container_width=True):
            with st.spinner(t('optimizer.optimizing')):
                # Pass context data directly to optimizer
                result = optimizer.optimize_for_target(
                    target_leads,
                    budget_constraints,
                    period_days=context_info['n_days'],
                    context_data=context_df
                )
                # Get detailed context breakdown
                if context_df is not None and not context_df.empty:
                    ctx_details = optimizer.predict_with_context(result['allocation'], context_df)
                    result['base_response'] = ctx_details['base_response']
                    result['context_multiplier'] = ctx_details['context_multiplier']
                    result['crm_contribution'] = ctx_details['crm_contribution']
                st.session_state['optimization_result'] = result

    with col2:
        if 'optimization_result' in st.session_state:
            result = st.session_state['optimization_result']

            st.subheader(t('optimizer.results_for', month=context_info['month_name']))

            # Show warning if target not achievable
            if not result.get('success', True) and 'message' in result:
                st.warning(f"⚠️ {result['message']}")
                st.info(f"💡 {t('optimizer.warning_target')}")

            # Metrics with baseline
            baseline = context_info.get('baseline_monthly', 0)
            paid_media_revenue = result['predicted_leads']
            total_revenue = baseline + paid_media_revenue

            metric_cols = st.columns(4)
            metric_cols[0].metric(t('optimizer.budget_required'), fmt_currency(result['total_budget']))
            metric_cols[1].metric("Paid Media Revenue", fmt_currency(paid_media_revenue))
            metric_cols[2].metric("Baseline (Organic)", fmt_currency(baseline))
            metric_cols[3].metric("Total Revenue", fmt_currency(total_revenue))

            # Achievement
            achievement_pct = (total_revenue/result['target_leads']*100) if result['target_leads'] > 0 else 0
            if achievement_pct < 95:
                st.warning(f"⚠️ Target achievement: {achievement_pct:.1f}% - {t('optimizer.target_not_achievable')}")

            # Show revenue breakdown
            c = get_currency()
            st.caption(f"💡 Total Revenue = {c}{baseline:,.0f} (baseline) + {c}{paid_media_revenue:,.0f} (paid media) = {c}{total_revenue:,.0f}")

            # Show context breakdown if applicable
            if result.get('context_multiplier', 1.0) != 1.0 or result.get('crm_contribution', 0) > 0:
                base = result.get('base_response', paid_media_revenue)
                mult = result.get('context_multiplier', 1.0)
                crm = result.get('crm_contribution', 0)
                st.caption(f"Context adjustment: {c}{base:,.0f} × {mult:.2f} + {c}{crm:,.0f} CRM = {c}{paid_media_revenue:,.0f}")

            # Allocation chart
            st.plotly_chart(
                create_allocation_chart(result['allocation']),
                use_container_width=True
            )

            # Allocation table
            st.subheader(t('optimizer.budget_breakdown'))
            st.dataframe(
                create_budget_table(result['allocation']),
                use_container_width=True,
                hide_index=True
            )


def render_budget_optimizer(optimizer: RobynOptimizer, channel_params: Dict, context_info: Dict):
    """Use Case 2: Maximize leads for given budget"""

    st.header(f"2️⃣ {t('use_cases.maximize_revenue')} - {context_info['month_name']}")

    # Calculate historical monthly spend for reference
    hist_monthly_spend = sum(p.get('total_spend', 0) for p in channel_params.values()) / 12

    # Get daily context data
    context_df = context_info.get('daily_data')

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(t('optimizer.inputs'))

        st.info(f"📊 {t('optimizer.historical_monthly_spend')}: **{fmt_currency(hist_monthly_spend)}**")

        # Show context info
        n_holidays = context_info['summary'].get('n_holidays', 0)
        n_promos = context_info['summary'].get('n_promotion_days', 0)
        if n_holidays > 0 or n_promos > 0:
            st.success(f"📅 {context_info['month_name']}: {n_holidays} {t('sidebar.holidays').lower()}, {n_promos} {t('sidebar.promotion_days').lower()}")

        # Total budget input
        total_budget = st.number_input(
            t('optimizer.budget_for', month=context_info['month_name'], currency=get_currency()),
            min_value=0.0,
            value=min(500000.0, hist_monthly_spend),  # Default to historical or 500K
            step=50000.0
        )

        # Channel constraints (optional)
        st.markdown(f"### {t('optimizer.channel_constraints', days=context_info['n_days'])}")
        st.caption(f"💡 {t('optimizer.constraints_hint')}")

        n_channels = len(channel_params)
        default_min = total_budget * 0.05  # 5% minimum per channel
        default_max = total_budget * 0.50  # 50% maximum per channel

        budget_constraints = {}

        for channel in channel_params.keys():
            with st.expander(f"📺 {channel.replace('_', ' ').title()}"):
                # Checkbox to enable/disable constraint for this channel
                enabled = st.checkbox(
                    t('optimizer.enable_constraint'),
                    value=False,
                    key=f"budget_constraint_enabled_{channel}"
                )

                if enabled:
                    col_min, col_max = st.columns(2)
                    min_budget = col_min.number_input(
                        t('optimizer.min', currency=get_currency()),
                        min_value=0.0,
                        value=default_min,
                        step=100.0,
                        key=f"min_budget_{channel}"
                    )
                    max_budget = col_max.number_input(
                        t('optimizer.max', currency=get_currency()),
                        min_value=0.0,
                        value=default_max,
                        step=1000.0,
                        key=f"max_budget_{channel}"
                    )
                    budget_constraints[channel] = (min_budget, max_budget)
                else:
                    st.caption(t('optimizer.no_constraints_allocation'))

        # Optimize button
        if st.button(f"🚀 {t('optimizer.optimize')}", type="primary", use_container_width=True):
            with st.spinner(t('optimizer.optimizing')):
                # Only apply constraints for channels that have them enabled
                # For channels without constraints, optimizer finds optimal allocation freely
                enforced_constraints = {}
                for ch, (ch_min, ch_max) in budget_constraints.items():
                    enforced_max = max(ch_max, ch_min)  # Max >= min
                    enforced_constraints[ch] = (ch_min, enforced_max)

                # Pass context data directly to optimizer
                result = optimizer.optimize_for_budget(
                    total_budget,
                    enforced_constraints,
                    period_days=context_info['n_days'],
                    context_data=context_df
                )
                # Get detailed context breakdown
                if context_df is not None and not context_df.empty:
                    ctx_details = optimizer.predict_with_context(result['allocation'], context_df)
                    result['base_response'] = ctx_details['base_response']
                    result['context_multiplier'] = ctx_details['context_multiplier']
                    result['crm_contribution'] = ctx_details['crm_contribution']
                result['roi'] = result['predicted_leads'] / total_budget if total_budget > 0 else 0
                st.session_state['budget_optimization_result'] = result

    with col2:
        if 'budget_optimization_result' in st.session_state:
            result = st.session_state['budget_optimization_result']

            st.subheader(t('optimizer.results_for', month=context_info['month_name']))

            # Metrics with baseline
            baseline = context_info.get('baseline_monthly', 0)
            paid_media_revenue = result['predicted_leads']
            total_revenue = baseline + paid_media_revenue
            total_roi = total_revenue / result['total_budget'] if result['total_budget'] > 0 else 0

            metric_cols = st.columns(4)
            metric_cols[0].metric(t('optimizer.budget'), fmt_currency(result['total_budget']))
            metric_cols[1].metric("Paid Media Revenue", fmt_currency(paid_media_revenue))
            metric_cols[2].metric("Total Revenue", fmt_currency(total_revenue))
            metric_cols[3].metric(t('optimizer.roi'), f"{total_roi:.2f}")

            # Show revenue breakdown
            c = get_currency()
            st.caption(f"💡 Total Revenue = {c}{baseline:,.0f} (baseline) + {c}{paid_media_revenue:,.0f} (paid media) = {c}{total_revenue:,.0f}")

            # Show context breakdown if applicable
            if result.get('context_multiplier', 1.0) != 1.0 or result.get('crm_contribution', 0) > 0:
                base = result.get('base_response', paid_media_revenue)
                mult = result.get('context_multiplier', 1.0)
                crm = result.get('crm_contribution', 0)
                st.caption(f"Context adjustment: {c}{base:,.0f} × {mult:.2f} + {c}{crm:,.0f} CRM = {c}{paid_media_revenue:,.0f}")

            # Allocation chart
            st.plotly_chart(
                create_allocation_chart(result['allocation']),
                use_container_width=True
            )

            # Allocation table
            st.subheader(t('optimizer.budget_breakdown'))
            st.dataframe(
                create_budget_table(result['allocation']),
                use_container_width=True,
                hide_index=True
            )


def render_scenario_analysis(optimizer: RobynOptimizer, channel_params: Dict, context_info: Dict):
    """Use Case 3: Scenario analysis across budget ranges"""

    st.header(f"3️⃣ {t('use_cases.scenario_analysis')} - {context_info['month_name']}")

    # Calculate historical monthly spend for reference
    hist_monthly_spend = sum(p.get('total_spend', 0) for p in channel_params.values()) / 12

    # Get daily context data
    context_df = context_info.get('daily_data')

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader(t('scenario.settings'))

        st.info(f"📊 {t('optimizer.historical_monthly_spend')}: **{fmt_currency(hist_monthly_spend)}**")

        # Show context info
        n_holidays = context_info['summary'].get('n_holidays', 0)
        n_promos = context_info['summary'].get('n_promotion_days', 0)
        if n_holidays > 0 or n_promos > 0:
            st.success(f"📅 {context_info['month_name']}: {n_holidays} {t('sidebar.holidays').lower()}, {n_promos} {t('sidebar.promotion_days').lower()}")

        st.caption(f"💡 {t('scenario.hint')}")

        # Budget range
        min_budget = st.number_input(
            t('scenario.min_budget', month=context_info['month_name'], currency=get_currency()),
            min_value=0.0,
            value=100000.0,
            step=50000.0
        )

        max_budget = st.number_input(
            t('scenario.max_budget', month=context_info['month_name'], currency=get_currency()),
            min_value=min_budget,
            value=min(1000000.0, hist_monthly_spend * 1.5),  # Up to 1.5x historical
            step=50000.0
        )

        steps = st.slider(
            t('scenario.num_scenarios'),
            min_value=5,
            max_value=20,
            value=10,
            help=t('scenario.scenarios_hint')
        )

        # Run analysis button
        if st.button(f"📈 {t('scenario.run_analysis')}", type="primary", use_container_width=True):
            with st.spinner(t('scenario.running')):
                # Pass context data directly to scenario analysis
                scenarios = optimizer.scenario_analysis(
                    (min_budget, max_budget),
                    steps,
                    period_days=context_info['n_days'],
                    context_data=context_df
                )
                # Get context breakdown for display
                if context_df is not None and not context_df.empty and scenarios:
                    ctx_details = optimizer.predict_with_context(scenarios[0]['allocation'], context_df)
                    context_mult = ctx_details['context_multiplier']
                    crm_contribution = ctx_details['crm_contribution']
                else:
                    context_mult = 1.0
                    crm_contribution = 0

                # Add baseline and calculate total revenue/ROI
                baseline = context_info.get('baseline_monthly', 0)
                for s in scenarios:
                    s['paid_media_revenue'] = s['predicted_leads']
                    s['total_revenue'] = baseline + s['predicted_leads']
                    s['roi'] = s['total_revenue'] / s['total_budget'] if s['total_budget'] > 0 else 0

                st.session_state['scenarios'] = scenarios
                st.session_state['scenarios_context'] = {
                    'month_name': context_info['month_name'],
                    'context_mult': context_mult,
                    'crm_contribution': crm_contribution,
                    'baseline': baseline
                }

    with col2:
        if 'scenarios' in st.session_state:
            scenarios = st.session_state['scenarios']
            scenario_context = st.session_state.get('scenarios_context', {})

            st.subheader(t('scenario.results_for', month=scenario_context.get('month_name', context_info['month_name'])))

            # Prepare data for plotting (now using total_revenue which includes baseline)
            c = get_currency()
            revenue_col = t('scenario.revenue_axis', currency=c).replace(f' ({c})', '')
            baseline = scenario_context.get('baseline', 0)

            scenario_df = pd.DataFrame([
                {
                    t('optimizer.budget'): s['total_budget'],
                    revenue_col: s['total_revenue'],
                    'Paid Media': s['paid_media_revenue'],
                    'Baseline': baseline,
                    t('optimizer.roi'): s['roi']
                }
                for s in scenarios
            ])

            # Response curve
            fig1 = px.line(
                scenario_df,
                x=t('optimizer.budget'),
                y=revenue_col,
                title=t('scenario.response_curve_title', month=scenario_context.get('month_name', '')),
                markers=True
            )
            fig1.update_layout(
                height=400,
                xaxis_title=t('scenario.budget_axis', currency=c),
                yaxis_title=t('scenario.revenue_axis', currency=c)
            )
            st.plotly_chart(fig1, use_container_width=True)

            # ROI curve
            fig2 = px.line(
                scenario_df,
                x=t('optimizer.budget'),
                y=t('optimizer.roi'),
                title=t('scenario.roi_curve_title'),
                markers=True
            )
            fig2.update_layout(
                height=400,
                xaxis_title=t('scenario.budget_axis', currency=c),
                yaxis_title=t('scenario.roi_axis')
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Show baseline info
            st.info(f"💡 All scenarios include baseline revenue of {c}{baseline:,.0f}/month (organic traffic, brand, CRM)")

            # Show context adjustment info
            if scenario_context.get('context_mult', 1.0) != 1.0 or scenario_context.get('crm_contribution', 0) > 0:
                st.caption(t('scenario.context_note', mult=scenario_context.get('context_mult', 1), crm=scenario_context.get('crm_contribution', 0), currency=c))

            # Scenario table
            st.subheader(t('scenario.details'))
            st.dataframe(
                scenario_df.style.format({
                    t('optimizer.budget'): f'{c}{{:,.0f}}',
                    revenue_col: f'{c}{{:,.0f}}',
                    t('optimizer.roi'): '{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )


if __name__ == "__main__":
    main()
