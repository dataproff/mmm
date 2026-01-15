"""
Saturation Curves Visualization Page

Shows the saturation (response) curves for each marketing channel
based on the trained MMM model parameters.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import sys
from pathlib import Path
import base64
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_loader import ModelLoader
from utils.i18n import t, render_language_selector, get_currency, fmt_currency

# Page configuration
st.set_page_config(
    page_title="Saturation Curves | DataProf MMM",
    page_icon="📈",
    layout="wide"
)

# DataProf CSS
DATAPROF_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .dataprof-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1rem 2rem;
        margin: -6rem -1rem 1rem -1rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
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
    .dataprof-logo { height: 40px; display: block; }
    .dataprof-title { color: white; font-size: 1.5rem; font-weight: 600; margin: 0; }
    .dataprof-subtitle { color: rgba(255,255,255,0.7); font-size: 0.9rem; margin: 0; }

    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }

    /* Navigation links in sidebar - make text white */
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] a span,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] span,
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a,
    [data-testid="stSidebar"] .stPageLink span,
    [data-testid="stSidebar"] nav span,
    [data-testid="stSidebar"] nav a { color: white !important; }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p { color: white !important; }

    /* Keep selectbox text dark on white background */
    [data-testid="stSidebar"] [data-baseweb="select"] span { color: #1a1a2e !important; }

    .stButton > button {
        background: linear-gradient(135deg, #e94560 0%, #c73e54 100%);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 1.5rem; font-weight: 600;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.3);
    }
    h1, h2, h3 { color: #1a1a2e; font-weight: 600; }

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
    [data-testid="stMetric"] > div,
    [data-testid="stMetric"] > div > div {
        overflow: visible !important;
        text-overflow: unset !important;
    }
</style>
"""
st.markdown(DATAPROF_CSS, unsafe_allow_html=True)


def get_logo_base64():
    logo_path = Path(__file__).parent.parent / "web_resources" / "dataprof.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def render_header():
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
                <p class="dataprof-title">{t('saturation.title')}</p>
                <p class="dataprof-subtitle">{t('app.subtitle')}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load MMM model with caching"""
    # Use absolute path relative to this file
    app_dir = Path(__file__).parent.parent
    model_path = app_dir / ".." / "robyn_training" / "models"
    model_path = model_path.resolve()

    loader = ModelLoader(model_path=str(model_path))
    try:
        results = loader.load_model_results()
        channel_params = loader.get_channel_parameters(results)
        return results, channel_params
    except FileNotFoundError:
        st.error(t('app.model_not_found'))
        st.stop()


DAYS_PER_MONTH = 30  # Standard month for calculations


def calculate_saturation_curve(
    channel_params: dict,
    channel: str,
    spend_range: np.ndarray
) -> tuple:
    """
    Calculate saturation curve data for a channel using Robyn's Hill formula

    Args:
        channel_params: Dictionary of channel parameters
        channel: Channel name
        spend_range: Array of MONTHLY spend values to calculate curves for

    Returns:
        Tuple of (response_values, roi_values, marginal_roi_values) - all monthly
    """
    params = channel_params[channel]

    responses = []
    rois = []
    marginal_rois = []

    prev_response = 0
    prev_spend = 0

    # Get parameters
    theta = params.get('adstock_theta', 0.3)
    alpha = params.get('saturation_alpha', 2.0)
    gamma_normalized = params.get('saturation_gamma', 0.5)  # Normalized 0-1

    # Historical MONTHLY spend for scaling (convert from annual)
    historical_monthly_spend = params.get('total_spend', 100000) / 12
    historical_daily_spend = historical_monthly_spend / DAYS_PER_MONTH
    roi = params.get('roi', 2.0)

    # Calculate inflexion using Robyn's formula (daily basis for saturation calc):
    # inflexion = x_min * (1 - gamma) + x_max * gamma
    x_min = 0
    x_max = historical_daily_spend * 10  # 10x growth potential
    inflexion = x_min * (1 - gamma_normalized) + x_max * gamma_normalized

    # Calculate max_daily_response to match model ROI at historical spend
    adstocked_historical = historical_daily_spend * (1 + theta)
    if adstocked_historical > 0 and inflexion > 0:
        saturation_at_historical = (adstocked_historical ** alpha) / (adstocked_historical ** alpha + inflexion ** alpha)
    else:
        saturation_at_historical = 0.5  # Fallback

    response_at_historical = historical_daily_spend * roi
    if saturation_at_historical > 0:
        max_daily_response = response_at_historical / saturation_at_historical
    else:
        max_daily_response = historical_daily_spend * roi * 10  # Fallback

    for monthly_spend in spend_range:
        # Convert monthly spend to daily for saturation calculation
        daily_spend = monthly_spend / DAYS_PER_MONTH

        # Apply simple adstock (multiplicative)
        adstocked = daily_spend * (1 + theta)

        # Apply Robyn Hill formula: (x^alpha) / (x^alpha + inflexion^alpha)
        if adstocked > 0 and inflexion > 0:
            saturation = (adstocked ** alpha) / (adstocked ** alpha + inflexion ** alpha)
        else:
            saturation = 0

        # Daily response, then scale to monthly
        daily_response = max_daily_response * saturation
        monthly_response = daily_response * DAYS_PER_MONTH
        responses.append(monthly_response)

        # ROI = response / spend (same for daily or monthly)
        current_roi = monthly_response / monthly_spend if monthly_spend > 0 else 0
        rois.append(current_roi)

        # Marginal ROI = delta_response / delta_spend
        if monthly_spend > prev_spend:
            marginal_roi = (monthly_response - prev_response) / (monthly_spend - prev_spend)
        else:
            marginal_roi = 0
        marginal_rois.append(marginal_roi)

        prev_response = monthly_response
        prev_spend = monthly_spend

    return np.array(responses), np.array(rois), np.array(marginal_rois)


def main():
    render_header()

    # Language selector
    render_language_selector()
    st.sidebar.markdown("---")

    st.markdown(f"""
    {t('saturation.description')}

    - **{t('saturation.response_curve')}**: {t('saturation.response_curve_desc')}
    - **{t('saturation.roi_curve')}**: {t('saturation.roi_curve_desc')}
    - **{t('saturation.marginal_roi')}**: {t('saturation.marginal_roi_desc')}
    """)

    # Load model
    with st.spinner(t('common.loading')):
        results, channel_params = load_model()

    channels = list(channel_params.keys())

    # Get baseline info
    baseline_info = results.get('baseline', {})
    baseline_monthly = baseline_info.get('intercept', 0) / 12

    # Show baseline context
    if baseline_info:
        c = get_currency()
        st.info(f"""
        💡 **Note**: These curves show **paid media response only** for each channel.

        Total revenue also includes baseline (organic) of {c}{baseline_monthly:,.0f}/month
        from brand strength, organic traffic, and CRM activities.
        """)

    # Sidebar controls
    st.sidebar.header(t('saturation.settings'))

    # Spend range controls - ensure we include all inflection points
    # All calculations now use MONTHLY values
    max_historical_monthly_spend = max(
        params.get('total_spend', 100000) / 12  # Annual to monthly
        for params in channel_params.values()
    )

    # Calculate max inflection point across all channels (monthly)
    # inflexion is daily, so multiply by 30 to get monthly
    max_inflection_monthly_spend = max(
        (params.get('total_spend', 100000) / 12 / DAYS_PER_MONTH * 10 * params.get('saturation_gamma', 0.5))
        / (1 + params.get('adstock_theta', 0.3)) * DAYS_PER_MONTH
        for params in channel_params.values()
    )

    c = get_currency()
    spend_min = st.sidebar.number_input(
        t('saturation.min_monthly_spend', currency=c),
        min_value=0,
        value=0,
        step=5000
    )

    # Default max should include all inflection points (with 20% buffer)
    default_max = max(
        int(max_historical_monthly_spend * 3),
        int(max_inflection_monthly_spend * 1.2),  # 20% beyond max inflection
        int(spend_min) + 100000
    )
    spend_max = st.sidebar.number_input(
        t('saturation.max_monthly_spend', currency=c),
        min_value=int(spend_min) + 10000,
        value=default_max,
        step=10000
    )

    num_points = st.sidebar.slider(
        t('saturation.data_points'),
        min_value=20,
        max_value=200,
        value=100
    )

    spend_range = np.linspace(spend_min, spend_max, num_points)
    # Ensure we don't start at exactly 0 (causes div by zero)
    spend_range[0] = max(spend_range[0], 1)

    # View selection
    view_mode = st.sidebar.radio(
        "View Mode",
        [t('saturation.all_channels'), t('saturation.single_channel')]
    )

    if view_mode == t('saturation.all_channels'):
        render_all_channels_view(channels, spend_range, channel_params)
    else:
        render_single_channel_view(channels, spend_range, channel_params)


def render_all_channels_view(
    channels: list,
    spend_range: np.ndarray,
    channel_params: dict
):
    """Render comparison view of all channels"""

    st.header(t('saturation.title'))
    c = get_currency()

    # Get translated column names with currency
    spend_col = t('saturation.monthly_spend', currency=c)
    revenue_col = t('saturation.monthly_revenue', currency=c)

    # Calculate curves for all channels
    all_data = []
    for channel in channels:
        responses, rois, marginal_rois = calculate_saturation_curve(
            channel_params, channel, spend_range
        )

        for i, spend in enumerate(spend_range):
            all_data.append({
                t('charts.channel'): channel.replace('_', ' ').title(),
                spend_col: spend,
                revenue_col: responses[i],
                t('optimizer.roi'): rois[i],
                t('saturation.marginal_roi'): marginal_rois[i]
            })

    df = pd.DataFrame(all_data)

    # Response curves
    fig1 = px.line(
        df,
        x=spend_col,
        y=revenue_col,
        color=t('charts.channel'),
        title=t('saturation.saturation_curves_title'),
        labels={spend_col: spend_col, revenue_col: revenue_col}
    )
    fig1.update_layout(height=500)
    st.plotly_chart(fig1, use_container_width=True)

    # ROI curves
    col1, col2 = st.columns(2)

    with col1:
        fig2 = px.line(
            df,
            x=spend_col,
            y=t('optimizer.roi'),
            color=t('charts.channel'),
            title=t('saturation.roi_curves_title'),
            labels={spend_col: spend_col, t('optimizer.roi'): t('saturation.return_on_investment')}
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.line(
            df,
            x=spend_col,
            y=t('saturation.marginal_roi'),
            color=t('charts.channel'),
            title=t('saturation.marginal_roi_curves_title'),
            labels={spend_col: spend_col, t('saturation.marginal_roi'): t('saturation.incremental_roi')}
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    # Channel parameters table
    st.header(t('saturation.channel_parameters'))

    def calc_monthly_inflexion(params):
        """Calculate monthly inflexion point from normalized gamma"""
        hist_monthly = params.get('total_spend', 0) / 12
        hist_daily = hist_monthly / DAYS_PER_MONTH
        gamma_norm = params.get('saturation_gamma', 0.5)
        theta = params.get('adstock_theta', 0.3)
        # Daily inflexion using 10x growth potential
        daily_inflexion = hist_daily * 10 * gamma_norm
        # Convert to monthly spend that produces this inflexion
        monthly_inflexion_spend = (daily_inflexion / (1 + theta)) * DAYS_PER_MONTH
        return monthly_inflexion_spend

    params_df = pd.DataFrame([
        {
            t('charts.channel'): ch.replace('_', ' ').title(),
            t('saturation.historical_monthly_spend'): fmt_currency(params.get('total_spend', 0) / 12),
            t('saturation.adstock'): f"{params.get('adstock_theta', 0):.2f}",
            t('saturation.alpha'): f"{params.get('saturation_alpha', 0):.2f}",
            t('saturation.gamma_norm'): f"{params.get('saturation_gamma', 0):.2f}",
            t('saturation.inflexion_point'): fmt_currency(calc_monthly_inflexion(params)),
            t('optimizer.roi'): f"{params.get('roi', 0):.2f}",
        }
        for ch, params in channel_params.items()
    ])

    st.dataframe(params_df, use_container_width=True, hide_index=True)


def render_single_channel_view(
    channels: list,
    spend_range: np.ndarray,
    channel_params: dict
):
    """Render detailed view for a single channel"""

    # Channel selector
    selected_channel = st.selectbox(
        t('saturation.select_channel'),
        channels,
        format_func=lambda x: x.replace('_', ' ').title()
    )

    params = channel_params[selected_channel]

    # Channel info
    st.header(f"📺 {selected_channel.replace('_', ' ').title()}")

    col1, col2, col3, col4 = st.columns(4)
    historical_monthly = params.get('total_spend', 0) / 12
    historical_daily = historical_monthly / DAYS_PER_MONTH
    gamma_normalized = params.get('saturation_gamma', 0.5)
    theta = params.get('adstock_theta', 0.3)

    # Calculate monthly inflexion point
    # Daily inflexion using 10x potential growth range
    daily_inflexion = historical_daily * 10 * gamma_normalized
    # Convert to monthly spend that produces this inflexion
    monthly_inflexion = (daily_inflexion / (1 + theta)) * DAYS_PER_MONTH

    col1.metric(t('saturation.historical_monthly_spend'), fmt_currency(historical_monthly))
    col2.metric(t('saturation.adstock'), f"{params.get('adstock_theta', 0):.2f}")
    col3.metric(t('saturation.alpha'), f"{params.get('saturation_alpha', 0):.2f}")
    col4.metric(t('saturation.inflexion_point'), fmt_currency(monthly_inflexion))

    # Calculate curves
    responses, rois, marginal_rois = calculate_saturation_curve(
        channel_params, selected_channel, spend_range
    )

    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            t('saturation.response_curve_s'),
            t('saturation.roi_curve'),
            t('saturation.marginal_roi'),
            t('saturation.response_vs_roi')
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.12
    )

    # Response curve
    fig.add_trace(
        go.Scatter(
            x=spend_range,
            y=responses,
            mode='lines',
            name='Response',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )

    # Mark inflection point (gamma)
    # At inflexion point, saturation = 0.5 (50%)
    # monthly_inflexion was already calculated above
    # Calculate the response at inflection point (50% of max_response)
    alpha = params.get('saturation_alpha', 2.0)
    roi = params.get('roi', 2.0)

    # Calculate max_daily_response using same logic as calculate_saturation_curve
    adstocked_historical = historical_daily * (1 + theta)
    if adstocked_historical > 0 and daily_inflexion > 0:
        saturation_at_historical = (adstocked_historical ** alpha) / (adstocked_historical ** alpha + daily_inflexion ** alpha)
    else:
        saturation_at_historical = 0.5
    response_at_historical = historical_daily * roi
    if saturation_at_historical > 0:
        max_daily_response = response_at_historical / saturation_at_historical
    else:
        max_daily_response = historical_daily * roi * 10

    # Monthly response at inflection (50% saturation)
    monthly_response_at_inflexion = max_daily_response * 0.5 * DAYS_PER_MONTH

    fig.add_trace(
        go.Scatter(
            x=[monthly_inflexion],
            y=[monthly_response_at_inflexion],
            mode='markers',
            name='Inflection Point',
            marker=dict(color='red', size=10, symbol='diamond')
        ),
        row=1, col=1
    )

    # ROI curve
    fig.add_trace(
        go.Scatter(
            x=spend_range,
            y=rois,
            mode='lines',
            name='ROI',
            line=dict(color='#2ca02c', width=2)
        ),
        row=1, col=2
    )

    # Marginal ROI curve
    fig.add_trace(
        go.Scatter(
            x=spend_range,
            y=marginal_rois,
            mode='lines',
            name='Marginal ROI',
            line=dict(color='#ff7f0e', width=2)
        ),
        row=2, col=1
    )

    # Response vs ROI scatter
    fig.add_trace(
        go.Scatter(
            x=responses,
            y=rois,
            mode='lines+markers',
            name='Trade-off',
            line=dict(color='#9467bd', width=2),
            marker=dict(size=4)
        ),
        row=2, col=2
    )

    # Update axes labels
    c = get_currency()
    spend_col = t('saturation.monthly_spend', currency=c)
    revenue_col = t('saturation.monthly_revenue', currency=c)

    fig.update_xaxes(title_text=spend_col, row=1, col=1)
    fig.update_yaxes(title_text=revenue_col, row=1, col=1)

    fig.update_xaxes(title_text=spend_col, row=1, col=2)
    fig.update_yaxes(title_text=t('optimizer.roi'), row=1, col=2)

    fig.update_xaxes(title_text=spend_col, row=2, col=1)
    fig.update_yaxes(title_text=t('saturation.marginal_roi'), row=2, col=1)

    fig.update_xaxes(title_text=revenue_col, row=2, col=2)
    fig.update_yaxes(title_text=t('optimizer.roi'), row=2, col=2)

    fig.update_layout(height=800, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    st.markdown(f"### {t('saturation.interpretation')}")

    alpha = params.get('saturation_alpha', 2.0)

    if alpha > 1.5:
        interpretation = t('saturation.strong_s_curve', alpha=alpha)
    elif alpha > 1:
        interpretation = t('saturation.moderate_curve', alpha=alpha)
    else:
        interpretation = t('saturation.concave_curve', alpha=alpha)

    # Add position info
    position_info = t('saturation.current_position', hist=historical_monthly, inflexion=monthly_inflexion, currency=c)
    if historical_monthly < monthly_inflexion:
        position_info += " " + t('saturation.below_inflexion')
    else:
        position_info += " " + t('saturation.above_inflexion')

    st.info(f"{interpretation}\n\n{position_info}")

    # Data table
    st.markdown(f"### {t('saturation.curve_data')}")

    curve_df = pd.DataFrame({
        spend_col: spend_range,
        revenue_col: responses,
        t('optimizer.roi'): rois,
        t('saturation.marginal_roi'): marginal_rois
    })

    # Sample every 10th row for display
    display_df = curve_df.iloc[::10].copy()

    st.dataframe(
        display_df.style.format({
            spend_col: f'{c}{{:,.0f}}',
            revenue_col: f'{c}{{:,.0f}}',
            t('optimizer.roi'): '{:.3f}',
            t('saturation.marginal_roi'): '{:.3f}'
        }),
        use_container_width=True,
        hide_index=True
    )


if __name__ == "__main__":
    main()
