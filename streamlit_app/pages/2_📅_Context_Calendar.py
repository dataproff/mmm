"""
Context Calendar Page

Displays the context calendar with holidays, promotions, and other variables
that affect MMM predictions.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar
import base64

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.context_calendar import ContextCalendar
from utils.i18n import t, render_language_selector, get_currency, fmt_currency

st.set_page_config(
    page_title="Context Calendar | DataProf MMM",
    page_icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAAsTAAALEwEAmpwYAAAB8ElEQVR4nO2WPUsDQRCGn4sRFBsLC7GwsLCwsLCw0B+gtbWVnY2NjY2NjY2NjY2NjZWF/gD/gJWFhYV/wEJERPwiJjqySQ4ul9y5d5dLkQcWdm92Zt6d2dkLhBBCCCGE/w4VoA84B16Bb+ATuAa2gWagPqL9KLAJ3AJfwA9wB+wDXUA9UBPKhwJwDLwB38ATcAzsAF1AA1ANlIXQXgCOgBfgE3gAjoADoBuoBqqAshDai8AhcA+8A4/AIdAP9ADVQCVQGkJ7ATgA7oA34AE4APqBXqAGqAAqQmgvBPqBG+AFeAYOgQGgD6gFKoBy/J8A7AM3wDPwBBwAg0A/UA/UAuX4/wSwF7gGnoAn4AAYAgaABqAOKMf/KYC9wBXwCDwC+8AwMAg0AvVAOf5PAewBLoEH4AHYB0aAIaAJaADK8H8SYDdwAdwD98AeMAqMAM1AI1CG/5MAu4AL4Ba4A/aAMWAUaAGagDL8nwTYBZwDN8AtsBuYAMaAVqAZKMX/SYCdwBlwDdwAu4FJYBxoA1qAUvyfBNgJnAJXwDWwG5gCJoB2oBUowf9JgB3ACXAJXAe7gGlgEugA2oASIITvBNgOnAAXQVdgBpgCOoEOoJjgdwJsB46B8yB+MAeYBbqBLqCY4HcCbAOOgLMg/rEAzAPdQDfwC7y/hFqfqDdLAAAAAElFTkSuQmCC",
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
    [data-testid="stSidebar"] [data-baseweb="select"] div[data-baseweb="select"] { background: white; }

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
                <p class="dataprof-title">{t('calendar.title')}</p>
                <p class="dataprof-subtitle">{t('calendar.description')}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


render_header()


@st.cache_resource
def load_calendar():
    """Load context calendar with caching"""
    return ContextCalendar()


def create_calendar_heatmap(df: pd.DataFrame, year: int, month: int):
    """Create a calendar-style heatmap for the month"""
    if df.empty:
        return None

    # Get first day of month and number of days
    first_day = datetime(year, month, 1)
    num_days = calendar.monthrange(year, month)[1]
    first_weekday = first_day.weekday()  # Monday = 0

    # Create calendar grid (6 weeks x 7 days)
    cal_data = []
    day = 1

    for week in range(6):
        for weekday in range(7):
            if week == 0 and weekday < first_weekday:
                cal_data.append({'week': week, 'weekday': weekday, 'day': None,
                               'is_holiday': 0, 'is_promotion': 0})
            elif day > num_days:
                cal_data.append({'week': week, 'weekday': weekday, 'day': None,
                               'is_holiday': 0, 'is_promotion': 0})
            else:
                date_str = f"{year}-{month:02d}-{day:02d}"
                day_data = df[df['date'].dt.strftime('%Y-%m-%d') == date_str]

                is_holiday = int(day_data['is_holiday'].iloc[0]) if not day_data.empty else 0
                is_promotion = int(day_data['is_promotion'].iloc[0]) if not day_data.empty else 0

                # Color code: 0=normal, 1=holiday, 2=promotion, 3=both
                color_code = is_holiday + (is_promotion * 2)

                cal_data.append({
                    'week': week,
                    'weekday': weekday,
                    'day': day,
                    'is_holiday': is_holiday,
                    'is_promotion': is_promotion,
                    'color_code': color_code
                })
                day += 1

    cal_df = pd.DataFrame(cal_data)

    # Create heatmap
    fig = go.Figure()

    # Add cells
    for _, row in cal_df.iterrows():
        if pd.notna(row['day']):
            # Determine color
            if row.get('color_code', 0) == 3:
                color = '#9b59b6'  # Purple for both
                text_color = 'white'
            elif row.get('color_code', 0) == 2:
                color = '#3498db'  # Blue for promotion
                text_color = 'white'
            elif row.get('color_code', 0) == 1:
                color = '#e74c3c'  # Red for holiday
                text_color = 'white'
            else:
                color = '#ecf0f1'  # Light gray for normal
                text_color = 'black'

            fig.add_shape(
                type="rect",
                x0=row['weekday'] - 0.45, x1=row['weekday'] + 0.45,
                y0=5 - row['week'] - 0.45, y1=5 - row['week'] + 0.45,
                fillcolor=color,
                line=dict(color='white', width=2)
            )

            fig.add_annotation(
                x=row['weekday'],
                y=5 - row['week'],
                text=str(int(row['day'])),
                showarrow=False,
                font=dict(size=14, color=text_color)
            )

    # Configure layout
    fig.update_layout(
        title=f"{calendar.month_name[month]} {year}",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(7)),
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            range=[-0.5, 6.5]
        ),
        yaxis=dict(
            showticklabels=False,
            range=[-0.5, 5.5]
        ),
        height=400,
        showlegend=False
    )

    return fig


def main():
    # Language selector at top of sidebar
    render_language_selector()

    calendar_data = load_calendar()

    # Month selector in sidebar
    st.sidebar.title(t('sidebar.select_month'))

    available_months = calendar_data.get_available_months()
    if not available_months:
        st.error(t('sidebar.no_calendar'))
        return

    # Parse available months
    month_options = []
    for m in available_months:
        year, month = m.split('-')
        month_name = calendar.month_name[int(month)]
        month_options.append(f"{month_name} {year}")

    selected_month_str = st.sidebar.selectbox(
        t('sidebar.select_month'),
        month_options,
        key="calendar_month_selector"
    )

    # Parse selected month
    parts = selected_month_str.split()
    month_name = parts[0]
    year = int(parts[1])
    month = list(calendar.month_name).index(month_name)

    # Store in session state for other pages
    st.session_state['selected_year'] = year
    st.session_state['selected_month'] = month
    st.session_state['selected_month_name'] = f"{month_name} {year}"

    # Get month data
    month_df = calendar_data.get_month_data(year, month)
    summary = calendar_data.get_month_summary(year, month)
    multipliers = calendar_data.calculate_context_multipliers(year, month)

    # Display calendar heatmap
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(t('calendar.calendar_view'))
        fig = create_calendar_heatmap(month_df, year, month)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Legend
        st.markdown(f"""
        **{t('calendar.legend')}:**
        - 🔴 **{t('calendar.legend_holiday')}**
        - 🔵 **{t('calendar.legend_promotion')}**
        - 🟣 **{t('calendar.legend_both')}**
        - ⬜ **{t('calendar.legend_regular')}**
        """)

    with col2:
        st.subheader(t('calendar.month_overview'))

        # Key metrics
        st.metric(t('calendar.total_days'), summary['n_days'])
        st.metric(t('calendar.holiday_days'), summary['n_holidays'])
        st.metric(t('calendar.promotion_days'), summary['n_promotion_days'])

        # Holidays list
        if summary['holiday_names']:
            st.markdown(f"**{t('calendar.holidays_list')}:**")
            for h in summary['holiday_names']:
                st.markdown(f"- {h}")

        # Promotions list
        if summary['promotion_names']:
            st.markdown(f"**{t('calendar.promotions_list')}:**")
            for p in summary['promotion_names']:
                st.markdown(f"- {p}")

    st.markdown("---")

    # Context Variables Section
    st.header(t('calendar.context_variables'))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader(t('calendar.refinancing_rate'))
        st.metric(
            t('calendar.average_rate'),
            f"{summary['avg_refinancing_rate']:.2f}%",
            delta=f"{summary['avg_refinancing_rate'] - 7.5:.2f}% {t('calendar.vs_baseline')}"
        )
        st.caption(f"{t('calendar.range')}: {summary['min_refinancing_rate']:.2f}% - {summary['max_refinancing_rate']:.2f}%")

    with col2:
        st.subheader(t('calendar.email_crm'))
        st.metric(t('calendar.total_sends'), f"{summary['total_email_sends']:,}")
        st.metric(t('calendar.total_clicks'), f"{summary['total_email_clicks']:,}")
        click_rate = summary['total_email_clicks'] / summary['total_email_sends'] * 100 if summary['total_email_sends'] > 0 else 0
        st.caption(f"{t('calendar.click_rate')}: {click_rate:.2f}%")

    with col3:
        st.subheader(t('calendar.push_crm'))
        st.metric(t('calendar.total_sends'), f"{summary['total_push_sends']:,}")
        st.metric(t('calendar.total_clicks'), f"{summary['total_push_clicks']:,}")
        click_rate = summary['total_push_clicks'] / summary['total_push_sends'] * 100 if summary['total_push_sends'] > 0 else 0
        st.caption(f"{t('calendar.click_rate')}: {click_rate:.2f}%")

    st.markdown("---")

    # Impact Multipliers Section
    st.header(t('calendar.impact_multipliers'))
    st.markdown(t('calendar.multipliers_description'))

    mult_cols = st.columns(4)

    mult_cols[0].metric(
        t('calendar.holiday_effect'),
        f"{multipliers['holiday_multiplier']:.2%}",
        delta=f"+{(multipliers['holiday_multiplier'] - 1) * 100:.1f}%"
    )

    mult_cols[1].metric(
        t('calendar.promotion_effect'),
        f"{multipliers['promotion_multiplier']:.2%}",
        delta=f"+{(multipliers['promotion_multiplier'] - 1) * 100:.1f}%"
    )

    mult_cols[2].metric(
        t('calendar.refinancing_effect'),
        f"{multipliers['refinancing_multiplier']:.2%}",
        delta=f"{(multipliers['refinancing_multiplier'] - 1) * 100:.1f}%"
    )

    mult_cols[3].metric(
        t('calendar.combined_multiplier'),
        f"{multipliers['combined_multiplier']:.2%}",
        delta=f"{(multipliers['combined_multiplier'] - 1) * 100:.1f}%"
    )

    st.info(f"**{t('calendar.crm_contribution')}:** {fmt_currency(multipliers['crm_contribution'])} ({t('calendar.additive')})")

    st.markdown("---")

    # Daily Data Table
    st.header(t('calendar.daily_context'))

    if not month_df.empty:
        display_df = month_df.copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df['is_holiday'] = display_df['is_holiday'].map({0: '', 1: '🎄'})
        display_df['is_promotion'] = display_df['is_promotion'].map({0: '', 1: '🏷️'})

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'date': t('calendar.date'),
                'is_holiday': t('calendar.is_holiday'),
                'is_promotion': t('calendar.is_promotion'),
                'refinancing_rate': st.column_config.NumberColumn(t('calendar.refinancing_rate'), format="%.2f%%"),
                'email_sends': st.column_config.NumberColumn(t('calendar.email_sends'), format="%d"),
                'email_clicks': st.column_config.NumberColumn(t('calendar.email_clicks'), format="%d"),
                'push_sends': st.column_config.NumberColumn(t('calendar.push_sends'), format="%d"),
                'push_clicks': st.column_config.NumberColumn(t('calendar.push_clicks'), format="%d"),
                'holiday_name': t('calendar.holiday_name'),
                'promotion_name': t('calendar.promotion_name'),
            }
        )

    # Charts
    st.header(t('calendar.trends'))

    if not month_df.empty:
        # Refinancing rate trend
        fig_rate = px.line(
            month_df,
            x='date',
            y='refinancing_rate',
            title=t('calendar.refi_rate_trend'),
            markers=True
        )
        fig_rate.update_layout(height=300)
        st.plotly_chart(fig_rate, use_container_width=True)

        # CRM metrics
        crm_df = month_df.melt(
            id_vars=['date'],
            value_vars=['email_sends', 'email_clicks', 'push_sends', 'push_clicks'],
            var_name='Metric',
            value_name='Value'
        )

        fig_crm = px.line(
            crm_df,
            x='date',
            y='Value',
            color='Metric',
            title=t('calendar.crm_metrics_trend')
        )
        fig_crm.update_layout(height=400)
        st.plotly_chart(fig_crm, use_container_width=True)


if __name__ == "__main__":
    main()
