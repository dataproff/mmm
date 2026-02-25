"""
Context Calendar Page

Displays the context calendar with holidays, promotions, and other variables
that affect MMM predictions. Data displayed by week.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import calendar

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.context_calendar import ContextCalendar
from utils.i18n import t, render_language_selector, get_currency, fmt_currency
from utils.theme import inject_css, render_header, render_sidebar_nav, apply_dark_theme

st.set_page_config(
    page_title="Context Calendar | DataProf MMM",
    page_icon=":material/calendar_month:",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_css()
render_sidebar_nav("Context Calendar")


def render_page_header():
    render_header('calendar.title', 'calendar.description')


render_page_header()


@st.cache_resource
def load_calendar():
    """Load context calendar with caching"""
    return ContextCalendar()


def create_calendar_heatmap(df: pd.DataFrame, year: int, month: int):
    """Create a calendar-style heatmap for the month"""
    if df.empty:
        return None

    first_day = datetime(year, month, 1)
    num_days = calendar.monthrange(year, month)[1]
    first_weekday = first_day.weekday()

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
                color_code = is_holiday + (is_promotion * 2)

                cal_data.append({
                    'week': week, 'weekday': weekday, 'day': day,
                    'is_holiday': is_holiday, 'is_promotion': is_promotion,
                    'color_code': color_code
                })
                day += 1

    cal_df = pd.DataFrame(cal_data)
    fig = go.Figure()

    for _, row in cal_df.iterrows():
        if pd.notna(row['day']):
            if row.get('color_code', 0) == 3:
                color = '#a78bfa'
                text_color = 'white'
            elif row.get('color_code', 0) == 2:
                color = '#6366f1'
                text_color = 'white'
            elif row.get('color_code', 0) == 1:
                color = '#ef4444'
                text_color = 'white'
            else:
                color = 'rgba(255,255,255,0.06)'
                text_color = '#94a3b8'

            fig.add_shape(
                type="rect",
                x0=row['weekday'] - 0.45, x1=row['weekday'] + 0.45,
                y0=5 - row['week'] - 0.45, y1=5 - row['week'] + 0.45,
                fillcolor=color,
                line=dict(color='rgba(255,255,255,0.1)', width=1)
            )

            fig.add_annotation(
                x=row['weekday'], y=5 - row['week'],
                text=str(int(row['day'])),
                showarrow=False,
                font=dict(size=14, color=text_color)
            )

    fig.update_layout(
        title=f"{calendar.month_name[month]} {year}",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(7)),
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            range=[-0.5, 6.5]
        ),
        yaxis=dict(showticklabels=False, range=[-0.5, 5.5]),
        height=400,
        showlegend=False
    )
    apply_dark_theme(fig)
    return fig


def main():
    render_language_selector()
    calendar_data = load_calendar()

    st.sidebar.title(t('sidebar.select_month'))

    available_months = calendar_data.get_available_months()
    if not available_months:
        st.error(t('sidebar.no_calendar'))
        return

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

    parts = selected_month_str.split()
    month_name = parts[0]
    year = int(parts[1])
    month = list(calendar.month_name).index(month_name)

    st.session_state['selected_year'] = year
    st.session_state['selected_month'] = month
    st.session_state['selected_month_name'] = f"{month_name} {year}"

    month_df = calendar_data.get_month_data(year, month)
    weekly_df = calendar_data.get_weekly_data(year, month)
    summary = calendar_data.get_month_summary(year, month)
    multipliers = calendar_data.calculate_context_multipliers(year, month)

    # Calendar heatmap + month overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(t('calendar.calendar_view'))
        fig = create_calendar_heatmap(month_df, year, month)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
**{t('calendar.legend')}:**
- <span style="color:#ef4444">&#9632;</span> **{t('calendar.legend_holiday')}**
- <span style="color:#6366f1">&#9632;</span> **{t('calendar.legend_promotion')}**
- <span style="color:#a78bfa">&#9632;</span> **{t('calendar.legend_both')}**
- <span style="color:rgba(255,255,255,0.3)">&#9632;</span> **{t('calendar.legend_regular')}**
        """, unsafe_allow_html=True)

    with col2:
        st.subheader(t('calendar.month_overview'))
        st.metric(t('calendar.total_days'), summary['n_days'])
        st.metric(t('calendar.holiday_days'), summary['n_holidays'])
        st.metric(t('calendar.promotion_days'), summary['n_promotion_days'])

        if summary['holiday_names']:
            st.markdown(f"**{t('calendar.holidays_list')}:**")
            for h in summary['holiday_names']:
                st.markdown(f"- {h}")

        if summary['promotion_names']:
            st.markdown(f"**{t('calendar.promotions_list')}:**")
            for p in summary['promotion_names']:
                st.markdown(f"- {p}")

    st.markdown("---")

    # Context Variables
    st.header(t('calendar.context_variables'))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(t('calendar.fed_funds_rate'))
        st.metric(
            t('calendar.average_rate'),
            f"{summary['avg_fed_funds_rate']:.2f}%",
            delta=f"{summary['avg_fed_funds_rate'] - 3.84:.2f}% {t('calendar.vs_baseline')}"
        )
        st.caption(f"6-month MA baseline: 3.84%")

    with col2:
        st.subheader(t('calendar.competitor_pressure'))
        st.metric(
            t('calendar.avg_index'),
            f"{summary['avg_competitor_pressure']:.1f}",
            delta=f"{summary['avg_competitor_pressure'] - 50:.1f} {t('calendar.vs_baseline')}"
        )
        st.caption(f"{t('calendar.range')}: {summary['min_competitor_pressure']:.0f} – {summary['max_competitor_pressure']:.0f}")

    st.markdown("---")

    # Impact Multipliers
    st.header(t('calendar.impact_multipliers'))
    st.markdown(t('calendar.multipliers_description'))

    mult_cols = st.columns(5)

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
        t('calendar.fed_funds_effect'),
        f"{multipliers['fed_funds_multiplier']:.2%}",
        delta=f"{(multipliers['fed_funds_multiplier'] - 1) * 100:.1f}%"
    )

    mult_cols[3].metric(
        t('calendar.competitor_effect'),
        f"{multipliers['competitor_multiplier']:.2%}",
        delta=f"{(multipliers['competitor_multiplier'] - 1) * 100:.1f}%"
    )

    mult_cols[4].metric(
        t('calendar.combined_multiplier'),
        f"{multipliers['combined_multiplier']:.2%}",
        delta=f"{(multipliers['combined_multiplier'] - 1) * 100:.1f}%"
    )

    st.markdown("---")

    # Weekly Data Table
    st.header(t('calendar.weekly_context'))

    if not weekly_df.empty:
        display_df = weekly_df.copy()
        display_df = display_df.rename(columns={
            'week_label': t('calendar.week'),
            'days_in_week': t('calendar.days_count'),
            'holidays': t('calendar.holiday_days'),
            'promotion_days': t('calendar.promotion_days'),
            'fed_funds_rate': t('calendar.fed_funds_rate'),
            'competitor_pressure_index': t('calendar.competitor_pressure'),
            'holiday_names': t('calendar.holiday_name'),
            'promotion_names': t('calendar.promotion_name'),
        })
        display_df = display_df.drop(columns=['week_start'], errors='ignore')

        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Charts
    st.header(t('calendar.trends'))

    if not weekly_df.empty:
        # Competitor pressure trend
        fig_pressure = px.line(
            weekly_df,
            x='week_label',
            y='competitor_pressure_index',
            title=t('calendar.competitor_trend'),
            markers=True
        )
        fig_pressure.update_layout(height=300, xaxis_title=t('calendar.week'), yaxis_title=t('calendar.index_value'))
        apply_dark_theme(fig_pressure)
        st.plotly_chart(fig_pressure, use_container_width=True)


if __name__ == "__main__":
    main()
