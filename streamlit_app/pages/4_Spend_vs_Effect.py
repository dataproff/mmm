"""
Spend Share vs Effect Share Page

Scatter plot comparing each channel's budget share against its revenue effect share.
Points above the diagonal are efficient (effect > spend); below are inefficient.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_loader import ModelLoader
from utils.i18n import t, render_language_selector, get_currency, fmt_currency
from utils.theme import (
    inject_css, render_header, render_sidebar_nav, apply_dark_theme,
    INDIGO_500, EMERALD_400, SLATE_400, SLATE_500,
)

st.set_page_config(
    page_title="Spend vs Effect | DataProf MMM",
    page_icon=":material/bubble_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_css()
render_sidebar_nav("Spend vs Effect")


@st.cache_resource
def load_model():
    app_dir = Path(__file__).parent.parent
    model_path = (app_dir / ".." / "robyn_training" / "models").resolve()
    loader = ModelLoader(model_path=str(model_path))
    try:
        results = loader.load_model_results()
        channel_params = loader.get_channel_parameters(results)
        return results, channel_params
    except FileNotFoundError:
        return None, None


def build_channel_df(channel_params: dict) -> pd.DataFrame:
    """Build DataFrame with spend share, effect share, and efficiency index."""
    rows = []
    for ch, params in channel_params.items():
        rows.append({
            "channel": ch.replace("_", " ").title(),
            "total_spend": params.get("total_spend", 0),
            "contribution": params.get("contribution", 0),
            "contribution_pct": params.get("contribution_pct", 0),
            "roi": params.get("roi", 0),
        })

    df = pd.DataFrame(rows)
    total_spend = df["total_spend"].sum()
    df["spend_share"] = (df["total_spend"] / total_spend * 100) if total_spend > 0 else 0
    df["effect_share"] = df["contribution_pct"] * 100
    df["efficiency_index"] = (df["effect_share"] / df["spend_share"]).round(2)
    return df


def render_scatter(df: pd.DataFrame):
    """Render the scatter plot with diagonal reference line."""
    c = get_currency()

    fig = px.scatter(
        df,
        x="spend_share",
        y="effect_share",
        color="efficiency_index",
        size="spend_share",
        size_max=45,
        color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
        hover_name="channel",
        hover_data={
            "spend_share": ":.1f",
            "effect_share": ":.1f",
            "efficiency_index": ":.2f",
        },
        labels={
            "spend_share": t("spend_effect.spend_share"),
            "effect_share": t("spend_effect.effect_share"),
            "efficiency_index": t("spend_effect.efficiency_index"),
        },
    )

    # Diagonal line: spend = effect
    max_val = max(df["spend_share"].max(), df["effect_share"].max()) + 5
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        line=dict(color=SLATE_500, width=2, dash="dash"),
        name=t("spend_effect.diagonal_label"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Zone annotations
    fig.add_annotation(
        x=max_val * 0.25, y=max_val * 0.75,
        text=t("spend_effect.efficient_zone"),
        showarrow=False,
        font=dict(size=14, color=EMERALD_400),
        opacity=0.5,
    )
    fig.add_annotation(
        x=max_val * 0.75, y=max_val * 0.25,
        text=t("spend_effect.inefficient_zone"),
        showarrow=False,
        font=dict(size=14, color="#ef4444"),
        opacity=0.5,
    )

    # Channel name labels next to points
    for _, row in df.iterrows():
        fig.add_annotation(
            x=row["spend_share"],
            y=row["effect_share"],
            text=row["channel"],
            showarrow=False,
            yshift=20,
            font=dict(size=11, color=SLATE_400),
        )

    fig.update_layout(
        height=550,
        xaxis=dict(range=[0, max_val], dtick=5),
        yaxis=dict(range=[0, max_val], dtick=5),
        coloraxis_colorbar=dict(title=t("spend_effect.efficiency_index")),
    )

    apply_dark_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def render_summary_table(df: pd.DataFrame):
    """Render the channel summary table."""
    c = get_currency()

    display = df[[
        "channel", "spend_share", "effect_share",
        "efficiency_index", "total_spend", "contribution", "roi"
    ]].copy()

    display.columns = [
        t("spend_effect.channel"),
        t("spend_effect.spend_share"),
        t("spend_effect.effect_share"),
        t("spend_effect.efficiency_index"),
        t("spend_effect.total_spend"),
        t("spend_effect.contribution"),
        t("spend_effect.roi"),
    ]

    st.dataframe(
        display.style.format({
            t("spend_effect.spend_share"): "{:.1f}%",
            t("spend_effect.effect_share"): "{:.1f}%",
            t("spend_effect.efficiency_index"): "{:.2f}",
            t("spend_effect.total_spend"): lambda x: fmt_currency(x),
            t("spend_effect.contribution"): lambda x: fmt_currency(x),
            t("spend_effect.roi"): "{:.1f}x",
        }),
        use_container_width=True,
        hide_index=True,
    )


def main():
    render_header("spend_effect.title", "spend_effect.description")
    render_language_selector()
    st.sidebar.markdown("---")

    with st.spinner(t("common.loading")):
        results, channel_params = load_model()

    if not results or not channel_params:
        st.error(t("app.model_not_found"))
        return

    df = build_channel_df(channel_params)

    # Chart
    render_scatter(df)

    # Legend
    st.markdown(f"""
- <span style="color:#10b981">&#9650;</span> **{t('spend_effect.above_diagonal')}**
- <span style="color:#ef4444">&#9660;</span> **{t('spend_effect.below_diagonal')}**
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Summary table
    st.subheader(t("spend_effect.summary_table"))
    render_summary_table(df)


if __name__ == "__main__":
    main()
