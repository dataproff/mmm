"""
Internationalization (i18n) module for MMM Budget Optimizer

Provides translation support for English and Russian languages.
"""
import json
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional


# Available languages
LANGUAGES = {
    'en': 'English',
    'ru': 'Русский'
}

DEFAULT_LANGUAGE = 'en'


# Currency symbols per language
CURRENCY_SYMBOLS = {
    'en': '$',
    'ru': '₽'
}


def get_currency() -> str:
    """Get currency symbol for current language"""
    lang = get_language()
    return CURRENCY_SYMBOLS.get(lang, '$')


def fmt_currency(value: float, decimals: int = 0) -> str:
    """Format a value with the current language's currency symbol"""
    symbol = get_currency()
    if decimals == 0:
        return f"{symbol}{value:,.0f}"
    else:
        return f"{symbol}{value:,.{decimals}f}"


def load_translations(lang: str) -> Dict[str, Any]:
    """Load translations for a given language"""
    locales_dir = Path(__file__).parent.parent / "locales"
    lang_file = locales_dir / f"{lang}.json"

    if not lang_file.exists():
        # Fallback to English
        lang_file = locales_dir / "en.json"

    try:
        with open(lang_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def get_language() -> str:
    """Get current language from session state"""
    if 'language' not in st.session_state:
        st.session_state['language'] = DEFAULT_LANGUAGE
    return st.session_state['language']


def set_language(lang: str) -> None:
    """Set current language in session state"""
    if lang in LANGUAGES:
        st.session_state['language'] = lang


def get_translations() -> Dict[str, Any]:
    """Get translations for current language"""
    lang = get_language()
    cache_key = f'translations_{lang}'

    if cache_key not in st.session_state:
        st.session_state[cache_key] = load_translations(lang)

    return st.session_state[cache_key]


def t(key: str, **kwargs) -> str:
    """
    Get translated string by dot-notation key

    Args:
        key: Dot-notation key like "sidebar.planning_period"
        **kwargs: Format arguments for string interpolation

    Returns:
        Translated string or key if not found

    Example:
        t("sidebar.planning_period")  # Returns "Planning Period" or "Период планирования"
        t("optimizer.target_revenue_for", month="January")  # With interpolation
    """
    translations = get_translations()

    # Navigate nested dict by dot notation
    parts = key.split('.')
    value = translations

    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            # Key not found, return the key itself
            return key

    if not isinstance(value, str):
        return key

    # Apply format arguments if provided
    if kwargs:
        try:
            return value.format(**kwargs)
        except (KeyError, ValueError):
            return value

    return value


def render_language_selector(location: str = "sidebar") -> None:
    """
    Render language selector widget

    Args:
        location: Where to render - "sidebar" or "main"
    """
    current_lang = get_language()

    # Get index of current language
    lang_codes = list(LANGUAGES.keys())
    current_index = lang_codes.index(current_lang) if current_lang in lang_codes else 0

    # Create selector
    container = st.sidebar if location == "sidebar" else st

    selected_lang = container.selectbox(
        "🌐 " + t("sidebar.language"),
        options=lang_codes,
        format_func=lambda x: LANGUAGES[x],
        index=current_index,
        key="language_selector"
    )

    # Update language if changed
    if selected_lang != current_lang:
        set_language(selected_lang)
        # Clear translation cache
        for key in list(st.session_state.keys()):
            if key.startswith('translations_'):
                del st.session_state[key]
        st.rerun()


# Convenience class for namespaced translations
class Translator:
    """Helper class for getting translations with a namespace prefix"""

    def __init__(self, namespace: str):
        self.namespace = namespace

    def __call__(self, key: str, **kwargs) -> str:
        full_key = f"{self.namespace}.{key}" if self.namespace else key
        return t(full_key, **kwargs)
