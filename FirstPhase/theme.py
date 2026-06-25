"""Light / dark palettes and a Qt stylesheet builder for the trial logger."""

THEMES = {
    "light": {
        "window": "#eef2f7",
        "panel":  "#ffffff",
        "text":   "#1e293b",
        "border": "#bcccdc",
        "accent": "#2563eb",
        "plot_bg": "w",
        "axis":   "#334155",
        "grid":   0.20,
        "btn_text": "🌙  Dark",
    },
    "dark": {
        "window": "#0f172a",
        "panel":  "#1e293b",
        "text":   "#e2e8f0",
        "border": "#334155",
        "accent": "#3b82f6",
        "plot_bg": "#111827",
        "axis":   "#94a3b8",
        "grid":   0.30,
        "btn_text": "☀  Light",
    },
}

# EMG channel + microphone curve colors (bright; readable on both themes)
CH_COLORS = ("#FF6B6B", "#22B14C", "#3498DB")
MIC_COLOR = "#9B59B6"


def build_stylesheet(pal):
    """Qt stylesheet for the main window for a given palette dict."""
    return f"""
    QWidget {{ background-color: {pal['window']}; color: {pal['text']}; }}
    QLabel {{ color: {pal['text']}; background: transparent; }}
    QLineEdit {{
        background-color: {pal['panel']}; color: {pal['text']};
        border: 1px solid {pal['border']}; border-radius: 4px; padding: 3px;
    }}
    QPushButton {{
        background-color: {pal['panel']}; color: {pal['text']};
        border: 1px solid {pal['border']}; border-radius: 6px; padding: 4px 10px;
    }}
    QPushButton:hover {{ border-color: {pal['accent']}; }}
    QPushButton:checked {{ background-color: {pal['accent']}; color: white; }}
    QPushButton:disabled {{ color: gray; border-color: {pal['border']}; }}
    """
