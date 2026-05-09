import os
import sys
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Quant V2 - AI Scalping Dashboard",
    update_title=None,
    suppress_callback_exceptions=True,
)

BG = "#0d1117"
CARD = "#161b22"
BORDER = "#30363d"
GREEN = "#3fb950"
RED = "#f85149"
BLUE = "#58a6ff"
YELLOW = "#d29922"
PURPLE = "#bc8cff"
GRAY = "#8b949e"
WHITE = "#e6edf3"

STAT_BOX = {
    "textAlign": "center", "padding": "12px 8px",
    "backgroundColor": CARD, "border": f"1px solid {BORDER}",
    "borderRadius": "10px", "marginBottom": "8px",
}

HEADER = {
    "backgroundColor": "#010409", "padding": "14px 24px",
    "borderBottom": f"2px solid {BLUE}", "display": "flex",
    "justifyContent": "space-between", "alignItems": "center",
}

TRADE_ROW_GREEN = {"backgroundColor": "#0d2818", "borderLeft": f"3px solid {GREEN}", "padding": "6px 12px", "marginBottom": "2px"}
TRADE_ROW_RED = {"backgroundColor": "#2d1115", "borderLeft": f"3px solid {RED}", "padding": "6px 12px", "marginBottom": "2px"}
TRADE_ROW_NEUTRAL = {"backgroundColor": CARD, "borderLeft": f"3px solid {GRAY}", "padding": "6px 12px", "marginBottom": "2px"}


def stat_card(label, value_id, suffix="", color=WHITE, large=False):
    fs = "26px" if large else "17px"
    return html.Div([
        html.Div(label, style={"fontSize": "10px", "color": GRAY, "textTransform": "uppercase", "letterSpacing": "1px"}),
        html.Div([
            html.Span(id=value_id, style={"fontSize": fs, "fontWeight": "bold", "color": color}),
            html.Span(suffix, style={"fontSize": "12px", "color": GRAY, "marginLeft": "4px"}) if suffix else html.Span(""),
        ]),
    ], style=STAT_BOX)


def pair_card(pair_id):
    return html.Div([
        html.Div([
            html.Span(id=f"{pair_id}-symbol", style={"fontSize": "14px", "fontWeight": "bold", "color": WHITE}),
            html.Span(id=f"{pair_id}-price", style={"fontSize": "14px", "color": BLUE, "marginLeft": "10px"}),
        ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"}),
        html.Div([
            stat_card("Signal", f"{pair_id}-signal", color=YELLOW),
            stat_card("Z-Score", f"{pair_id}-zscore", color=WHITE),
            stat_card("Hurst", f"{pair_id}-hurst", color=WHITE),
            stat_card("ML Conf", f"{pair_id}-mlconf", color=PURPLE),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "4px"}),
    ], style={"backgroundColor": CARD, "border": f"1px solid {BORDER}", "borderRadius": "10px", "padding": "12px", "marginBottom": "8px"})


def build_layout(pairs=None, refresh_ms=3000):
    if pairs is None:
        pairs = ["XRP/USDT", "ADA/USDT", "SOL/USDT"]

    return html.Div([
        html.Div([
            html.Div([
                html.Span("Quant V2", style={"fontSize": "22px", "fontWeight": "bold", "color": WHITE, "marginRight": "8px"}),
                html.Span("AI Scalping Bot", style={"fontSize": "14px", "color": GRAY}),
            ]),
            html.Div([
                html.Span(id="conn-status", style={"fontSize": "12px", "color": GREEN, "marginRight": "16px"}),
                html.Span(id="time-display", style={"fontSize": "12px", "color": GRAY}),
            ]),
        ], style=HEADER),

        html.Div([
            html.Div([
                html.Div("PORTFOLIO", style={"fontSize": "11px", "color": GRAY, "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "8px"}),
                html.Div([
                    stat_card("Equity", "equity-display", suffix="$", color=GREEN, large=True),
                    stat_card("P&L Today", "pnl-display", suffix="$", color=YELLOW),
                    stat_card("Win Rate", "winrate-display", suffix="%", color=BLUE),
                    stat_card("Trades", "trades-display", color=WHITE),
                    stat_card("Expectancy", "expectancy-display", suffix="$", color=PURPLE),
                    stat_card("Drawdown", "drawdown-display", suffix="%", color=RED),
                ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "4px"}),
            ], style={"backgroundColor": CARD, "border": f"1px solid {BORDER}", "borderRadius": "10px", "padding": "14px", "marginBottom": "10px"}),

            html.Div("RISK GUARDS", style={"fontSize": "11px", "color": GRAY, "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "8px"}),
            html.Div([
                html.Div([
                    html.Div("Daily Loss", style={"fontSize": "10px", "color": GRAY}),
                    html.Div(id="daily-loss-display", style={"fontSize": "15px", "fontWeight": "bold", "color": RED}),
                ], style=STAT_BOX),
                html.Div([
                    html.Div("Consec Losses", style={"fontSize": "10px", "color": GRAY}),
                    html.Div(id="consec-loss-display", style={"fontSize": "15px", "fontWeight": "bold", "color": YELLOW}),
                ], style=STAT_BOX),
                html.Div([
                    html.Div("Status", style={"fontSize": "10px", "color": GRAY}),
                    html.Div(id="breaker-status", style={"fontSize": "14px", "fontWeight": "bold", "color": GREEN}),
                ], style=STAT_BOX),
                html.Div([
                    html.Div("Sentiment", style={"fontSize": "10px", "color": GRAY}),
                    html.Div(id="sentiment-display", style={"fontSize": "14px", "fontWeight": "bold", "color": YELLOW}),
                ], style=STAT_BOX),
            ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "4px", "marginBottom": "10px"}),

            html.Div("PAIRS", style={"fontSize": "11px", "color": GRAY, "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "8px"}),
            *[pair_card(p.replace("/", "")) for p in pairs],
        ], style={"flex": "1", "overflowY": "auto", "paddingRight": "10px"}),

        html.Div([
            html.Div("EQUITY CURVE", style={"fontSize": "11px", "color": GRAY, "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "8px"}),
            dcc.Graph(id="equity-chart", style={"height": "250px"}, config={"displayModeBar": False}),

            html.Div("TRADE LOG", style={"fontSize": "11px", "color": GRAY, "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "8px", "marginTop": "12px"}),
            html.Div([
                html.Div([
                    html.Span("Time", style={"flex": "1", "fontSize": "10px", "color": GRAY}),
                    html.Span("Pair", style={"flex": "1", "fontSize": "10px", "color": GRAY}),
                    html.Span("Dir", style={"flex": "1", "fontSize": "10px", "color": GRAY}),
                    html.Span("Entry", style={"flex": "1", "fontSize": "10px", "color": GRAY}),
                    html.Span("Exit", style={"flex": "1", "fontSize": "10px", "color": GRAY}),
                    html.Span("P&L", style={"flex": "1", "fontSize": "10px", "color": GRAY}),
                    html.Span("Reason", style={"flex": "1", "fontSize": "10px", "color": GRAY}),
                ], style={"display": "flex", "padding": "4px 12px", "borderBottom": f"1px solid {BORDER}"}),
            ], style={"backgroundColor": CARD, "borderRadius": "8px", "overflow": "hidden"}),
            html.Div(id="trade-log", style={"maxHeight": "300px", "overflowY": "auto", "backgroundColor": CARD, "borderRadius": "0 0 8px 8px", "border": f"1px solid {BORDER}", "borderTop": "none"}),
        ], style={"flex": "1"}),

    ], style={"display": "flex", "gap": "16px", "padding": "16px", "backgroundColor": BG, "minHeight": "100vh", "fontFamily": "'Segoe UI', system-ui, sans-serif", "color": WHITE},
    id="main-container")


def build_full_layout(pairs=None, refresh_ms=3000):
    return html.Div([
        build_layout(pairs, refresh_ms),
        dcc.Interval(id="interval-refresh", interval=refresh_ms, n_intervals=0),
    ])