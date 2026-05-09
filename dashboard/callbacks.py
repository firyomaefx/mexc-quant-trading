import sys
_script = os.path.dirname(os.path.abspath(__file__))

import plotly.graph_objects as go
from dash import Input, Output, html
import numpy as np
from datetime import datetime

from dashboard.layout import GREEN, RED, BLUE, YELLOW, PURPLE, GRAY, WHITE, BG, CARD, BORDER, TRADE_ROW_GREEN, TRADE_ROW_RED, TRADE_ROW_NEUTRAL


def register_callbacks(app, provider):
    pairs = getattr(provider, "pairs", ["XRP/USDT", "ADA/USDT", "SOL/USDT"])

    pair_ids = [p.replace("/", "") for p in pairs]

    stat_outputs = [
        Output("conn-status", "children"),
        Output("conn-status", "style"),
        Output("time-display", "children"),
        Output("equity-display", "children"),
        Output("pnl-display", "children"),
        Output("pnl-display", "style"),
        Output("winrate-display", "children"),
        Output("trades-display", "children"),
        Output("expectancy-display", "children"),
        Output("drawdown-display", "children"),
        Output("daily-loss-display", "children"),
        Output("consec-loss-display", "children"),
        Output("breaker-status", "children"),
        Output("breaker-status", "style"),
        Output("sentiment-display", "children"),
        Output("sentiment-display", "style"),
    ]

    pair_outputs = []
    for pid in pair_ids:
        pair_outputs.extend([
            Output(f"{pid}-symbol", "children"),
            Output(f"{pid}-price", "children"),
            Output(f"{pid}-signal", "children"),
            Output(f"{pid}-signal", "style"),
            Output(f"{pid}-zscore", "children"),
            Output(f"{pid}-hurst", "children"),
            Output(f"{pid}-mlconf", "children"),
        ])

    @app.callback(
        stat_outputs + pair_outputs + [Output("equity-chart", "figure"), Output("trade-log", "children")],
        [Input("interval-refresh", "n_intervals")],
    )
    def update_all(n):
        data = provider.refresh()

        connected = data.get("connected", False)
        conn_text = "LIVE" if connected else "OFFLINE"
        conn_style = {"fontSize": "12px", "color": GREEN if connected else RED, "fontWeight": "bold"}
        time_text = datetime.now().strftime("%H:%M:%S")

        equity = data.get("equity", 0)
        pnl = data.get("daily_pnl", 0)
        win_rate = data.get("win_rate", 0)
        trades = data.get("total_trades", 0)
        expectancy = data.get("expectancy", 0)
        drawdown = data.get("drawdown_pct", 0)
        daily_loss = data.get("daily_loss_pct", 0)
        consec_losses = data.get("consecutive_losses", 0)
        breaker_status = data.get("breaker_status", "OK")
        breaker_halted = data.get("is_halted", False)
        sentiment = data.get("sentiment", 0)

        equity_text = f"${equity:.2f}"
        pnl_text = f"${pnl:+.2f}"
        pnl_style = {"fontSize": "17px", "fontWeight": "bold", "color": GREEN if pnl >= 0 else RED}
        wr_text = f"{win_rate:.1f}"
        trades_text = f"{trades}"
        exp_text = f"${expectancy:.4f}"
        dd_text = f"{drawdown:.1f}"
        dl_text = f"${daily_loss:.2f}"
        cl_text = f"{consec_losses}"
        bs_text = "HALTED" if breaker_halted else breaker_status
        bs_style = {"fontSize": "14px", "fontWeight": "bold", "color": RED if breaker_halted else GREEN}
        sent_text = f"{sentiment:+.2f}"
        sent_label = "bullish" if sentiment > 0.15 else "bearish" if sentiment < -0.15 else "neutral"
        sent_style = {"fontSize": "14px", "fontWeight": "bold", "color": GREEN if sentiment > 0.15 else RED if sentiment < -0.15 else YELLOW}

        pair_data = []
        for p in pairs:
            pid = p.replace("/", "")
            pdata = data.get("pair_data", {}).get(p, {})
            sym = p.split("/")[0]
            price = pdata.get("price", 0)
            sig = pdata.get("signal", 0)
            zs = pdata.get("zscore", 0)
            hu = pdata.get("hurst", 0.5)
            ml = pdata.get("ml_conf", 0)

            sig_text = "LONG" if sig == 1 else "SHORT" if sig == -1 else "WAIT"
            sig_color = GREEN if sig == 1 else RED if sig == -1 else GRAY
            sig_style = {"fontSize": "17px", "fontWeight": "bold", "color": sig_color}
            zs_text = f"{zs:.2f}"
            hu_text = f"{hu:.2f}"
            ml_text = f"{ml:.2f}" if ml > 0 else "—"

            pair_data.extend([sym + "/USDT", f"${price:.4f}", sig_text, sig_style, zs_text, hu_text, ml_text])

        equity_curve = data.get("equity_curve", [])
        fig = _build_equity_chart(equity_curve, equity)

        trade_log = data.get("trade_log", [])[-30:]
        trade_rows = []
        for t in reversed(trade_log):
            pnl_val = t.get("pnl", 0)
            row_style = TRADE_ROW_GREEN if pnl_val > 0 else TRADE_ROW_RED if pnl_val < 0 else TRADE_ROW_NEUTRAL
            ts = t.get("exit_time", t.get("timestamp", ""))
            if hasattr(ts, "strftime"):
                ts = ts.strftime("%H:%M:%S")
            elif isinstance(ts, str) and len(ts) > 8:
                ts = ts[11:19] if len(ts) > 19 else ts
            trade_rows.append(html.Div([
                html.Span(str(ts), style={"flex": "1", "fontSize": "11px", "color": WHITE}),
                html.Span(t.get("symbol", ""), style={"flex": "1", "fontSize": "11px", "color": BLUE}),
                html.Span(t.get("direction", "—"), style={"flex": "1", "fontSize": "11px", "color": GREEN if t.get("direction") == "long" else RED}),
                html.Span(f"{t.get('entry_price', 0):.4f}", style={"flex": "1", "fontSize": "11px", "color": WHITE}),
                html.Span(f"{t.get('exit_price', 0):.4f}", style={"flex": "1", "fontSize": "11px", "color": WHITE}),
                html.Span(f"${pnl_val:+.4f}", style={"flex": "1", "fontSize": "11px", "fontWeight": "bold", "color": GREEN if pnl_val > 0 else RED}),
                html.Span(t.get("reason", ""), style={"flex": "1", "fontSize": "10px", "color": GRAY}),
            ], style={**row_style, "display": "flex"}))

        results = [
            conn_text, conn_style, time_text,
            equity_text, pnl_text, pnl_style, wr_text, trades_text,
            exp_text, dd_text, dl_text, cl_text,
            bs_text, bs_style, sent_text, sent_style,
        ]
        results.extend(pair_data)
        results.extend([fig, trade_rows])
        return results


def _build_equity_chart(equity_curve, current_equity):
    fig = go.Figure()

    if equity_curve and len(equity_curve) > 1:
        x_vals = list(range(len(equity_curve)))
        colors = []
        for i in range(1, len(equity_curve)):
            colors.append(GREEN if equity_curve[i] >= equity_curve[i-1] else RED)

        fig.add_trace(go.Scatter(
            x=x_vals, y=equity_curve,
            mode="lines", name="Equity",
            line=dict(color=BLUE, width=2),
            fill="tozeroy",
            fillcolor="rgba(88, 166, 255, 0.08)",
        ))

        fig.add_hline(y=equity_curve[0] if equity_curve else 0, line_dash="dot", line_color=GRAY, line_width=1)
    else:
        fig.add_trace(go.Scatter(
            x=[0], y=[current_equity],
            mode="markers", marker=dict(color=BLUE, size=8),
        ))

    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        margin=dict(l=40, r=10, t=10, b=30),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(color=GRAY, size=10), zeroline=False),
        font=dict(color=WHITE),
        height=250,
    )
    return fig
