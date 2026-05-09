import os
import sys
"""
Quant V2 - MEXC Live Paper Trading Dashboard
Run: python quant_v2/dashboard/run.py --mode paper
Then open: http://127.0.0.1:8052
"""
import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
_quant_v2_dir = os.path.dirname(_script_dir)
_math_trading_dir = os.path.dirname(_quant_v2_dir)
sys.path.insert(0, _math_trading_dir)

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import argparse
import threading

from dotenv import load_dotenv
load_dotenv(os.path.join(_math_trading_dir, ".env"))

# ─── theme ────────────────────────────────────────────────────────
BG="#0d1117"; CARD="#161b22"; BORDER="#30363d"
GREEN="#3fb950"; RED="#f85149"; BLUE="#58a6ff"
YELLOW="#d29922"; PURPLE="#bc8cff"; GRAY="#8b949e"; WHITE="#e6edf3"

STAT_BOX={"textAlign":"center","padding":"12px 8px","backgroundColor":CARD,
           "border":f"1px solid {BORDER}","borderRadius":"10px","marginBottom":"8px"}
TR_GRN={"backgroundColor":"#0d2818","borderLeft":f"3px solid {GREEN}","padding":"6px 12px","marginBottom":"2px"}
TR_RED={"backgroundColor":"#2d1115","borderLeft":f"3px solid {RED}","padding":"6px 12px","marginBottom":"2px"}

def sc(lbl, vid, suffix="", color=WHITE, large=False):
    fs="26px" if large else "17px"
    return html.Div([
        html.Div(lbl, style={"fontSize":"10px","color":GRAY,"textTransform":"uppercase","letterSpacing":"1px"}),
        html.Div([html.Span(id=vid, style={"fontSize":fs,"fontWeight":"bold","color":color}),
                  html.Span(suffix, style={"fontSize":"12px","color":GRAY,"marginLeft":"4px"}) if suffix else None]),
    ], style=STAT_BOX)

def pc(pid):
    return html.Div([
        html.Div([
            html.Span(id=f"{pid}-symbol", style={"fontSize":"14px","fontWeight":"bold","color":WHITE}),
            html.Span(id=f"{pid}-price", style={"fontSize":"14px","color":BLUE,"marginLeft":"10px"}),
        ], style={"display":"flex","justifyContent":"space-between","marginBottom":"6px"}),
        html.Div([
            sc("Signal",f"{pid}-signal",color=YELLOW), sc("Z-Score",f"{pid}-zscore",color=WHITE),
            sc("Hurst",f"{pid}-hurst",color=WHITE), sc("ML",f"{pid}-mlconf",color=PURPLE),
        ], style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"4px"}),
    ], style={"backgroundColor":CARD,"border":f"1px solid {BORDER}","borderRadius":"10px","padding":"12px","marginBottom":"8px"})

def build_layout(pairs):
    HEADER={"backgroundColor":"#010409","padding":"14px 24px","borderBottom":f"2px solid {BLUE}",
             "display":"flex","justifyContent":"space-between","alignItems":"center"}
    return html.Div([
        html.Div([
            html.Div([html.Span("MEXC | Quant V2",style={"fontSize":"22px","fontWeight":"bold","color":WHITE,"marginRight":"8px"}),
                      html.Span("LIVE PAPER TRADING",style={"fontSize":"14px","color":GREEN,"fontWeight":"bold"})]),
            html.Div([html.Span(id="conn-status",style={"fontSize":"12px","color":GREEN,"marginRight":"16px"}),
                      html.Span(id="time-display",style={"fontSize":"12px","color":GRAY})]),
        ], style=HEADER),

        html.Div([
            html.Div([
                html.Div("PORTFOLIO",style={"fontSize":"11px","color":GRAY,"textTransform":"uppercase","letterSpacing":"2px","marginBottom":"8px"}),
                html.Div([
                    sc("Equity","equity-display",suffix="$",color=GREEN,large=True),
                    sc("P&L","pnl-display",suffix="$",color=YELLOW),
                    sc("Win Rate","winrate-display",suffix="%",color=BLUE),
                    sc("Trades","trades-display",color=WHITE),
                    sc("Expectancy","expectancy-display",suffix="$",color=PURPLE),
                    sc("Drawdown","drawdown-display",suffix="%",color=RED),
                ], style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"4px"}),
            ], style={"backgroundColor":CARD,"border":f"1px solid {BORDER}","borderRadius":"10px","padding":"14px","marginBottom":"10px"}),

            html.Div("RISK GUARDS",style={"fontSize":"11px","color":GRAY,"textTransform":"uppercase","letterSpacing":"2px","marginBottom":"8px"}),
            html.Div([
                html.Div([html.Div("Daily Loss",style={"fontSize":"10px","color":GRAY}),html.Div(id="daily-loss-display",style={"fontSize":"15px","fontWeight":"bold","color":RED})],style=STAT_BOX),
                html.Div([html.Div("Consec",style={"fontSize":"10px","color":GRAY}),html.Div(id="consec-loss-display",style={"fontSize":"15px","fontWeight":"bold","color":YELLOW})],style=STAT_BOX),
                html.Div([html.Div("Status",style={"fontSize":"10px","color":GRAY}),html.Div(id="breaker-status",style={"fontSize":"14px","fontWeight":"bold","color":GREEN})],style=STAT_BOX),
                html.Div([html.Div("Sentiment",style={"fontSize":"10px","color":GRAY}),html.Div(id="sentiment-display",style={"fontSize":"14px","fontWeight":"bold","color":YELLOW})],style=STAT_BOX),
            ], style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"4px","marginBottom":"10px"}),

            html.Div("PAIRS (MEXC)",style={"fontSize":"11px","color":GRAY,"textTransform":"uppercase","letterSpacing":"2px","marginBottom":"8px"}),
            *[pc(p.replace("/","")) for p in pairs],

            html.Div(id="pos-display", style={"fontSize":"11px","color":BLUE,"marginBottom":"10px"}),
        ], style={"flex":"1","overflowY":"auto","paddingRight":"10px"}),

        html.Div([
            html.Div("EQUITY CURVE",style={"fontSize":"11px","color":GRAY,"textTransform":"uppercase","letterSpacing":"2px","marginBottom":"8px"}),
            dcc.Graph(id="equity-chart",style={"height":"260px"},config={"displayModeBar":False}),

            html.Div("TRADE LOG",style={"fontSize":"11px","color":GRAY,"textTransform":"uppercase","letterSpacing":"2px","marginBottom":"8px","marginTop":"12px"}),
            html.Div([html.Div([
                html.Span("Time",style={"flex":"1","fontSize":"10px","color":GRAY}),
                html.Span("Pair",style={"flex":"1","fontSize":"10px","color":GRAY}),
                html.Span("Dir",style={"flex":"1","fontSize":"10px","color":GRAY}),
                html.Span("Entry",style={"flex":"1","fontSize":"10px","color":GRAY}),
                html.Span("Exit",style={"flex":"1","fontSize":"10px","color":GRAY}),
                html.Span("P&L",style={"flex":"1","fontSize":"10px","color":GRAY}),
                html.Span("Reason",style={"flex":"1","fontSize":"10px","color":GRAY}),
            ], style={"display":"flex","padding":"4px 12px","borderBottom":f"1px solid {BORDER}"})
            ], style={"backgroundColor":CARD,"borderRadius":"8px","overflow":"hidden"}),
            html.Div(id="trade-log",style={"maxHeight":"250px","overflowY":"auto","backgroundColor":CARD,"borderRadius":"0 0 8px 8px","border":f"1px solid {BORDER}","borderTop":"none"}),
        ], style={"flex":"1"}),

        dcc.Interval(id="refresh", interval=2000, n_intervals=0),
    ], style={"display":"flex","gap":"16px","padding":"16px","backgroundColor":BG,"minHeight":"100vh",
               "fontFamily":"'Segoe UI',system-ui,sans-serif","color":WHITE})


def build_chart(ec, cur):
    fig = go.Figure()
    if ec and len(ec) > 1:
        fig.add_trace(go.Scatter(x=list(range(len(ec))),y=ec,mode="lines",name="Equity",
                                  line=dict(color=BLUE,width=2),fill="tozeroy",fillcolor="rgba(88,166,255,0.08)"))
        fig.add_hline(y=ec[0],line_dash="dot",line_color=GRAY,line_width=1)
    else:
        fig.add_trace(go.Scatter(x=[0],y=[cur],mode="markers",marker=dict(color=BLUE,size=8)))
    fig.update_layout(plot_bgcolor=BG,paper_bgcolor=BG,margin=dict(l=40,r=10,t=10,b=30),
                      xaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
                      yaxis=dict(showgrid=True,gridcolor=BORDER,tickfont=dict(color=GRAY,size=10),zeroline=False),
                      font=dict(color=WHITE),height=260)
    return fig


def register_cb(app, engine):
    pairs = engine.pairs
    pids = [p.replace("/","") for p in pairs]

    s_out = [Output("conn-status","children"),Output("conn-status","style"),Output("time-display","children"),
             Output("equity-display","children"),Output("pnl-display","children"),Output("pnl-display","style"),
             Output("winrate-display","children"),Output("trades-display","children"),
             Output("expectancy-display","children"),Output("drawdown-display","children"),
             Output("daily-loss-display","children"),Output("consec-loss-display","children"),
             Output("breaker-status","children"),Output("breaker-status","style"),
             Output("sentiment-display","children"),Output("sentiment-display","style"),
             Output("pos-display","children")]

    p_out = []
    for pid in pids:
        p_out += [Output(f"{pid}-symbol","children"),Output(f"{pid}-price","children"),
                   Output(f"{pid}-signal","children"),Output(f"{pid}-signal","style"),
                   Output(f"{pid}-zscore","children"),Output(f"{pid}-hurst","children"),
                   Output(f"{pid}-mlconf","children")]

    @app.callback(s_out+p_out+[Output("equity-chart","figure"),Output("trade-log","children")],[Input("refresh","n_intervals")])
    def update(_n):
        d = engine.get_state()
        conn = "LIVE" if d["connected"] else "OFFLINE"
        cs = {"fontSize":"12px","color":GREEN if d["connected"] else RED,"fontWeight":"bold"}
        pnl = d["daily_pnl"]
        ps = {"fontSize":"17px","fontWeight":"bold","color":GREEN if pnl>=0 else RED}
        br = "HALTED" if d["is_halted"] else d["breaker_status"]
        bs = {"fontSize":"14px","fontWeight":"bold","color":RED if d["is_halted"] else GREEN}
        sn = d["sentiment"]
        ss = {"fontSize":"14px","fontWeight":"bold","color":GREEN if sn>0.15 else RED if sn<-0.15 else YELLOW}

        pos_text = f"Open Positions: {len(engine._open_positions)}" if hasattr(engine, '_open_positions') else ""

        stats = [conn,cs,datetime.now().strftime("%H:%M:%S"),
                 f"${d['equity']:.2f}",f"${pnl:+.2f}",ps,
                 f"{d['win_rate']:.1f}",f"{d['total_trades']}",
                 f"${d['expectancy']:.4f}",f"{d['drawdown_pct']:.1f}",
                 f"${d['daily_loss_pct']:.2f}",f"{d['consecutive_losses']}",
                 br,bs,f"{sn:+.2f}",ss,pos_text]

        pdl = []
        for p in pairs:
            pid = p.replace("/","")
            pdata = d["pair_data"].get(p,{})
            sig = pdata.get("signal",0)
            sig_txt = "LONG" if sig==1 else "SHORT" if sig==-1 else "WAIT"
            sig_col = GREEN if sig==1 else RED if sig==-1 else GRAY
            sig_s = {"fontSize":"17px","fontWeight":"bold","color":sig_col}
            ml = pdata.get("ml_conf",0)
            pdl += [p.split("/")[0]+"/USDT",f"${pdata.get('price',0):.4f}",
                     sig_txt,sig_s,f"{pdata.get('zscore',0):.2f}",
                     f"{pdata.get('hurst',0.5):.2f}",f"{ml:.2f}" if ml>0 else "--"]

        fig = build_chart(d.get("equity_curve",[d["equity"]]), d["equity"])

        log = d.get("trade_log",[])[-30:]
        rows = []
        for t in reversed(log):
            pv = t.get("pnl",0)
            rs = TR_GRN if pv>0 else TR_RED if pv<0 else {"backgroundColor":CARD,"borderLeft":f"3px solid {GRAY}","padding":"6px 12px"}
            ts = t.get("exit_time",t.get("entry_time",""))
            if hasattr(ts,"strftime"): ts=ts.strftime("%H:%M:%S")
            elif isinstance(ts,str) and "T" in ts: ts=ts[11:19]
            rows.append(html.Div([
                html.Span(str(ts),style={"flex":"1","fontSize":"11px","color":WHITE}),
                html.Span(t.get("symbol",""),style={"flex":"1","fontSize":"11px","color":BLUE}),
                html.Span(t.get("direction","--"),style={"flex":"1","fontSize":"11px","color":GREEN if t.get("direction")=="long" else RED}),
                html.Span(f"{t.get('entry_price',0):.4f}",style={"flex":"1","fontSize":"11px","color":WHITE}),
                html.Span(f"{t.get('exit_price',0):.4f}",style={"flex":"1","fontSize":"11px","color":WHITE}),
                html.Span(f"${pv:+.4f}",style={"flex":"1","fontSize":"11px","fontWeight":"bold","color":GREEN if pv>0 else RED}),
                html.Span(t.get("reason",""),style={"flex":"1","fontSize":"10px","color":GRAY}),
            ],style={**rs,"display":"flex"}))

        return stats+pdl+[fig,rows]


def main():
    parser = argparse.ArgumentParser(description="Quant V2 Dashboard")
    parser.add_argument("--mode",choices=["live","paper"],default="paper")
    parser.add_argument("--port",type=int,default=8052)
    parser.add_argument("--interval",type=int,default=60,help="Seconds between MEXC data polls")
    args = parser.parse_args()

    from config.crypto_config import CryptoConfig
    config = CryptoConfig.with_default_pairs()
    pairs = list(config.scalping.pairs)
    api_key = os.getenv("MEXC_API_KEY","")
    api_secret = os.getenv("MEXC_API_SECRET","")

    print(f"\n{'='*60}")
    print(f"  QUANT V2 LIVE PAPER TRADING - {args.mode.upper()} MODE")
    print(f"  Pairs: {pairs}")
    print(f"  Capital: ${config.scalping.initial_capital:.0f}")
    print(f"  Refresh: every {args.interval}s")
    print(f"{'='*60}\n")

    from live.mexc_hybrid import MEXCHybridConnector
    from live.live_paper import LivePaperEngine

    mexc = MEXCHybridConnector(api_key=api_key, api_secret=api_secret, proxy=os.getenv("MEXC_PROXY",""))
    mexc.connect()
    print(f"  MEXC: {'Direct' if mexc._mexc_connected else 'CoinGecko'} | Connected: {mexc.connected}")

    engine = LivePaperEngine(config, mexc)
    engine.load_initial_data()

    live_thread = threading.Thread(target=engine.run_loop, args=(args.interval,), daemon=True)
    live_thread.start()
    print(f"  Trading engine started (interval={args.interval}s)\n")

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                     title="MEXC Quant V2 - Live Paper Trading",
                     update_title=None, suppress_callback_exceptions=True)
    app.layout = build_layout(pairs)
    register_cb(app, engine)

    print(f"  Dashboard at http://127.0.0.1:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        app.run(debug=False, host="127.0.0.1", port=args.port)
    except KeyboardInterrupt:
        print("\nStopping...")
        engine.stop()
    finally:
        mexc.disconnect()


if __name__ == "__main__":
    main()