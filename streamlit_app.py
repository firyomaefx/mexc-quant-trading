import os
import sys
"""
MEXC Quant Trading Bot — Streamlit Dashboard
Deploy on: share.streamlit.io → firyomaefx/mexc-quant-trading → streamlit_app.py
"""
import sys
import os
import threading
import time
from datetime import datetime

_script = os.path.dirname(os.path.abspath(__file__))
if _script not in sys.path:
    sys.path.insert(0, _script)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="MEXC Quant Bot", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

from dotenv import load_dotenv
load_dotenv(os.path.join(_script, ".env") if os.path.exists(os.path.join(_script, ".env")) else "")

BG = "#0d1117"; CARD = "#161b22"; BORDER = "#30363d"
GREEN = "#3fb950"; RED = "#f85149"; BLUE = "#58a6ff"
YELLOW = "#d29922"; PURPLE = "#bc8cff"; GRAY = "#8b949e"


@st.cache_resource
def get_engine():
    from config.crypto_config import CryptoConfig
    from live.mexc_hybrid import MEXCHybridConnector
    from live.live_paper import LivePaperEngine

    cfg = CryptoConfig.with_default_pairs()
    api_key = st.secrets.get("MEXC_API_KEY", os.getenv("MEXC_API_KEY", ""))
    api_secret = st.secrets.get("MEXC_API_SECRET", os.getenv("MEXC_API_SECRET", ""))
    proxy = st.secrets.get("MEXC_PROXY", os.getenv("MEXC_PROXY", ""))

    mexc = MEXCHybridConnector(api_key=api_key, api_secret=api_secret, proxy=proxy)
    mexc.connect()

    engine = LivePaperEngine(cfg, mexc)
    engine.load_initial_data()

    t = threading.Thread(target=engine.run_loop, args=(30,), daemon=True)
    t.start()

    return engine


def metric_card(label, value, suffix="", color="#e6edf3"):
    st.markdown(f"""
    <div style="text-align:center;padding:8px 4px;background:{CARD};border:1px solid {BORDER};border-radius:8px;margin-bottom:6px">
        <div style="font-size:9px;color:{GRAY};text-transform:uppercase;letter-spacing:1px">{label}</div>
        <div style="font-size:18px;font-weight:bold;color:{color}">{value}<span style="font-size:11px;color:{GRAY};margin-left:3px">{suffix}</span></div>
    </div>
    """, unsafe_allow_html=True)


def pair_card(symbol, pdata):
    price = pdata.get("price", 0)
    sig = pdata.get("signal", 0)
    zs = pdata.get("zscore", 0)
    hu = pdata.get("hurst", 0.5)

    sig_txt = "LONG" if sig == 1 else "SHORT" if sig == -1 else "WAIT"
    sig_col = GREEN if sig == 1 else RED if sig == -1 else GRAY

    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:10px;margin-bottom:6px">
        <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="font-weight:bold;color:#e6edf3">{symbol}</span>
            <span style="color:{BLUE}">${price:.4f}</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:4px;text-align:center">
            <div><div style="font-size:8px;color:{GRAY}">SIGNAL</div><div style="font-size:12px;font-weight:bold;color:{sig_col}">{sig_txt}</div></div>
            <div><div style="font-size:8px;color:{GRAY}">Z</div><div style="font-size:12px;color:#e6edf3">{zs:.2f}</div></div>
            <div><div style="font-size:8px;color:{GRAY}">HURST</div><div style="font-size:12px;color:#e6edf3">{hu:.2f}</div></div>
            <div><div style="font-size:8px;color:{GRAY}">POS</div><div style="font-size:12px;color:{BLUE}">{price:.0f}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.markdown(f"""
    <div style="background:#010409;padding:10px 20px;border-bottom:2px solid {BLUE};margin-bottom:12px;display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:20px;font-weight:bold;color:#e6edf3">MEXC Quant Bot</span>
        <span style="font-size:12px;color:{GREEN};font-weight:bold">LIVE PAPER TRADING</span>
    </div>
    """, unsafe_allow_html=True)

    status = st.empty()
    engine = get_engine()

    col1, col2 = st.columns([1.6, 1])

    while True:
        d = engine.get_state()
        with status.container():
            con = d.get("connected", False)
            con_txt = "● LIVE" if con else "● OFFLINE"
            con_col = GREEN if con else RED
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;margin-bottom:10px;padding:0 4px">
                <span style="font-size:11px;color:{con_col};font-weight:bold">{con_txt}</span>
                <span style="font-size:11px;color:{GRAY}">{datetime.now().strftime("%H:%M:%S")}</span>
            </div>
            """, unsafe_allow_html=True)

        with col1:
            st.markdown(f"<div style='font-size:10px;color:{GRAY};text-transform:uppercase;letter-spacing:2px;margin-bottom:4px'>PAIRS (MEXC)</div>", unsafe_allow_html=True)
            for sym in engine.pairs:
                pdata = d.get("pair_data", {}).get(sym, {})
                pair_card(sym, pdata)

            trades = d.get("trade_log", [])[-20:]
            if trades:
                st.markdown(f"<div style='font-size:10px;color:{GRAY};text-transform:uppercase;letter-spacing:2px;margin:10px 0 4px'>TRADE LOG</div>", unsafe_allow_html=True)
                rows = []
                for t in reversed(trades):
                    pv = t.get("pnl", 0)
                    dir_col = GREEN if t.get("direction") == "long" else RED
                    pnl_col = GREEN if pv > 0 else RED
                    ts = str(t.get("exit_time", ""))
                    if len(ts) > 16:
                        ts = ts[11:19]
                    rows.append({
                        "Time": ts, "Pair": t.get("symbol", ""),
                        "Type": t.get("direction", ""), "P&L": f"${pv:+.4f}",
                        "Reason": t.get("reason", "")[:20],
                    })
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True, height=250)

        with col2:
            pnl = d.get("daily_pnl", 0)
            pnl_col = GREEN if pnl >= 0 else RED
            metric_card("Equity", f"${d.get('equity', 0):.2f}", color=GREEN)
            metric_card("P&L Today", f"${pnl:+.2f}", color=pnl_col)
            metric_card("Win Rate", f"{d.get('win_rate', 0):.1f}", suffix="%", color=BLUE)
            metric_card("Trades", f"{d.get('total_trades', 0)}")
            metric_card("Expectancy", f"${d.get('expectancy', 0):.4f}", color=PURPLE)
            metric_card("Drawdown", f"{d.get('drawdown_pct', 0):.1f}", suffix="%", color=RED)

            ec = d.get("equity_curve", [])
            if ec and len(ec) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(ec))), y=ec, mode="lines", line=dict(color=BLUE, width=2), fill="tozeroy", fillcolor="rgba(88,166,255,0.08)"))
                fig.add_hline(y=ec[0], line_dash="dot", line_color=GRAY, line_width=1)
                fig.update_layout(plot_bgcolor=BG, paper_bgcolor=BG, margin=dict(l=30, r=10, t=10, b=25), height=220, xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(showgrid=True, gridcolor=BORDER, tickfont=dict(color=GRAY, size=9)))
                st.plotly_chart(fig, use_container_width=True)

            cb_bre = "HALTED" if d.get("is_halted") else d.get("breaker_status", "OK")
            cb_col = RED if d.get("is_halted") else GREEN
            metric_card("Breaker", cb_bre, color=cb_col)
            metric_card("Consec Loss", f"{d.get('consecutive_losses', 0)}", color=YELLOW)
            metric_card("Sentiment", f"{d.get('sentiment', 0):+.2f}", color=GREEN if d.get("sentiment", 0) > 0.15 else RED if d.get("sentiment", 0) < -0.15 else YELLOW)

        time.sleep(3)
        st.rerun()


if __name__ == "__main__":
    main()