import sys, os, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

from config.crypto_config import CryptoConfig
from live.mexc_hybrid import MEXCHybridConnector

cfg = CryptoConfig.with_default_pairs()
api_key = os.getenv("MEXC_API_KEY", "")
api_secret = os.getenv("MEXC_API_SECRET", "")

mexc = MEXCHybridConnector(api_key=api_key, api_secret=api_secret)
mexc.connect()

capital = 18.0
balance = capital
peak = capital
max_dd = 0
all_trades = []

configs = [
    {"sym": "SOL/USDT", "mult": 3, "trail": 0.008, "time_stop": 30, "zs_entry": 0.5, "cooldown": 40},
    {"sym": "XRP/USDT", "mult": 1, "trail": 0.005, "time_stop": 25, "zs_entry": 0.3, "cooldown": 40},
    {"sym": "ADA/USDT", "mult": 1, "trail": 0.005, "time_stop": 25, "zs_entry": 0.3, "cooldown": 40},
]

for cfg_item in configs:
    sym = cfg_item["sym"]
    mult = cfg_item["mult"]
    trail_pct = cfg_item["trail"]
    time_stop = cfg_item["time_stop"]
    zs_entry = cfg_item["zs_entry"]
    cooldown = cfg_item["cooldown"]

    print(f"Fetching {sym}...")
    try:
        df = mexc.fetch_rates(sym, "1m", 500)
    except Exception as e:
        print(f"  Failed: {e}")
        continue
    print(f"  {len(df)} bars loaded")

    close = df["close"].values.astype(np.float64)
    ema20 = pd.Series(close).ewm(span=20).mean().values
    ema50 = pd.Series(close).ewm(span=50).mean().values

    zscore_100 = np.full(len(close), np.nan)
    for i in range(100, len(close)):
        window = close[i-100:i]
        mean = np.mean(window)
        std = np.std(window)
        if std > 1e-10:
            zscore_100[i] = (close[i] - mean) / std

    direction = 0
    entry_price = 0
    entry_bar = 0
    last_entry_bar = -cooldown
    local_trades = []

    for i in range(100, len(close)):
        price = close[i]
        zs = zscore_100[i]
        e20 = ema20[i]
        e50 = ema50[i]
        if np.isnan(zs):
            continue

        if direction == 0:
            if i - last_entry_bar < cooldown:
                continue

            if price > e50 and e20 > e50 and zs > zs_entry:
                direction = 1
                entry_price = price
                entry_bar = i
                last_entry_bar = i
            elif price < e50 and e20 < e50 and zs < -zs_entry:
                direction = -1
                entry_price = price
                entry_bar = i
                last_entry_bar = i
        else:
            bars_held = i - entry_bar
            should_exit = False
            reason = ""

            if direction == 1:
                trail_stop = entry_price * (1 - trail_pct)
                if price <= trail_stop:
                    should_exit, reason = True, "trail_stop"
                elif bars_held >= time_stop:
                    should_exit, reason = True, "time_stop"
                elif zs < -1.5 and bars_held >= 5:
                    should_exit, reason = True, "momentum_reversal"
            else:
                trail_stop = entry_price * (1 + trail_pct)
                if price >= trail_stop:
                    should_exit, reason = True, "trail_stop"
                elif bars_held >= time_stop:
                    should_exit, reason = True, "time_stop"
                elif zs > 1.5 and bars_held >= 5:
                    should_exit, reason = True, "momentum_reversal"

            if should_exit:
                risk = capital * 0.015 * mult
                price_diff = abs(entry_price - price)
                min_gap = entry_price * 0.001
                if price_diff < min_gap:
                    price_diff = min_gap
                size = risk / price_diff
                notional = size * entry_price
                commission = notional * 0.0002 * 2
                raw_pnl = (price - entry_price) * direction * size
                actual_pnl = raw_pnl - commission
                balance += actual_pnl
                peak = max(peak, balance)
                dd = (peak - balance) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
                local_trades.append({
                    "sym": sym, "dir": "LONG" if direction == 1 else "SHORT",
                    "entry": entry_price, "exit": price, "pnl": actual_pnl,
                    "raw": raw_pnl, "comm": commission, "reason": reason,
                    "bars": bars_held, "mult": mult,
                })
                direction = 0

    all_trades.extend(local_trades)
    print(f"  {len(local_trades)} trades")

mexc.disconnect()

wins = [t for t in all_trades if t["pnl"] > 0]
losses = [t for t in all_trades if t["pnl"] <= 0]
total_pnl = sum(t["pnl"] for t in all_trades)
wr = len(wins) / len(all_trades) * 100 if all_trades else 0
avg_w = np.mean([t["pnl"] for t in wins]) if wins else 0
avg_l = np.mean([t["pnl"] for t in losses]) if losses else 0
exp = total_pnl / len(all_trades) if all_trades else 0
total_comm = sum(t["comm"] for t in all_trades)
total_raw = sum(t["raw"] for t in all_trades)

print()
print("=" * 70)
print("  PAPER TEST v8 — FOCUSED TREND-FOLLOWING — REAL MEXC — $18")
print("  SOL: 3x futures, 0.8% trail, 30-bar stop, z>0.5 entry")
print("  XRP/ADA: 1x, 0.5% trail, 25-bar stop, z>0.3 entry")
print("  Min gap: 0.1% | Commission: 0.02% | Cooldown: 40 bars")
print("=" * 70)
print(f"  Capital:         ${capital:.2f}")
print(f"  Trades:          {len(all_trades)}")
print(f"  Win Rate:        {wr:.0f}%")
print(f"  Total PnL:       ${total_pnl:+.4f}")
print(f"  Final Equity:    ${balance:.2f}")
print(f"  Return:          {(balance / capital - 1) * 100:+.2f}%")
print(f"  Max Drawdown:    {max_dd:.2f}%")
print(f"  Expectancy:      ${exp:.4f}")
print(f"  Avg Win:         ${avg_w:.4f}")
print(f"  Avg Loss:        ${avg_l:.4f}")
print(f"  Commission:      ${total_comm:.4f}")
print(f"  Raw PnL:         ${total_raw:+.4f}")
if all_trades:
    profit_factor = abs(sum(t["pnl"] for t in wins)) / max(abs(sum(t["pnl"] for t in losses)), 0.0001) if losses else 999
    print(f"  Profit Factor:   {profit_factor:.2f}")
    monthly_est = exp * len(all_trades) * 3
    print(f"  Est. Monthly:    ${monthly_est:+.2f}")
print("=" * 70)
print()
for t in all_trades:
    arrow = "+" if t["pnl"] > 0 else ""
    mult_str = f"({t['mult']}x)" if t["mult"] > 1 else ""
    print(f"  {t['sym']:10s} {t['dir']:5s} {mult_str:4s} | entry={t['entry']:.4f} exit={t['exit']:.4f} | raw=${t['raw']:+.4f} comm=${t['comm']:.4f} net=${arrow}{t['pnl']:.4f} | {t['reason']} ({t['bars']} bars)")
print()