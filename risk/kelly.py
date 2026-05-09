import numpy as np
from typing import Tuple


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    avg_loss = abs(avg_loss)
    if avg_loss < 1e-15:
        return 0.0

    b = avg_win / avg_loss
    if b < 1e-15:
        return 0.0

    q = 1.0 - win_rate
    f_star = win_rate - q / b

    return max(0.0, min(f_star, 0.25))


def fractional_kelly(win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.5) -> float:
    f_star = kelly_fraction(win_rate, avg_win, avg_loss)
    return f_star * fraction


def position_size(
    account_equity: float,
    kelly_f: float,
    atr: float,
    risk_pct: float = 0.02,
    max_position_risk: float = 0.25,
) -> float:
    kelly_f = min(kelly_f, max_position_risk)

    risk_amount = account_equity * risk_pct
    trade_risk = min(kelly_f * account_equity, risk_amount)

    if atr < 1e-15:
        atr = 1.0

    position = trade_risk / atr

    return position


def calculate_trade_stats(trades: list) -> Tuple[float, float, float, float]:
    if not trades:
        return 0.5, 0.01, 0.01, 0.0

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]

    win_rate = len(wins) / len(trades) if trades else 0.5
    avg_win = np.mean(wins) if wins else 0.01
    avg_loss = np.mean(losses) if losses else 0.01

    kf = kelly_fraction(win_rate, avg_win, abs(avg_loss))

    return win_rate, avg_win, abs(avg_loss), kf
