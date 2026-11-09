"""第二十五部分 (2)(3)：盤面（日線狀態）過濾的順勢單，以及「漂移本身」當策略。
輸出：python3 part25_regime_drift.py > results_part25_regime.txt

(2) 順勢單 + 盤面過濾：第二十二部分的突破擺動點事件（無背離 / 各種背離），只做「和盤面同方向」的順勢單。
    盤面：D = 日線狀態（part24_common.daily_regime：SMA50 + 20 天斜率，前一天資料）
          E = 該週期 EMA200（前一根收盤在上 / 下）
          D+E = 兩者同向
    進出場：C1S2（第二十三部分：突破訊號棒高點進場、停損回檔低點、2R、32 根）與統一出場（1.5 ATR / 3 ATR）。

(3) 漂移策略（日線，用 M15 合成日 K；訊號在當日收盤產生、持有到下一日收盤）：
    買進持有 / 日線狀態多空 / 日線狀態只做多 / 收盤在 SMA50 上下 / 20 日突破（Donchian）/ 60 日動能。
    部位 = 1 倍名目（不加槓桿），報酬以帳戶 % 表示。
    成本：每次換倉扣 0.02%（點差 + 滑價）；隔夜利息假設 做多 −5%/年、做空 −1%/年（沒有券商實際 swap 資料，只是估計）。
"""
import numpy as np
import pandas as pd
from part22_divergence import get_frames, indicators, detect, nonoverlap, SPLIT
from part23_trend_hook import trend_trades
from part24_common import sim, daily_regime, fmt, per_year, both_spaces

COST_SWITCH = 0.0002
SWAP_LONG, SWAP_SHORT = 0.05 / 365, 0.01 / 365


def cont_trades(df):
    X = indicators(df)
    idx = df.index
    reg = daily_regime(df)
    ema = np.sign(df.close - df.close.ewm(span=200, adjust=False).mean()).shift(1).to_numpy()
    A, U = [], []
    for Y, s_short in both_spaces(X):
        ev = detect(Y)
        for tr in trend_trades(Y, ev, idx, s_short):
            if tr["strat"] == "C1S2":
                A.append(tr)
        for e in ev:
            r = sim(Y, e["t"], 1)
            if r is not None:
                U.append(dict(cond=e["cond"], side=-s_short, entry_time=idx[r[0]], exit_time=idx[r[1]], pnl_R=r[2],
                              t=e["t"]))
    A = pd.DataFrame(A)
    U = pd.DataFrame(U)
    pos = pd.Series(np.arange(len(idx)), index=idx)
    for T in (A, U):
        # 盤面用訊號當下（進場那根的前一根）判斷
        k = pos.reindex(T.entry_time).to_numpy() - 1
        T["reg"] = reg[k]
        T["ema"] = ema[k]
    return A, U


def regime_section():
    F = get_frames()
    for tf in ("M15", "H1"):
        A, U = cont_trades(F[tf])
        print(f"\n{'=' * 110}\n## {tf} 順勢單 + 盤面過濾   研究期 2023~24 | 驗證期 2025~26/10 | 逐年")
        for title, T in (("C1S2", A), ("統一出場", U)):
            print(f"\n### {title}")
            for cnd in ("base", "push", "speed", "cci", "obv", "rsi", "macd", "stoch"):
                x = T[T.cond == cnd]
                filt = {
                    "不過濾": np.ones(len(x), bool),
                    "D 日線同向": x.side == x.reg,
                    "E EMA200同向": x.side == x.ema,
                    "D+E": (x.side == x.reg) & (x.side == x.ema),
                }
                for fn, m in filt.items():
                    split_line_side(f"{cnd} {fn}", x[m])
                print()


def split_line_side(name, T):
    if len(T) == 0:
        return
    x = nonoverlap(T)
    a, b = x[x.entry_time < SPLIT], x[x.entry_time >= SPLIT]
    sh = lambda g, s: g[g.side == s].pnl_R
    print(f"  {name:20s} 合計 {fmt(a.pnl_R):25s} {fmt(b.pnl_R):25s} | 多 {sh(a, 1).mean():+.2f}/{sh(b, 1).mean():+.2f}"
          f" 空 {sh(a, -1).mean():+.2f}/{sh(b, -1).mean():+.2f} (空筆數 {len(sh(a, -1))}/{len(sh(b, -1))}) | {per_year(x)}")


# ---------------------------------------------------------------- (3) 漂移策略
def daily_bars():
    m15 = get_frames()["M15"]
    d = m15.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return d


def backtest_pos(d, pos, name, swap=True):
    """pos：當日收盤時決定、持有到下一日收盤的部位（+1 / 0 / −1）。"""
    ret = d.close.pct_change().shift(-1)                   # 今日收 → 明日收
    p = pos.fillna(0)
    days = (d.index.to_series().shift(-1) - d.index.to_series()).dt.days.fillna(1)
    carry = np.where(p > 0, -SWAP_LONG, np.where(p < 0, -SWAP_SHORT, 0)) * days if swap else 0
    cost = (p.diff().abs().fillna(p.abs())) * COST_SWITCH
    r = (p * ret + carry - cost).dropna()
    r = r[r.index >= START]
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252)
    by = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    inmkt = (p[p.index >= START] != 0).mean()
    sw = (p.diff().abs() > 0).sum()
    return dict(name=name, total=eq.iloc[-1] - 1, cagr=cagr, maxdd=dd, sharpe=sharpe, inmkt=inmkt, switches=sw,
                **{str(y): v for y, v in by.items()}), r


def drift_section():
    d = daily_bars()
    global START
    START = d.index[70]                                     # 讓所有指標都有足夠暖機（約 2023/4 起）
    c = d.close
    sma = c.rolling(50).mean()
    up = (c > sma) & (sma > sma.shift(20))
    dn = (c < sma) & (sma < sma.shift(20))
    regime = pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=d.index)
    hi20, lo20 = d.high.rolling(20).max().shift(1), d.low.rolling(20).min().shift(1)
    don = pd.Series(np.nan, index=d.index)
    don[c > hi20] = 1
    don[c < lo20] = -1
    don = don.ffill().fillna(0)
    strats = {
        "買進持有": pd.Series(1.0, index=d.index),
        "日線狀態 多空（SMA50+斜率）": regime.astype(float),
        "日線狀態 只做多": (regime == 1).astype(float),
        "收盤在 SMA50 上做多 / 下做空": np.sign(c - sma),
        "20 日突破（Donchian）多空": don,
        "60 日動能 多空": np.sign(c / c.shift(60) - 1),
        "60 日動能 只做多": (c / c.shift(60) > 1).astype(float),
    }
    print(f"\n{'=' * 110}\n## (3) 漂移策略（日線，{START:%Y-%m-%d} ~ {d.index[-1]:%Y-%m-%d}，1 倍名目）")
    print(f"假設成本：換倉 {COST_SWITCH:.2%}、隔夜利息 做多 −5%/年、做空 −1%/年")
    rows, R = [], {}
    for k, p in strats.items():
        s, r = backtest_pos(d, p, k)
        rows.append(s)
        R[k] = r
        s2, _ = backtest_pos(d, p, k, swap=False)
        rows[-1]["total_no_swap"] = s2["total"]
    S = pd.DataFrame(rows).set_index("name")
    cols = ["total", "total_no_swap", "cagr", "maxdd", "sharpe", "inmkt", "switches"] + [c for c in S.columns if c.isdigit()]
    fm = S[cols].copy()
    for col in ["total", "total_no_swap", "cagr", "maxdd", "inmkt"] + [c for c in S.columns if c.isdigit()]:
        fm[col] = fm[col].map(lambda v: f"{v:+.1%}")
    fm["sharpe"] = fm["sharpe"].map(lambda v: f"{v:.2f}")
    print(fm.rename(columns={"total": "總報酬", "total_no_swap": "未扣利息", "cagr": "年化", "maxdd": "最大回撤",
                             "inmkt": "持倉時間", "switches": "換倉次數"}).to_string())

    # 在日線空頭期間的表現（漂移策略是否能在下跌中賺錢）
    print("\n  只看「日線空頭」日子（前一天判定）的累計報酬：")
    for k, r in R.items():
        m = regime.shift(0).reindex(r.index) == -1
        print(f"    {k:28s} {((1 + r[m]).prod() - 1):+.1%}（{m.sum()} 天）")
    # 主要下跌段
    print("\n  主要下跌段（買進持有跌幅 > 8% 的區間）各策略報酬：")
    bh = R["買進持有"]
    eq = (1 + bh).cumprod()
    dd = eq / eq.cummax() - 1
    in_dd, segs, s0 = False, [], None
    for t, v in dd.items():
        if not in_dd and v < 0:
            in_dd, s0, low_t, low_v = True, t, t, v
        if in_dd:
            if v < low_v:
                low_t, low_v = t, v
            if v == 0:
                if low_v < -0.08:
                    segs.append((s0, low_t, low_v))
                in_dd = False
    if in_dd and low_v < -0.08:
        segs.append((s0, low_t, low_v))
    for s, e, v in segs:
        line = "  ".join(f"{k[:8]} {((1 + R[k][(R[k].index >= s) & (R[k].index <= e)]).prod() - 1):+.1%}" for k in R)
        print(f"    {s:%Y-%m-%d}~{e:%Y-%m-%d}（買進持有 {v:+.1%}）: {line}")


if __name__ == "__main__":
    regime_section()
    drift_section()
