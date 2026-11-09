"""XAUUSD H1「六合一」背離順勢策略 —— 單檔、可獨立執行的回測。

只需要一個 M15 資料檔（MT5 匯出格式，Tab 分隔），會自動合成 H1。
規則詳見 STRATEGY.md；本檔與研究程式（research/part32_final.py）逐筆結果一致，可用 verify.py 檢查。

用法：
  python3 h1_six_divergence.py                                   # 預設 data/XAUUSD_M15_2023_2026.csv，輸出到 output/
  python3 h1_six_divergence.py --data 你的檔案.csv --out output
  python3 h1_six_divergence.py --signals                         # 另外列出最後幾根 H1 仍有效的掛單
資料格式（MT5「匯出柱狀圖」）：
  <DATE>	<TIME>	<OPEN>	<HIGH>	<LOW>	<CLOSE>	<TICKVOL>	<VOL>	<SPREAD>
  2023.01.03	01:00:00	1827.05	1830.73	1826.18	1826.93	1313	0	50
  時間 = broker 時間；SPREAD 單位為點（0.01 美元）。
"""
import argparse
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 參數（全部凍結，請勿修改）
K = 3            # 擺動點：左右各 K 根
PULL_ATR = 1.0   # P1 之後到突破前的回檔至少 1 ATR
MAXGAP = 60      # 突破需在 P1 之後 60 根內
MAXPUSH = 10     # 第二推從第一根突破起最多看 10 根
WAIT = 4         # 訊號棒後 4 根內要突破訊號棒高點才進場
BUF = 0.1        # 停損放在回檔低點外 0.1 ATR
MIN_R = 0.5      # 停損距離至少 0.5 ATR
TP = 3.0         # 停利 3R
HOLD = 32        # 最長持有 32 根 H1
W = 1 / 6        # 每個版本每筆風險 1/6 R
SIX = ["push", "obv", "speed", "rsi", "cci", "stoch"]
NAME = {"push": "推進", "obv": "OBV", "speed": "速度衰減", "rsi": "RSI", "cci": "CCI", "stoch": "Stoch"}
SPLIT = pd.Timestamp("2025-01-01")   # 研究期 / 驗證期分界（只用於報表）


# ---------------------------------------------------------------- 資料
def load_m15(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    return df.set_index("time")[["open", "high", "low", "close", "tickvol", "spread"]]


def to_h1(m15):
    return m15.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "tickvol": "sum", "spread": "max"}).dropna()


# ---------------------------------------------------------------- 指標
def indicators(df):
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "tickvol"))
    s = pd.Series(c)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()                      # ATR14（簡單平均）
    macd = (s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()).to_numpy()
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100 - 100 / (1 + up / dn)).to_numpy()                            # RSI14（Wilder）
    hh, ll = pd.Series(h).rolling(14).max(), pd.Series(l).rolling(14).min()
    stoch = (100 * (s - ll) / (hh - ll)).rolling(3).mean().to_numpy()       # Stoch %K(14,3)
    tp = pd.Series((h + l + c) / 3)
    md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = ((tp - tp.rolling(20).mean()) / (0.015 * md)).to_numpy()         # CCI20
    obv = np.cumsum(np.sign(np.r_[0, np.diff(c)]) * v)                     # OBV（tick volume）
    return dict(o=o, h=h, l=l, c=c, atr=atr, macd=macd, rsi=rsi, stoch=stoch, cci=cci, obv=obv,
                spr=df.spread.to_numpy(float) * 0.01)


def mirror(X):
    """把資料上下翻轉：找底部 / 做空的邏輯 = 翻轉後找頂部 / 做多。"""
    Y = dict(X)
    Y["o"], Y["c"] = -X["o"], -X["c"]
    Y["h"], Y["l"] = -X["l"], -X["h"]
    Y["cci"], Y["obv"], Y["macd"] = -X["cci"], -X["obv"], -X["macd"]
    Y["rsi"], Y["stoch"] = 100 - X["rsi"], 100 - X["stoch"]
    return Y


def daily_regime(df):
    """日線狀態（只用前一天以前的資料）：+1 多頭 / −1 空頭 / 0 盤整。"""
    d = df.close.resample("1D").last().dropna()
    sma = d.rolling(50).mean()
    up = (d > sma) & (sma > sma.shift(20))
    dn = (d < sma) & (sma < sma.shift(20))
    reg = pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=d.index).shift(1)
    return reg.reindex(df.index.normalize()).ffill().to_numpy()


# ---------------------------------------------------------------- 訊號（在「向上突破」空間偵測；做空用 mirror 後的資料）
def detect(X):
    """回傳 [(版本, 訊號棒 t, P1)]：t 收盤時突破前一擺動高點 P1，且該版本背離成立（每個 P1 每個版本取第一根）。"""
    h, l, atr = X["h"], X["l"], X["atr"]
    n = len(h)
    rm = pd.Series(h).rolling(2 * K + 1, center=True).max().to_numpy()
    piv = np.where((h == rm) & np.isfinite(atr))[0]
    out, prev = [], None
    for P1 in piv:
        P0, prev = prev, P1
        if P1 < 250:                                                        # 暖機
            continue
        a1 = max(P1 - 40, P0 if P0 is not None else 0)
        s1 = a1 + int(np.argmin(l[a1:P1 + 1]))                              # 第一推起點
        A1 = (h[P1] - l[s1]) / atr[s1]                                      # 第一推幅度（起點 ATR 倍數）
        V1 = A1 / max(P1 - s1, 1)                                           # 第一推速度
        tb = next((t for t in range(P1 + K + 1, min(P1 + MAXGAP + 1, n)) if h[t] > h[P1]), None)
        if tb is None or h[P1] - l[P1:tb].min() < PULL_ATR * atr[P1]:
            continue
        found, top = {}, h[P1]
        for t in range(tb, min(tb + MAXPUSH, P1 + MAXGAP + 1, n - 1)):
            if h[t] <= top:                                                 # 只檢查第二推中創新高的 K 棒
                continue
            top = h[t]
            s2 = P1 + int(np.argmin(l[P1:t + 1]))                           # 第二推起點（回檔低點）
            A2 = (h[t] - l[s2]) / atr[s2]
            V2 = A2 / max(t - s2, 1)
            cond = dict(push=A2 < A1, speed=V2 < V1,
                        obv=X["obv"][t] < X["obv"][P1], rsi=X["rsi"][t] < X["rsi"][P1],
                        cci=X["cci"][t] < X["cci"][P1], stoch=X["stoch"][t] < X["stoch"][P1])
            for k, ok in cond.items():
                if ok and k not in found:
                    found[k] = t
            if len(found) == len(SIX):
                break
        out += [(k, t, P1) for k, t in found.items()]
    return out


# ---------------------------------------------------------------- 進場與出場（在偵測空間一律做多）
def exit_long(Y, ei, ep, stop):
    o, h, l, c, spr = Y["o"], Y["h"], Y["l"], Y["c"], Y["spr"]
    n = len(c)
    R = ep - stop
    tgt = ep + TP * R
    for j in range(ei, min(ei + HOLD + 1, n)):
        if l[j] <= stop:                                                    # 同根同時碰到：先算停損
            xp = stop if j == ei else min(stop, o[j])                       # 跳空穿過停損用開盤價
            return j, (xp - ep - spr[ei]) / R, "stop"
        if j > ei and h[j] >= tgt:                                          # 進場那根不給停利
            return j, (tgt - ep - spr[ei]) / R, "tp"
    j = min(ei + HOLD, n - 1)
    return j, (c[j] - ep - spr[ei]) / R, "time"


def nonoverlap(T):
    """同一版本一次只持有一筆。同一根進場有多個訊號時（同一根 K 棒同時突破兩個擺動點），
    取最近的擺動點 P1（停損放在最近的回檔低點）。排序規則固定，結果不受 pandas / numpy 版本影響。"""
    T = T.sort_values(["entry_time", "p1_time"], ascending=[True, False], kind="stable")
    keep, busy = [], pd.Timestamp.min
    for i, r in zip(T.index, T.itertuples()):
        if r.entry_time > busy:
            keep.append(i)
            busy = r.exit_time
    return T.loc[keep]


def backtest(h1):
    X = indicators(h1)
    idx = h1.index
    reg = daily_regime(h1)
    ema = np.sign(h1.close - h1.close.ewm(span=200, adjust=False).mean()).shift(1).to_numpy()
    rows = []
    for Y, side in ((X, 1), (mirror(X), -1)):                               # side：原始方向（+1 做多、−1 做空）
        o, h, l, atr = Y["o"], Y["h"], Y["l"], Y["atr"]
        n = len(h)
        for cond, t, P1 in detect(Y):
            A = atr[t]
            s2 = P1 + int(np.argmin(l[P1:t + 1]))
            base_stop = l[s2] - BUF * A
            ei = None
            for j in range(t + 1, min(t + 1 + WAIT, n - 1)):
                if h[j] > h[t]:                                             # 突破訊號棒高點 → 停損單成交
                    ei, ep = j, max(h[t], o[j])
                    break
                if l[j] <= base_stop:                                       # 先碰停損價 → 取消
                    break
            if ei is None:
                continue
            k = ei - 1                                                      # 盤面過濾用進場前一根
            if not (side == reg[k] and side == ema[k]):
                continue
            stop = min(base_stop, ep - MIN_R * A)
            xi, pnl_R, why = exit_long(Y, ei, ep, stop)
            Rp = ep - stop
            e_px = ep if side == 1 else -ep
            rows.append(dict(cond=cond, side=side, signal_time=idx[t], p1_time=idx[P1], entry_time=idx[ei], exit_time=idx[xi],
                             entry_px=e_px, stop_px=e_px - side * Rp, target_px=e_px + side * TP * Rp,
                             R_usd=Rp, R_atr=Rp / Y["atr"][ei - 1], spread=Y["spr"][ei], exit=why, pnl_R=pnl_R))
    T = pd.DataFrame(rows)
    T = pd.concat([nonoverlap(T[T.cond == k]) for k in SIX]).sort_values(["entry_time", "cond"], kind="stable").reset_index(drop=True)
    T["pnl_usd_per_oz"] = T.pnl_R * T.R_usd
    T["exit_px"] = T.entry_px + T.side * (T.pnl_usd_per_oz + T.spread)
    T["combo_R"] = T.pnl_R * W
    T["hold_hours"] = (T.exit_time - T.entry_time).dt.total_seconds() / 3600 + 1
    T["n_versions"] = T.groupby(["entry_time", "side"]).cond.transform("count")
    return T, X


# ---------------------------------------------------------------- 報表
def mtm_curve(h1, T, usd=False):
    """逐小時淨值：已實現 + 未實現（未實現用 K 棒最差價）。usd=True 時單位為「每版本 1 盎司」的美元，否則為組合 R。"""
    idx = h1.index
    H, L = h1.high.to_numpy(), h1.low.to_numpy()
    pos = pd.Series(np.arange(len(idx)), index=idx)
    real, worst, nopen = np.zeros(len(idx)), np.zeros(len(idx)), np.zeros(len(idx), int)
    for r in T.itertuples():
        i0, i1 = pos[r.entry_time], pos[r.exit_time]
        real[i1] += r.pnl_usd_per_oz if usd else r.combo_R
        if i1 > i0:
            k = np.arange(i0, i1)
            px = L[k] if r.side == 1 else H[k]
            worst[k] += r.side * (px - r.entry_px) * (1 if usd else W / r.R_usd)
            nopen[k] += 1
    cum = np.cumsum(real)
    return pd.Series(cum, index=idx), pd.Series(cum + worst, index=idx), nopen, worst


def report(h1, T):
    ec, em, nopen, worst = mtm_curve(h1, T)
    r = T.combo_R
    days = pd.date_range(T.entry_time.min().normalize(), T.exit_time.max().normalize(), freq="B")
    daily = r.groupby(T.exit_time.dt.normalize()).sum().reindex(days, fill_value=0)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    S = T.groupby(["entry_time", "side"]).size()
    usd = T.pnl_usd_per_oz
    uc, um, _, uw = mtm_curve(h1, T, usd=True)
    L = [f"XAUUSD H1 六合一背離順勢（3R 停利，每版本 1/6 R），{T.entry_time.min():%Y-%m-%d} ~ {T.exit_time.max():%Y-%m-%d}",
         f"  交易 {len(T)} 筆（多 {(T.side == 1).sum()} / 空 {(T.side == -1).sum()}），不同訊號 {len(S)} 個",
         f"  勝率 {(T.pnl_R > 0).mean():.1%}；每筆 {T.pnl_R.mean():+.3f}R；PF {gp / gl:.2f}",
         f"  研究期（< 2025）每筆 {T[T.entry_time < SPLIT].pnl_R.mean():+.3f}R；驗證期 {T[T.entry_time >= SPLIT].pnl_R.mean():+.3f}R",
         f"  組合總損益 {r.sum():+.2f}R；最大回撤 平倉 {(ec.cummax() - ec).max():.2f}R / 含浮動 {(em.cummax() - em).max():.2f}R",
         f"  日 Sharpe {daily.mean() / daily.std() * np.sqrt(252):.2f}；同時持倉最多 {nopen.max()} 筆；同時最大浮虧 {worst.min():+.2f}R",
         "  逐年（組合 R）：" + "  ".join(f"{y}: {v:+.1f}" for y, v in r.groupby(T.entry_time.dt.year).sum().items()),
         "  出場：" + "  ".join(f"{e} {(T.exit == e).mean():.0%}" for e in ("tp", "stop", "time")),
         f"  每版本固定 0.01 手（1 盎司）：總 ${usd.sum():,.0f}；最大回撤 平倉 ${(uc.cummax() - uc).max():,.0f} / 含浮動 ${(um.cummax() - um).max():,.0f}；"
         f"同時最大浮虧 ${-uw.min():,.0f}（已扣點差、未扣隔夜利息）",
         "    逐年 " + "  ".join(f"{y}: ${v:+,.0f}" for y, v in usd.groupby(T.entry_time.dt.year).sum().items())]
    L.append("  依版本：")
    for k in SIX:
        g = T[T.cond == k]
        L.append(f"    {NAME[k]:6s} {len(g):4d} 筆 每筆 {g.pnl_R.mean():+.3f}R")
    return "\n".join(L), ec, em


def pending_signals(h1, X):
    """列出最後 WAIT 根內出現、尚未觸發、且目前盤面過濾仍成立的訊號（給實盤參考）。"""
    idx, n = h1.index, len(h1)
    reg = daily_regime(h1)
    ema_now = np.sign(h1.close.iloc[-1] - h1.close.ewm(span=200, adjust=False).mean().iloc[-1])
    out = []
    for Y, side in ((X, 1), (mirror(X), -1)):
        h, l, atr = Y["h"], Y["l"], Y["atr"]
        for cond, t, P1 in detect(Y):
            if t < n - WAIT:
                continue
            A = atr[t]
            s2 = P1 + int(np.argmin(l[P1:t + 1]))
            base_stop = l[s2] - BUF * A
            later = range(t + 1, n)
            if any(h[j] > h[t] for j in later) or any(l[j] <= base_stop for j in later):
                continue                                                    # 已經觸發或已取消
            ok = side == reg[-1] and side == ema_now
            trig = h[t] if side == 1 else -h[t]
            stp = base_stop if side == 1 else -base_stop
            out.append(dict(版本=NAME[cond], 方向="多" if side == 1 else "空", 訊號K棒=idx[t], 有效到=idx[t] + pd.Timedelta(hours=WAIT),
                            掛單價=round(trig, 2), 預估停損=round(stp, 2), 盤面過濾=ok))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.path.join(here, "data", "XAUUSD_M15_2023_2026.csv"))
    ap.add_argument("--out", default=os.path.join(here, "output"))
    ap.add_argument("--signals", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    h1 = to_h1(load_m15(a.data))
    T, X = backtest(h1)
    txt, ec, em = report(h1, T)
    print(txt)
    open(os.path.join(a.out, "stats.txt"), "w", encoding="utf-8").write(txt + "\n")
    D = T.copy()
    D["cond"] = D.cond.map(NAME)
    D["side"] = D.side.map({1: "多", -1: "空"})
    D.round({c: 4 for c in D.select_dtypes("number").columns}).to_csv(os.path.join(a.out, "trades.csv"), index=False, encoding="utf-8-sig")
    print(f"\n已輸出 {os.path.join(a.out, 'trades.csv')}、stats.txt")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(em.index, em.values, color="#9aa7b8", lw=0.8, label="mark-to-market")
        ax.plot(ec.index, ec.values, color="#1f5aa6", lw=1.6, label="realized")
        ax.axvline(SPLIT, color="#888", ls="--", lw=1)
        ax.set_ylabel("combo R"); ax.set_title("XAUUSD H1 six-divergence continuation (3R)"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "equity.png"), dpi=110)
        print("已輸出 equity.png")
    except ImportError:
        print("（沒有安裝 matplotlib，略過圖表）")
    if a.signals:
        P = pending_signals(h1, X)
        print(f"\n最後一根 H1：{h1.index[-1]}；仍有效的掛單：")
        print(P.to_string(index=False) if len(P) else "  （無）")


if __name__ == "__main__":
    main()
