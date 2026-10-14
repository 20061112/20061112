"""第二十三部分：(A) 背離事件改做順勢；(B) RSI 拐頭確認後再進場。
輸出：python3 part23_trend_hook.py > results_part23.txt

研究期 2023~2024、驗證期 2025~2026/10；M5（2023/6 起）/ M15 / H1。

(A) 順勢：沿用第二十二部分的事件（突破前高 + 各種背離），方向改成「跟著突破走」。
    進場  C0 = 訊號棒下一根開盤；C1 = 4 根內突破訊號棒高點（停損單，先碰停損價則取消）
    停損  S1 = 訊號棒低點 − 0.1 ATR；S2 = 第二推起點（回檔低點）− 0.1 ATR；距離至少 0.5 ATR
    出場  2R 停利或最長 32 根；同根同時碰停損停利算停損；扣點差。

(B) RSI 拐頭：
    B1 背離 + 拐頭：背離訊號後 8 根內 RSI 拐頭（前一根是 RSI 局部極值、這一根反向）→ 下一根開盤做反轉；
       停損 = 至今極值外 0.1 ATR（至少 0.5 ATR），2R / 32 根。
    B2 單純 RSI 超賣 / 超買拐頭（不需要背離），以超賣為例：
       RSI 在 j-1 為局部最低（RSI[j-1] < RSI[j-2] 且 RSI[j] > RSI[j-1]），且 RSI[j-1] <= L（L = 30 / 20 / 15 / 10），
       拐頭幅度 RSI[j] − RSI[j-1] >= H（H = 0 / 2）→ j+1 開盤做多。
       停損 = 過去 10 根最低點 − 0.1 ATR（至少 0.5 ATR），2R / 32 根。
       對照組「不等拐頭」：RSI 第一次跌破 L 的那根收盤後直接進場，停損同樣定義。
"""
import numpy as np
import pandas as pd
from part22_divergence import (get_frames, indicators, mirror, detect, nonoverlap, SPLIT,
                               BUF, MIN_R, TP, HOLD, DIVS)

WAIT_C = 4
WAIT_HOOK = 8


def exit_long(X, ei, ep, stop, entry_bar_tp=False):
    """轉換後空間做多：回傳 (xi, pnl_R, why)。進場那根不給停利；進場那根碰停損算停損。"""
    o, h, l, c, spr = X["o"], X["h"], X["l"], X["c"], X["spr"]
    n = len(c)
    R = ep - stop
    tgt = ep + TP * R
    for j in range(ei, min(ei + HOLD + 1, n)):
        if l[j] <= stop:
            xp = stop if j == ei else min(stop, o[j])
            return j, ((xp - ep) - spr[ei]) / R, "stop"
        if j > ei and h[j] >= tgt:
            return j, ((tgt - ep) - spr[ei]) / R, "tp"
    j = min(ei + HOLD, n - 1)
    return j, ((c[j] - ep) - spr[ei]) / R, "time"


def flip(X):
    """把「做空」轉成「做多」的空間：價格與指標再取一次負號。"""
    return mirror(X)


# ---------------------------------------------------------------- (A) 順勢
def trend_trades(X, events, idx, side):
    o, h, l, atr = X["o"], X["h"], X["l"], X["atr"]
    n = len(h)
    out = []
    for e in events:
        t, P1 = e["t"], e["P1"]
        A = atr[t]
        s2 = P1 + int(np.argmin(l[P1:t + 1]))
        for sname, base_stop in (("S1", l[t] - BUF * A), ("S2", l[s2] - BUF * A)):
            for entry in ("C0", "C1"):
                ei = None
                if entry == "C0":
                    if t + 1 < n:
                        ei, ep = t + 1, o[t + 1]
                else:
                    for j in range(t + 1, min(t + 1 + WAIT_C, n - 1)):
                        if h[j] > h[t]:
                            ei, ep = j, max(h[t], o[j])
                            break
                        if l[j] <= base_stop:
                            break
                if ei is None:
                    continue
                stop = min(base_stop, ep - MIN_R * A)
                if ep - stop <= 0:
                    continue
                xi, r, why = exit_long(X, ei, ep, stop)
                out.append(dict(cond=e["cond"], strat=f"{entry}{sname}", side=-side, entry_time=idx[ei],
                                exit_time=idx[xi], pnl_R=r, exit=why, R_atr=(ep - stop) / A))
    return out


# ---------------------------------------------------------------- (B1) 背離 + RSI 拐頭（反轉方向）
def hook_after_div(X, events, idx, side):
    """X 為頂背離空間；轉成做多空間後 RSI 拐頭 = RSI' 局部最低後上升。"""
    Y = flip(X)
    o, h, l, atr, rsi = Y["o"], Y["h"], Y["l"], Y["atr"], Y["rsi"]
    n = len(h)
    out = []
    for e in events:
        t = e["t"]
        A = atr[t]
        ext = l[t]                       # 做多空間中，背離極值是低點
        ei = None
        for j in range(t + 1, min(t + 1 + WAIT_HOOK, n - 1)):
            if l[j] <= ext - BUF * A:    # 先跌破極值（= 原空間創新高）→ 取消
                break
            ext = min(ext, l[j])
            if rsi[j - 1] < rsi[j - 2] and rsi[j] > rsi[j - 1]:
                ei, ep = j + 1, o[j + 1]
                break
        # 訊號棒本身就是拐頭的情況也算（RSI 在 t 已比 t-1 反向）
        if ei is None:
            continue
        stop = min(ext - BUF * A, ep - MIN_R * A)
        xi, r, why = exit_long(Y, ei, ep, stop)
        out.append(dict(cond=e["cond"], strat="B1", side=side, entry_time=idx[ei], exit_time=idx[xi], pnl_R=r, exit=why))
    return out


# ---------------------------------------------------------------- (B2) 單純 RSI 超賣拐頭
def rsi_hook(Y, idx, side, L, H, wait_hook=True):
    """Y 為做多空間（超賣 → 做多）。"""
    o, h, l, c, atr, rsi = Y["o"], Y["h"], Y["l"], Y["c"], Y["atr"], Y["rsi"]
    n = len(c)
    out = []
    if wait_hook:
        sig = np.where((rsi[1:-1] < rsi[:-2]) & (rsi[2:] > rsi[1:-1]) & (rsi[1:-1] <= L)
                       & (rsi[2:] - rsi[1:-1] >= H))[0] + 2          # j：拐頭那根
    else:
        sig = np.where((rsi[1:] <= L) & (rsi[:-1] > L))[0] + 1      # 第一次跌破 L
    for j in sig:
        if j < 300 or j + 1 >= n or not atr[j] > 0:
            continue
        A = atr[j]
        ei, ep = j + 1, o[j + 1]
        stop = min(l[j - 9:j + 1].min() - BUF * A, ep - MIN_R * A)
        xi, r, why = exit_long(Y, ei, ep, stop)
        f8 = (c[j + 8] - c[j] - Y["spr"][j]) / A if j + 8 < n else np.nan
        f16 = (c[j + 16] - c[j] - Y["spr"][j]) / A if j + 16 < n else np.nan
        out.append(dict(L=L, H=H, hook=wait_hook, side=side, entry_time=idx[ei], exit_time=idx[xi],
                        pnl_R=r, exit=why, f8=f8, f16=f16, rsi_low=rsi[j - 1] if wait_hook else rsi[j]))
    return out


# ---------------------------------------------------------------- 報表
def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 2 else np.nan


def fmt(r):
    return f"n={len(r):5d} {r.mean():+.3f}R t={tstat(r):+5.2f}" if len(r) >= 5 else f"n={len(r):5d}       -        "


def per_year(T):
    return " ".join(f"{y}:{v.mean():+.2f}" for y, v in T.groupby(T.entry_time.dt.year).pnl_R)


def line(name, T):
    x = nonoverlap(T)
    a, b = x[x.entry_time < SPLIT], x[x.entry_time >= SPLIT]
    print(f"  {name:22s} {fmt(a.pnl_R):26s} {fmt(b.pnl_R):26s} {per_year(x)}")


def main():
    for tf, df in get_frames().items():
        idx = df.index
        X = indicators(df)
        print(f"\n{'=' * 100}\n## {tf}")
        A, B1, B2 = [], [], []
        # side：原始空間方向。頂背離空間（X）對應 side=-1；底背離空間（mirror）對應 side=+1
        for side, Y in ((-1, X), (1, mirror(X))):
            ev = detect(Y)
            A += trend_trades(Y, ev, idx, side)
            B1 += hook_after_div(Y, ev, idx, side)
        # B2：做多空間 = 原始（超賣做多）與 mirror（超買做空）
        for side, Y in ((1, X), (-1, mirror(X))):
            for L in (30, 20, 15, 10):
                for H in (0, 2):
                    B2 += rsi_hook(Y, idx, side, L, H, True)
                B2 += rsi_hook(Y, idx, side, L, 0, False)
        A, B1, B2 = pd.DataFrame(A), pd.DataFrame(B1), pd.DataFrame(B2)

        print("\n### A. 背離事件改做順勢（不重疊）      研究期 2023~24             驗證期 2025~26/10          逐年")
        for st in ("C0S1", "C0S2", "C1S1", "C1S2"):
            print(f"\n  [{st}]")
            for cnd in ["base"] + DIVS:
                line(cnd, A[(A.cond == cnd) & (A.strat == st)])
        x = A[A.strat == "C1S2"]
        print(f"\n  C1S2 停損距離中位數 {x.R_atr.median():.2f} ATR；出場：停損 {(x.exit == 'stop').mean():.2f} 停利 {(x.exit == 'tp').mean():.2f} 時間 {(x.exit == 'time').mean():.2f}")

        print("\n### A2. 順勢單拆多空 + 用 EMA200 判斷大方向（C1S2，前一根收盤在 EMA 上/下）")
        ema = df.close.ewm(span=200, adjust=False).mean()
        A["tr"] = np.sign(df.close - ema).shift(1).reindex(A.entry_time).to_numpy()
        for cnd in ["base"] + DIVS:
            x = A[(A.cond == cnd) & (A.strat == "C1S2")]
            for lab, m in (("多", x.side == 1), ("空", x.side == -1),
                           ("順勢多", (x.side == 1) & (x.tr == 1)), ("順勢空", (x.side == -1) & (x.tr == -1)),
                           ("逆勢多", (x.side == 1) & (x.tr == -1)), ("逆勢空", (x.side == -1) & (x.tr == 1))):
                line(f"{cnd} {lab}", x[m])
        print("\n  對照：任意 K 棒下一根突破其高點做多 / 跌破其低點做空（停損 S1、2R、32 根，可重疊）")
        for side, Y in ((1, X), (-1, mirror(X))):
            rows = []
            for t in range(300, len(df) - 2):
                j = t + 1
                if Y["h"][j] > Y["h"][t]:
                    a_ = Y["atr"][t]
                    ep = max(Y["h"][t], Y["o"][j])
                    xi, r, _ = exit_long(Y, j, ep, min(Y["l"][t] - BUF * a_, ep - MIN_R * a_))
                    rows.append((idx[j], r))
            Rr = pd.DataFrame(rows, columns=["entry_time", "pnl_R"])
            a, b = Rr[Rr.entry_time < SPLIT].pnl_R, Rr[Rr.entry_time >= SPLIT].pnl_R
            print(f"  {'做多' if side == 1 else '做空'}：研究期 {a.mean():+.3f}R（{len(a)}）  驗證期 {b.mean():+.3f}R（{len(b)}）")

        print("\n### B1. 背離後等 RSI 拐頭才做反轉（8 根內）      研究期              驗證期")
        for cnd in ["base"] + DIVS:
            line(cnd, B1[B1.cond == cnd])

        print("\n### B2. 單純 RSI14 超賣/超買拐頭（不需要背離）")
        print(f"  {'規則':22s} {'研究期':26s} {'驗證期':26s} 逐年")
        for L in (30, 20, 15, 10):
            line(f"RSI<={L} 直接進(不等拐頭)", B2[(B2.L == L) & (~B2.hook)])
            for H in (0, 2):
                line(f"RSI<={L} 拐頭>={H}", B2[(B2.L == L) & B2.hook & (B2.H == H)])
        print("\n  前瞻報酬（反轉方向，ATR，已扣點差，事件可重疊）：拐頭>=0 那根收盤後 8 / 16 根")
        for L in (30, 20, 15, 10):
            x = B2[(B2.L == L) & B2.hook & (B2.H == 0)]
            a, b = x[x.entry_time < SPLIT], x[x.entry_time >= SPLIT]
            print(f"  RSI<={L:2d}: 研究期 n={len(a):5d} f8 {a.f8.mean():+.3f} (t{tstat(a.f8):+.1f}) f16 {a.f16.mean():+.3f} (t{tstat(a.f16):+.1f})"
                  f" | 驗證期 n={len(b):5d} f8 {b.f8.mean():+.3f} (t{tstat(b.f8):+.1f}) f16 {b.f16.mean():+.3f} (t{tstat(b.f16):+.1f})")


if __name__ == "__main__":
    main()
