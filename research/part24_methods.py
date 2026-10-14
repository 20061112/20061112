"""第二十四部分 (3)：不同的背離算法比較。
輸出：python3 part24_methods.py > results_part24_methods.txt

在頂部空間定義（底部用 mirror 套同一段程式）。K=3 的擺動點在 p+3 收盤才確認。
  M1 即時        : 目前 K 棒創新高（> 前一個確認擺動高點），指標值 < 擺動高點那根的指標值（第二十二部分的做法）
  M2 經典確認    : 兩個確認擺動高點 P1 < P2，P2 價格較高、指標較低；P2+3 收盤才發訊號
  M3 指標波峰    : 同 M1，但比較「第二推期間指標最高值」與「第一推期間指標最高值」，不是單根的值
  M4 滾動視窗    : 收在 20 根新高，指標 < 前 5~19 根內的指標最高值（不需要擺動點）
  M5 斜率        : 收在 20 根新高，過去 20 根高點的回歸斜率 > 0，指標的回歸斜率 < 0
  M6 幅度門檻    : M1 且指標差距 >= 0.5 × 指標 200 根標準差
  M7 三重背離    : 三個連續確認擺動高點價格遞增、指標遞減；第三個點 +3 收盤發訊號
  M8 隱藏背離    : 兩個確認擺動高點，P2 價格較低、指標較高（傳統上視為「順勢做空」訊號）
  每種算法都在 Y 空間「做空」= 傳統的背離交易方向（M1~M7 為反轉，M8 為順勢）。

評估：
  f16      = 訊號後 16 根往「做空」方向的報酬（ATR，已扣點差）
  超額     = f16 − 同日線狀態的平均漂移（扣掉黃金多頭的 beta）
  反向單   = 統一出場（下一根開盤，停損 1.5 ATR / 停利 3 ATR / 32 根）做傳統方向
  順向單   = 同樣出場，但做相反方向
  每種算法 × 指標各自做不重疊交易，研究期 2023~24 | 驗證期 2025~26/10。
"""
import numpy as np
import pandas as pd
from part22_divergence import get_frames, indicators, nonoverlap, SPLIT, detect
from part24_common import sim, fwd, daily_regime, tstat, both_spaces

K = 3
IND = ["macd", "rsi", "stoch", "cci", "obv"]
METHODS = ["M1即時", "M2經典確認", "M3指標波峰", "M4滾動視窗", "M5斜率", "M6幅度門檻", "M7三重背離", "M8隱藏背離"]


def pivots(h):
    rm = pd.Series(h).rolling(2 * K + 1, center=True).max().to_numpy()
    return np.where(h == rm)[0]


def slope(a, n):
    x = np.arange(n) - (n - 1) / 2
    return pd.Series(a).rolling(n).apply(lambda y: np.dot(x, y), raw=True).to_numpy()


def detect_all(Y):
    h, l, atr = Y["h"], Y["l"], Y["atr"]
    n = len(h)
    ev = []                                                    # (method, ind, t)
    # M1 / M3 / M6：重用第二十二部分的即時偵測結構
    piv = pivots(h)
    piv = piv[piv > 250]
    std = {k: pd.Series(Y[k]).rolling(200).std().to_numpy() for k in IND}
    prev = None
    for P1 in piv:
        P0, prev = prev, P1
        tb = None
        for t in range(P1 + K + 1, min(P1 + 61, n)):
            if h[t] > h[P1]:
                tb = t
                break
        if tb is None or h[P1] - l[P1:tb].min() < atr[P1]:
            continue
        a1 = max(P1 - 40, P0 if P0 is not None else 0)
        found = set()
        top = h[P1]
        for t in range(tb, min(tb + 10, P1 + 61, n - 1)):
            if h[t] <= top:
                continue
            top = h[t]
            s2 = P1 + int(np.argmin(l[P1:t + 1]))
            s1 = a1 + int(np.argmin(l[a1:P1 + 1]))
            for k in IND:
                v = Y[k]
                if ("M1", k) not in found and v[t] < v[P1]:
                    found.add(("M1", k)); ev.append(("M1即時", k, t))
                if ("M6", k) not in found and v[P1] - v[t] >= 0.5 * std[k][t]:
                    found.add(("M6", k)); ev.append(("M6幅度門檻", k, t))
                if ("M3", k) not in found and v[s2:t + 1].max() < v[s1:P1 + 1].max():
                    found.add(("M3", k)); ev.append(("M3指標波峰", k, t))
    # M2 / M7 / M8：確認擺動點之間比較
    for i in range(1, len(piv)):
        P1, P2 = piv[i - 1], piv[i]
        if P2 - P1 > 60 or P2 - P1 < K + 1 or P2 + K >= n - 1:
            continue
        pull = min(h[P1], h[P2]) - l[P1:P2 + 1].min()
        if pull < atr[P1]:
            continue
        t = P2 + K
        for k in IND:
            v = Y[k]
            if h[P2] > h[P1] and v[P2] < v[P1]:
                ev.append(("M2經典確認", k, t))
                if i >= 2:
                    P0 = piv[i - 2]
                    if P1 - P0 <= 60 and h[P1] > h[P0] and v[P1] < v[P0]:
                        ev.append(("M7三重背離", k, t))
            if h[P2] < h[P1] and v[P2] > v[P1]:
                ev.append(("M8隱藏背離", k, t))
    # M4 / M5：不需要擺動點
    hh20 = pd.Series(h).rolling(20).max().to_numpy()
    newhi = (h >= hh20)
    sh = slope(h, 20)
    for k in IND:
        v = Y[k]
        vmax = pd.Series(v).shift(5).rolling(15).max().to_numpy()
        m4 = newhi & (v < vmax)
        sv = slope(v, 20)
        m5 = newhi & (sh > 0) & (sv < 0)
        for name, m in (("M4滾動視窗", m4), ("M5斜率", m5)):
            last = -99
            for t in np.where(m)[0]:
                if t < 300 or t >= n - 1:
                    continue
                if t - last > 5:                               # 同一段 5 根內只算第一次
                    ev.append((name, k, t))
                last = t
    return ev


def main():
    F = get_frames()
    for tf in ("M15", "H1"):
        df = F[tf]
        X = indicators(df)
        idx = df.index
        reg = daily_regime(df)
        c, atr = X["c"], X["atr"]
        f16a = np.full(len(c), np.nan)
        f16a[:-16] = (c[16:] - c[:-16]) / atr[:-16]
        drift = {r: np.nanmean(f16a[reg == r]) for r in (1, 0, -1)}
        rows = []
        for Y, s_short in both_spaces(X):
            for m, k, t in detect_all(Y):
                if not np.isfinite(reg[t]):
                    continue
                f = fwd(Y, t, 16, -1)
                rev, con = sim(Y, t, -1), sim(Y, t, 1)
                if rev is None:
                    continue
                rows.append(dict(method=m, ind=k, side=s_short, time=idx[t], reg=reg[t], f16=f,
                                 excess=f - s_short * drift[reg[t]],
                                 rev_e=idx[rev[0]], rev_x=idx[rev[1]], rev=rev[2],
                                 con_e=idx[con[0]], con_x=idx[con[1]], con=con[2]))
        E = pd.DataFrame(rows)
        E["res"] = E.time < SPLIT

        def tr(g, col):
            T = g.rename(columns={f"{col}_e": "entry_time", f"{col}_x": "exit_time", col: "pnl_R"})[["entry_time", "exit_time", "pnl_R"]]
            T = nonoverlap(T)
            a, b = T[T.entry_time < SPLIT].pnl_R, T[T.entry_time >= SPLIT].pnl_R
            return a.mean(), tstat(a), b.mean(), tstat(b)

        print(f"\n{'=' * 120}\n## {tf}")
        print("  算法       指標    事件數 | f16 研究  驗證 | 超額 研究   驗證 | 傳統方向單 研究(t)    驗證(t)   | 反方向單 研究(t)      驗證(t)")
        summ = []
        for m in METHODS:
            for k in IND:
                g = E[(E.method == m) & (E.ind == k)]
                if len(g) < 30:
                    continue
                a, b = g[g.res], g[~g.res]
                r = tr(g, "rev")
                cc = tr(g, "con")
                summ.append(dict(method=m, ind=k, n=len(g), ex_a=a.excess.mean(), ex_b=b.excess.mean(),
                                 rev_a=r[0], rev_b=r[2], con_a=cc[0], con_b=cc[2]))
                print(f"  {m:8s} {k:5s} {len(g):6d} | {a.f16.mean():+.3f} {b.f16.mean():+.3f} | {a.excess.mean():+.3f} {b.excess.mean():+.3f} |"
                      f" {r[0]:+.3f}({r[1]:+.1f}) {r[2]:+.3f}({r[3]:+.1f}) | {cc[0]:+.3f}({cc[1]:+.1f}) {cc[2]:+.3f}({cc[3]:+.1f})")
        S = pd.DataFrame(summ)
        print(f"\n  各算法平均（5 個指標）：")
        print(S.groupby("method", sort=False)[["ex_a", "ex_b", "rev_a", "rev_b", "con_a", "con_b"]].mean().round(3).to_string())
        print(f"\n  傳統方向單兩期都 > 0 的組合：{((S.rev_a > 0) & (S.rev_b > 0)).sum()} / {len(S)}；"
              f"反方向（順勢）單兩期都 > 0：{((S.con_a > 0) & (S.con_b > 0)).sum()} / {len(S)}")
        print(f"  超額 f16 兩期都 > 0（背離方向真的有額外報酬）：{((S.ex_a > 0) & (S.ex_b > 0)).sum()} / {len(S)}；"
              f"兩期都 < 0（反而延續）：{((S.ex_a < 0) & (S.ex_b < 0)).sum()} / {len(S)}")

        print("\n  依真實方向拆開（傳統方向單，不重疊，全部 5 個指標合併後各自不重疊）：")
        for m in METHODS:
            for s, sn in ((1, "做多(底背離)"), (-1, "做空(頂背離)")):
                g = E[(E.method == m) & (E.side == s)]
                out = []
                for k in IND:
                    gg = g[g.ind == k]
                    if len(gg) >= 30:
                        out.append(tr(gg, "rev"))
                if out:
                    o = np.array(out)
                    print(f"    {m:8s} {sn}: 研究 {np.nanmean(o[:, 0]):+.3f}  驗證 {np.nanmean(o[:, 2]):+.3f}")


if __name__ == "__main__":
    main()
