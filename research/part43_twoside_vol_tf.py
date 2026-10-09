"""第四十三部分：雙向進場、波動擴張濾網、多時框拆解與擴充。

基準（第 42 部分）：OU z，|z| 穿越 3，回看 70 / 100 / 130 / 200 小時；每組權重相同的集成；
出場：停損 1.5 ATR、最多 40 根、週五最後一根平倉、週五 20 點後不開新單；點差 + 隔夜；R = 損益 / 初始停損；同一設定最多 1 筆。

A. 進場方式（M5 + M15，8 組集成）
   順勢市價    下一根開盤往 OU 方向進（基準）
   雙向市價    下一根開盤同時做多 + 做空各一筆
   OCO b       訊號收盤價 ± b·ATR 各掛突破單，12 根內先碰到的成交、另一張取消（b = 0.5 / 1.0）
   順勢突破 b  只掛 OU 方向的突破單
   逆勢市價    下一根開盤往反方向進（對照）
   突破單成交那一根：若最低（做多）已碰到停損價就算停損（保守）。
B. 波動擴張：訊號當下 ATR14 / ATR200（同週期）三分位、以及 > 1 / > 1.2 濾網；對順勢市價、雙向市價、OCO 1.0 都做。
C. 時框：M1 / M5 / M10 / M15 / M20 / M30 / H1 / H2，每個時框 4 組（4 種回看）；
   各時框 Sharpe、獲利占比、每日 R 相關矩陣；逐步加入時框的集成效果。
"""
import numpy as np
import pandas as pd
from load import load
from spring import resample, atr
from part37_wf import zscore
from part39_ou_vs_er import Frame, daily, SWAP, SPLIT

HOURS = [70, 100, 130, 200]
THR, SL, HOLD, WAIT = 3.0, 1.5, 40, 12
TFS = {"1min": 1, "5min": 5, "10min": 10, "15min": 15, "20min": 20, "30min": 30, "1h": 60, "2h": 120}


class TFrame(Frame):
    def __init__(self, m1, tf, mins):
        self.tf, self.mins = tf, mins
        df = resample(m1, tf)
        self.df, self.idx = df, df.index
        self.o, self.h, self.l, self.c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.A = atr(df).to_numpy()
        self.A200 = atr(df, 200).to_numpy()
        self.spr = df.spread.to_numpy(float) * 0.01
        nxt = np.r_[self.idx[1:].dayofweek, [0]]
        self.wk_last = (nxt < self.idx.dayofweek) | (np.r_[np.diff(self.idx.asi8), [0]] > 2 * 86400e9)
        self.wk_last[-1] = True
        self.day = self.idx.normalize().asi8
        self.dow = self.idx.dayofweek.to_numpy()
        self.hour = self.idx.hour.to_numpy()


def signals(F, L):
    z, _ = zscore(F.df, "ou", L)
    a = np.abs(z)
    prev = np.r_[np.nan, a[:-1]]
    t = np.where((a >= THR) & (prev < THR))[0]
    t = t[(t + 1 < len(z)) & ~np.isnan(F.A[t])]
    t = t[~((F.dow[t + 1] == 4) & (F.hour[t + 1] >= 20))]
    return t, -np.sign(z[t]).astype(int)


def manage(F, k, d, e, a, check_fill_bar=False):
    """從第 k 根（進場那根）開始管理部位；回傳 (出場 index, R)。"""
    n = len(F.c)
    risk = SL * a
    stop = e - d * risk
    j_out, x = None, None
    for j in range(k, min(k + HOLD, n)):
        if (F.l[j] <= stop) if d == 1 else (F.h[j] >= stop):
            j_out, x = j, stop
            break
        if F.wk_last[j]:
            j_out, x = j, F.c[j]
            break
    if j_out is None:
        j_out = min(k + HOLD, n) - 1
        x = F.c[j_out]
    usd = (x - e) * d - F.spr[k] - SWAP * F.nights(k, j_out)
    return j_out, usd / risk


def run(F, ts, ds, mode, b=None, vfilt=None):
    rows = []
    busy = -1
    n = len(F.c)
    for t, d in zip(ts, ds):
        k = t + 1
        if k <= busy:
            continue
        a = F.A[t]
        vr = a / F.A200[t] if F.A200[t] > 0 else np.nan
        if vfilt is not None and not (vfilt[0] <= vr < vfilt[1]):
            continue
        out = []
        if mode in ("trend", "counter", "both"):
            dirs = {"trend": [d], "counter": [-d], "both": [d, -d]}[mode]
            for dd in dirs:
                out.append(manage(F, k, dd, F.o[k], a) + (dd,))
        else:                                     # 突破單
            ref = F.c[t]
            levels = {1: ref + b * a, -1: ref - b * a}
            allowed = [d] if mode == "tstop" else [1, -1]
            fill = None
            for j in range(k, min(k + WAIT, n)):
                hit = []
                for dd in allowed:
                    lv = levels[dd]
                    if (F.h[j] >= lv) if dd == 1 else (F.l[j] <= lv):
                        px = max(lv, F.o[j]) if dd == 1 else min(lv, F.o[j])
                        hit.append((abs(F.o[j] - lv), dd, px))
                if hit:
                    _, dd, px = min(hit)          # 兩邊同一根都碰到：取離開盤近的那邊
                    fill = (j, dd, px)
                    break
            if fill is None:
                continue
            j, dd, px = fill
            out.append(manage(F, j, dd, px, a) + (dd,))
        for j_out, R, dd in out:
            rows.append(dict(time=F.idx[t], xtime=F.idx[j_out], side=dd, trend=int(dd == d), R=R, vr=vr))
        busy = max(o[0] for o in out)
    T = pd.DataFrame(rows, columns=["time", "xtime", "side", "trend", "R", "vr"])
    T["xtime"] = pd.to_datetime(T.xtime)
    return T


def sharpe(s):
    return s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else np.nan


def ens(series_list, days):
    s = pd.concat(series_list, axis=1).mean(axis=1)
    eq = s.cumsum().to_numpy()
    mdd = -(eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
    first = [d for d in days if pd.Timestamp(d) < SPLIT]
    return dict(S=sharpe(s), S1=sharpe(s.loc[first]), S2=sharpe(s.drop(first)), PM=s.sum() / mdd if mdd > 0 else np.nan,
                total=s.sum(), mdd=mdd), s


def fmt(lab, e, books):
    n = sum(len(b) for b in books)
    allR = pd.concat([b.R for b in books]) if n else pd.Series(dtype=float)
    return (f"  {lab:<30} 筆數 {n:4d}（{n / 192:.2f}/天） 每筆 {allR.mean():+.3f}R 勝率 {np.mean(allR > 0):.0%} | "
            f"集成 Sharpe {e['S']:+.2f}（前 {e['S1']:+.2f} / 後 {e['S2']:+.2f}） 獲利/回撤 {e['PM']:5.1f}")


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    days = sorted(set(m1.index.date))
    frames = {tf: TFrame(m1, tf, mins) for tf, mins in TFS.items()}
    sigs = {}
    for tf, F in frames.items():
        for hrs in HOURS:
            sigs[(tf, hrs)] = signals(F, int(hrs * 60 / F.mins))
    lines = []
    P = lines.append
    base_keys = [(tf, h) for tf in ("5min", "15min") for h in HOURS]

    # ---------- A ----------
    P("=" * 130)
    P("A. 進場方式（M5 + M15 × 4 種回看 = 8 組集成；|z| 穿越 3）")
    modes = [("順勢市價（基準）", "trend", None), ("逆勢市價（對照）", "counter", None), ("雙向市價", "both", None),
             ("OCO 0.5 ATR", "oco", 0.5), ("OCO 1.0 ATR", "oco", 1.0),
             ("順勢突破 0.5 ATR", "tstop", 0.5), ("順勢突破 1.0 ATR", "tstop", 1.0)]
    for lab, mode, b in modes:
        books = [run(frames[tf], *sigs[(tf, h)], mode, b) for tf, h in base_keys]
        e, _ = ens([daily(B, days) for B in books], days)
        extra = ""
        if mode in ("oco", "both"):
            allT = pd.concat(books)
            extra = (f" | 順勢那邊 {allT[allT.trend == 1].R.mean():+.3f}R（{(allT.trend == 1).mean():.0%}）"
                     f" 逆勢那邊 {allT[allT.trend == 0].R.mean():+.3f}R")
        P(fmt(lab, e, books) + extra)

    # ---------- B ----------
    P("\n" + "=" * 130)
    P("B. 波動擴張：訊號當下 ATR14 / ATR200（M5 + M15 8 組）")
    allv = pd.concat([run(frames[tf], *sigs[(tf, h)], "trend") for tf, h in base_keys]).vr
    q1, q2 = np.nanquantile(allv, [1 / 3, 2 / 3])
    P(f"  訊號的波動比：中位數 {allv.median():.2f}，三分位切點 {q1:.2f} / {q2:.2f}；全部 K 棒的中位數約 1.0")
    for lab, mode, b in (("順勢市價", "trend", None), ("雙向市價", "both", None), ("OCO 1.0", "oco", 1.0),
                         ("逆勢市價", "counter", None)):
        for vlab, vf in (("全部", None), (f"低 <{q1:.2f}", (0, q1)), (f"中", (q1, q2)), (f"高 ≥{q2:.2f}", (q2, 99)),
                         ("> 1.0", (1.0, 99)), ("> 1.2", (1.2, 99)), ("≤ 1.0（未擴張）", (0, 1.0))):
            books = [run(frames[tf], *sigs[(tf, h)], mode, b, vf) for tf, h in base_keys]
            e, _ = ens([daily(B, days) for B in books], days)
            P(fmt(f"{lab} 波動 {vlab}", e, books))
        P("")

    # ---------- C ----------
    P("=" * 130)
    P("C. 時框拆解（順勢市價；每個時框 = 4 種回看的集成）")
    tf_series, tf_books = {}, {}
    for tf in TFS:
        books = [run(frames[tf], *sigs[(tf, h)], "trend") for h in HOURS]
        e, s = ens([daily(B, days) for B in books], days)
        tf_series[tf], tf_books[tf] = s, books
        P(fmt(f"{tf}", e, books) + f"  總 R（4 組平均）{e['total']:+.1f}  回撤 {e['mdd']:.1f}")
    S = pd.DataFrame(tf_series)
    S.to_csv("part43_tf_daily.csv")
    P("\n  每日 R 相關矩陣")
    P(S.corr().round(2).to_string())
    P("\n  有交易的日子才算的相關（兩個時框同一天都有出場）")
    nz = S.where(S != 0)
    P(nz.corr(min_periods=10).round(2).to_string())
    tot = S.sum()
    P("\n  獲利占比（各時框總 R / 全部時框總 R 合計；只算正的）")
    pos = tot.clip(lower=0)
    P("    " + "  ".join(f"{k} {v:+.1f}R（{v / pos.sum():.0%}）" for k, v in tot.items()))
    P("\n  逐月 R（各時框 4 組平均）")
    mo = S.copy()
    mo.index = pd.to_datetime(mo.index)
    P(mo.groupby(mo.index.strftime("%m")).sum().round(1).T.to_string())

    P("\n  集成組合（每個時框權重相同）")
    combos = [("M5+M15（基準）", ["5min", "15min"]),
              ("M5+M10+M15", ["5min", "10min", "15min"]),
              ("M5+M10+M15+M20", ["5min", "10min", "15min", "20min"]),
              ("M5~M30（5 個）", ["5min", "10min", "15min", "20min", "30min"]),
              ("M1+M5+M15", ["1min", "5min", "15min"]),
              ("M1~M30（6 個）", ["1min", "5min", "10min", "15min", "20min", "30min"]),
              ("全部 8 個", list(TFS)),
              ("M5+M15+H1", ["5min", "15min", "1h"]),
              ("M5+M15+H2", ["5min", "15min", "2h"])]
    for lab, tfs in combos:
        e, _ = ens([S[t] for t in tfs], days)
        P(f"  {lab:<20} Sharpe {e['S']:+.2f}（前 {e['S1']:+.2f} / 後 {e['S2']:+.2f}） 獲利/回撤 {e['PM']:5.1f}"
          f"  時框間平均相關 {S[tfs].corr().values[np.triu_indices(len(tfs), 1)].mean():+.2f}")
    P("\n  在 M5+M15 上逐一加入一個時框的邊際效果")
    base_e, _ = ens([S["5min"], S["15min"]], days)
    for tf in TFS:
        if tf in ("5min", "15min"):
            continue
        e, _ = ens([S["5min"], S["15min"], S[tf]], days)
        P(f"    + {tf:<6} Sharpe {base_e['S']:+.2f} → {e['S']:+.2f}  獲利/回撤 {base_e['PM']:.1f} → {e['PM']:.1f}"
          f"  與基準的相關 {S[tf].corr((S['5min'] + S['15min']) / 2):+.2f}")
    txt = "\n".join(lines)
    print(txt)
    open("results_part43.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
