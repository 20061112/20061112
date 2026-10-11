"""第五十三部分：用 2024/9 ~ 2026/2 的 M1 資料研究（data/XAUUSD_M1_2024_2026.csv）。

事先固定三個問題，參數都不改：
A. M5 很陡順勢（49g）：M5 L48 / 96 / 144 / 192，SMA / EWMA × W 50 ~ 1000 × 門檻 −1 / −2（每 L 24 格）。
   49g 在 2026 的結論：L48、L192 全部格子為正；L96、L144 大多為負。樣本外 = 2024/9 ~ 2025/12。
B. M1 L720 很陡順勢（49k2，M15 ATR 停損）：W 5000 / 15000 × SMA / EWMA × 門檻 −2。樣本外 = 2024/9 ~ 2025/12。
C. M15 規格書策略（STEEP_SPEC）用 M1 改進進場：訊號照舊（M15、L48、EWMA1000 z(cosθ) ≤ −2），
   出場時間照舊（訊號後 12 小時），停損距離照舊（1.5 × 訊號根 M15 ATR）。進場方式：
     V0  下一根 M15 開盤（= 規格書；在 M1 路徑上重算，檢查和 M15 回測一致）
     P25 / P50  掛限價：訊號收盤價往回 0.25 / 0.5 × ATR，60 分鐘內成交才進，沒成交就放棄
     P25m / P50m 同上，但 60 分鐘沒成交就在第 60 分鐘收盤市價進
   停損 = 成交價 − 方向 × 1.5 × ATR（距離不變，R 可比）。成本 = 成交那分鐘的點差。
   期間分兩半：2024/9 ~ 2025/6、2025/7 ~ 2026/2。
"""
import numpy as np
import pandas as pd
from load import load
from part49_friction import resample, features, atr, tst
from part49e_mua_detail import backtest
from part49g_zwin_m5 import z_sma, z_ewm
from part49k2_m1_stop import m15_atr_on_m1, feats as feats_m1

PATH = "data/XAUUSD_M1_2024_2026.csv"
OOS_END = pd.Timestamp("2026-01-01")


def per(T, a=None, b=None):
    m = np.ones(len(T), bool)
    if a is not None:
        m &= (T.entry_time >= a).to_numpy()
    if b is not None:
        m &= (T.entry_time < b).to_numpy()
    return T[m]


def line(T, base=None):
    if len(T) < 10:
        return f"n={len(T):4d}"
    m, t, _ = tst(T.R.to_numpy())
    p = T.pnl
    ex = f" 超{m - base:+.3f}" if base is not None else ""
    return f"n={len(T):4d} {m:+.3f}R(t={t:+.1f}){ex} PF{p[p > 0].sum() / -p[p < 0].sum():.2f} 總{p.sum():+6.0f}"


def part_a(m1, out):
    out.append("===== A. M5 很陡順勢（樣本外 2024/9 ~ 2025/12）=====")
    df = resample(m1, "5min")
    rows = []
    for L in (48, 96, 144, 192):
        F = features(df, L)
        valid = F.a.notna() & F.atr.notna() & (F.dir.fillna(0) != 0)
        B = per(backtest(df, F, valid, L), b=OOS_END)
        base = B.R.mean()
        for zn, zf in (("SMA", z_sma), ("EWMA", z_ewm)):
            for w in (50, 100, 150, 250, 500, 1000):
                z = zf(F.cos, w)
                for thr in (-1.0, -2.0):
                    T = per(backtest(df, F, valid & (z <= thr), L), b=OOS_END)
                    p = T.pnl
                    rows.append(dict(L=L, z=zn, W=w, thr=thr, n=len(T), R=T.R.mean(), ex=T.R.mean() - base,
                                     pf=p[p > 0].sum() / -p[p < 0].sum(), base=base))
    S = pd.DataFrame(rows)
    S.to_csv("part53_a_m5.csv", index=False, float_format="%.4f")
    g = S.groupby("L").agg(基準R=("base", "first"), 超額正比例=("ex", lambda v: (v > 0).mean()), 平均超額=("ex", "mean"),
                           最差=("ex", "min"), 最好=("ex", "max"), 平均R=("R", "mean"), PF大於1=("pf", lambda v: (v > 1).mean()),
                           平均筆數=("n", "mean"))
    out.append(g.round(3).to_string())
    out.append("（49g 在 2026：L48、L192 超額正比例 100%；L96 33%、L144 8% ~ 67%）")
    out.append("")


def part_b(m1, out):
    out.append("===== B. M1 L720 很陡順勢，M15 ATR 停損（樣本外 2024/9 ~ 2025/12）=====")
    A = m15_atr_on_m1(m1)
    F = feats_m1(m1, 720, A)
    valid = F.atr.notna() & (F.dir.fillna(0) != 0) & F.cos.notna()
    B = backtest(m1, F, valid, 720)
    bo = per(B, b=OOS_END)
    out.append(f"  基準 全部順勢  樣本外 {line(bo)} | 2026 {line(per(B, a=OOS_END))}")
    for zn, zf in (("EWMA", z_ewm), ("SMA ", z_sma)):
        for w in (5000, 15000):
            T = backtest(m1, F, valid & (zf(F.cos, w) <= -2), 720)
            out.append(f"  {zn} W={w:5d} ≤−2  樣本外 {line(per(T, b=OOS_END), bo.R.mean())} | 2026 {line(per(T, a=OOS_END))}")
    out.append("（49k2 在 2026：這 4 格每筆 +3.2 ~ +5.0 美元、PF 1.32 ~ 1.52）")
    out.append("")


def simulate_m1(m1, sig, mode, k=0.0, wait=60):
    """在 M1 路徑上模擬 M15 訊號的交易。sig：signal_time（M15 根開始時間）、close、atr、dir。"""
    idx = m1.index
    o, h, l, c = (m1[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    spr = m1.spread.to_numpy(float) * 0.01
    rows, busy_until = [], pd.Timestamp.min
    for s in sig.itertuples():
        t_close = s.signal_time + pd.Timedelta("15min")            # 訊號根收盤 = 下一根開始
        if t_close < busy_until:
            continue
        t_exit = s.signal_time + pd.Timedelta(hours=12) + pd.Timedelta("15min")   # 第 48 根收盤
        i0 = idx.searchsorted(t_close)
        iend = idx.searchsorted(t_exit) - 1                         # 最後一根 M1（收盤 = t_exit）
        if i0 >= len(idx) or iend <= i0:
            continue
        d, A = s.dir, s.atr
        fill_i, fill = None, None
        if mode == "open":
            fill_i, fill = i0, o[i0]
        else:
            lim = s.close - d * k * A
            iw = min(idx.searchsorted(t_close + pd.Timedelta(minutes=wait)) - 1, iend - 1)
            for j in range(i0, iw + 1):
                if (l[j] <= lim) if d == 1 else (h[j] >= lim):
                    fill_i, fill = j, (min(o[j], lim) if d == 1 else max(o[j], lim))
                    break
            if fill_i is None:
                if mode == "limit":
                    continue
                fill_i, fill = iw + 1, o[iw + 1]                    # 60 分鐘後市價
        risk = 1.5 * A
        stop = fill - d * risk
        x, why, out_i = None, "時間", iend
        for j in range(fill_i, iend + 1):
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                x, why, out_i = stop, "停損", j
                break
        if x is None:
            x = c[iend]
        pnl = (x - fill) * d - spr[fill_i]
        busy_until = idx[out_i] + pd.Timedelta("1min")
        rows.append(dict(signal_time=s.signal_time, entry_time=idx[fill_i], exit_time=idx[out_i], dir=d, entry=fill,
                         exit=x, risk=risk, exit_reason=why, pnl=pnl, R=pnl / risk,
                         delay_min=(idx[fill_i] - t_close).total_seconds() / 60))
    return pd.DataFrame(rows)


def part_c(m1, out):
    out.append("===== C. M15 規格書策略，用 M1 改進進場 =====")
    m15 = load("data/XAUUSD_M15_2023_2026.csv")
    F = features(m15, 48)
    z = z_ewm(F.cos, 1000)
    valid = F.a.notna() & F.atr.notna() & (F.dir.fillna(0) != 0)
    s = (valid & (z <= -2))
    sig = pd.DataFrame({"signal_time": F.index[s], "close": m15.close[s].to_numpy(), "atr": F.atr[s].to_numpy(),
                        "dir": F.dir[s].astype(int).to_numpy()})
    sig = sig[(sig.signal_time >= m1.index.min() + pd.Timedelta(hours=1)) & (sig.signal_time < m1.index.max() - pd.Timedelta(hours=13))]
    # M15 原版（同期間）對照
    T15 = backtest(m15, F, s, 48)
    T15 = T15[(T15.signal_time >= sig.signal_time.min()) & (T15.signal_time <= sig.signal_time.max())]
    H = pd.Timestamp("2025-07-01")
    out.append(f"  M15 回測（規格書）           全部 {line(T15)} | 前半 {line(per(T15, b=H))} | 後半 {line(per(T15, a=H))}")
    res = {}
    for name, mode, k in (("V0 下一根開盤", "open", 0), ("P25 回檔0.25ATR 限價", "limit", 0.25),
                          ("P50 回檔0.5ATR 限價", "limit", 0.5), ("P25m 沒成交改市價", "limit_mkt", 0.25),
                          ("P50m 沒成交改市價", "limit_mkt", 0.5)):
        T = simulate_m1(m1, sig, mode, k)
        res[name] = T
        extra = ""
        if mode != "open":
            extra = f" | 平均等待 {T.delay_min.mean():.0f} 分鐘"
        out.append(f"  {name:20s} 全部 {line(T)} | 前半 {line(per(T, b=H))} | 後半 {line(per(T, a=H))}{extra}")
    v0 = res["V0 下一根開盤"]
    out.append(f"  訊號數 {len(sig)}；V0 筆數 {len(v0)}，M15 回測筆數 {len(T15)}")
    for name in ("P25 回檔0.25ATR 限價", "P50 回檔0.5ATR 限價"):
        T = res[name]
        miss = v0[~v0.signal_time.isin(T.signal_time)]
        hit = v0[v0.signal_time.isin(T.signal_time)]
        out.append(f"  {name}：成交 {len(T)} 筆；V0 中「沒等到回檔」的單 {len(miss)} 筆 每筆 {miss.R.mean():+.3f}R、"
                   f"「有回檔」的單 {len(hit)} 筆 每筆 {hit.R.mean():+.3f}R")
    v0.to_csv("part53_c_v0_trades.csv", index=False, float_format="%.4f")
    out.append("")


def main():
    m1 = load(PATH)
    out = ["第五十三部分：M1 資料研究（2024/9 ~ 2026/2）", f"資料 {m1.index.min()} ~ {m1.index.max()}，{len(m1)} 根 M1", ""]
    part_a(m1, out)
    part_b(m1, out)
    part_c(m1, out)
    txt = "\n".join(out)
    print(txt)
    with open("results_part53.txt", "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
