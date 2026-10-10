"""第四十九部分：真正的樣本外驗證 — 規則完全不改，跑 2023~2025。

資料：data/XAUUSD_M15_2023_2026.csv（2023/1/3 ~ 2026/10/9）、data/XAUUSD_M5_2025_2026.csv（2025/5/13 ~ 2026/10/9）。
策略規則都是只用 2026 年資料決定的（第 35~48 部分），所以 2023/1 ~ 2025/12 全部是樣本外。
規則（第 48 部分 8 組版）：M5 / M15 × 回看 70/100/130/200 小時；OU |z| 穿越 3 → 下一根開盤順勢市價；停損 1.5 ATR14；
  持有 40 根 / 週五最後一根平倉；週五 20 點後不開新單；每組最多 1 筆；vr = ATR14/ATR200 < 1 → 0.02 手，否則 0.01 手；
  點差（該根 K 棒的點差欄位）+ 隔夜 $0.7/盎司/晚（週三 ×3）。
另外列出：只有 M5/M15 100h 兩組的簡單版；以及每筆 1R（不受停損美元大小影響）的結果。
"""
import numpy as np
import pandas as pd
from load import load
from part43_twoside_vol_tf import TFrame, signals
from part45_final_ou import trades, streak

LEGS = [(tf, m, h) for tf, m in (("5min", 5), ("15min", 15)) for h in (70, 100, 130, 200)]
FILES = {"5min": "data/XAUUSD_M5_2025_2026.csv", "15min": "data/XAUUSD_M15_2023_2026.csv"}


def stats(S, days, col="pnl"):
    if len(S) == 0:
        return None
    d = pd.Series(S[col].to_numpy(), index=S.xday).groupby(level=0).sum().reindex(days, fill_value=0.0)
    eq = d.cumsum().to_numpy()
    mdd = -(eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
    top = d.sort_values(ascending=False)
    return dict(n=len(S), per_day=len(S) / len(days), win=np.mean(S[col] > 0),
                pf=S[col][S[col] > 0].sum() / -S[col][S[col] < 0].sum() if (S[col] < 0).any() else np.inf,
                tot=d.sum(), mdd=mdd, pm=d.sum() / mdd if mdd > 0 else np.nan,
                sh=d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else np.nan,
                top5=top.head(5).sum() / d.sum() if d.sum() != 0 else np.nan,
                ex10=d.sum() - top.head(10).sum(), d=d)


def line(lab, s, unit="$"):
    if s is None:
        return f"  {lab:<30} 無交易"
    f = (lambda v: f"{v:+,.0f}") if unit == "$" else (lambda v: f"{v:+.1f}R")
    g = (lambda v: f"{v:,.0f}") if unit == "$" else (lambda v: f"{v:.1f}R")
    return (f"  {lab:<30} {s['n']:4d} 筆（{s['per_day']:.2f}/天） 勝率 {s['win']:.0%} PF {s['pf']:.2f} 總 {f(s['tot']):>8} "
            f"回撤 {g(s['mdd']):>6} 獲利/回撤 {s['pm']:5.1f} Sharpe {s['sh']:+.2f} 最好5天占 {s['top5']:.0%} "
            f"拿掉最好10天 {f(s['ex10'])}")


def main():
    data = {tf: load(p) for tf, p in FILES.items()}
    frames = {tf: TFrame(data[tf], tf, m) for tf, m in (("5min", 5), ("15min", 15))}
    rows = []
    for tf, mins, h in LEGS:
        rows += trades(frames[tf], *signals(frames[tf], int(h * 60 / mins)), f"{tf}|{h}h")
    T = pd.DataFrame(rows).sort_values("entry").reset_index(drop=True)
    T["oz"] = np.where(T.vr < 1, 2, 1)
    T["pnl"] = T.usd * T.oz
    T["R"] = T.usd / T.risk_usd
    T["Rw"] = T.R * T.oz / 8
    T["xday"] = (T.exit - pd.Timedelta(minutes=1)).dt.date
    T["year"] = T.exit.dt.year
    T["h"] = T.cfg.str.split("|").str[1]
    T.to_csv("part49_trades.csv", index=False)
    alldays = sorted(set(data["15min"].index.date))
    out = []
    P = out.append
    P("=" * 130)
    P("第四十九部分：樣本外驗證（規則完全沿用第 48 部分，2023~2025 為樣本外，2026 為樣本內）")
    P("=" * 130)

    periods = [("2023（樣本外）", "2023-01-01", "2024-01-01"), ("2024（樣本外）", "2024-01-01", "2025-01-01"),
               ("2025（樣本外）", "2025-01-01", "2026-01-01"), ("2025/5/13~12/31（樣本外，M5 也有）", "2025-05-13", "2026-01-01"),
               ("2026（樣本內）", "2026-01-01", "2026-10-10"), ("2023~2025 全部樣本外", "2023-01-01", "2026-01-01")]
    for lab, a, b in periods:
        days = [d for d in alldays if pd.Timestamp(a) <= pd.Timestamp(d) < pd.Timestamp(b)]
        S = T[(T.exit >= a) & (T.exit < b)]
        P(f"\n【{lab}】 {len(days)} 個交易日")
        P(line("8 組（0.01/0.02 手）", stats(S, days)))
        P(line("  只有 M15 4 組", stats(S[S.tf == "15min"], days)))
        if (S.tf == "5min").any():
            P(line("  只有 M5 4 組", stats(S[S.tf == "5min"], days)))
        P(line("簡單版 M5/M15 100h", stats(S[S.h == "100h"], days)))
        P(line("8 組 每筆 1R（R 單位，1/8 權重）", stats(S, days, "Rw"), "R"))
        if len(S):
            P(f"  vr<1：{(S.vr < 1).mean():.0%} 的單，每筆 {S[S.vr < 1].R.mean():+.2f}R；vr≥1 每筆 {S[S.vr >= 1].R.mean():+.2f}R；"
              f"順勢方向 多 {(S.side == '多').mean():.0%}")

    P("\n" + "=" * 130)
    P("M15 4 組（唯一有 2023~2026 完整資料的部分）逐年 / 逐月")
    M = T[T.tf == "15min"]
    for y, s in M.groupby("year"):
        days = [d for d in alldays if pd.Timestamp(d).year == y]
        st = stats(s, days)
        P(line(f"{y}", st))
    mo = M.groupby(M.exit.dt.strftime("%Y-%m")).pnl.sum()
    P("\n  逐月 $（M15 4 組）")
    for y in sorted(set(i[:4] for i in mo.index)):
        P(f"   {y}: " + "  ".join(f"{k[5:]}:{v:+.0f}" for k, v in mo.items() if k.startswith(y)))
    P(f"  正報酬月份：{np.mean(mo > 0):.0%}（{int((mo > 0).sum())}/{len(mo)}）")
    P("\n  依回看（M15，2023~2025 樣本外，每筆 $ / 總 $）")
    O = M[M.exit < "2026-01-01"]
    P("   " + "  ".join(f"{h} {len(s)} 筆 {s.pnl.mean():+.1f} / {s.pnl.sum():+,.0f}" for h, s in O.groupby("h")))
    P("\n  依波動比（M15，2023~2025 樣本外，每筆 R）")
    P(f"   vr<1 {O[O.vr < 1].R.mean():+.3f}R（{len(O[O.vr < 1])} 筆）  vr≥1 {O[O.vr >= 1].R.mean():+.3f}R（{len(O[O.vr >= 1])} 筆）")
    P("\n  2023~2025 樣本外最好 10 天：" + "、".join(
        f"{k} {v:+.0f}" for k, v in stats(O, [d for d in alldays if pd.Timestamp(d) < pd.Timestamp('2026-01-01')])['d']
        .sort_values(ascending=False).head(10).items()))
    txt = "\n".join(out)
    print(txt)
    open("results_part49.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
