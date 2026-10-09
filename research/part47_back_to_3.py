"""第四十七部分：回到原本的三組（M5 L1200 / M15 L400 / H1 L100，都是回看 100 小時），固定手數比較。

訊號 / 出場同第 45 部分：OU |z| 穿越 3、順勢市價、停損 1.5 ATR、40 根、週五平倉、週五 20 點後不開新單、點差 + 隔夜。
手數：0.01 手 = 1 盎司。三種倉位規則：
  全部 0.01 手
  vr < 1 → 0.02 手，其餘 0.01 手
  只做 vr < 1（0.01 手），其餘不做
對照：第 45 部分的 16 組（M5/M10/M15/M20 × 70/100/130/200 小時）。
最壞起始點回撤：任何一天開始做，最大的美元回撤（= 全期最大回撤）；需要資金 = 回撤 / 20%。
"""
import numpy as np
import pandas as pd
from load import load
from part39_ou_vs_er import SPLIT
from part43_twoside_vol_tf import TFrame, signals
from part45_final_ou import trades

SETS = {
    "原本三組 M5/M15/H1（100h）": [("5min", 5, 1200), ("15min", 15, 400), ("1h", 60, 100)],
    "只有 M5 + M15（100h）": [("5min", 5, 1200), ("15min", 15, 400)],
    "16 組（第 45 部分）": [(tf, m, int(h * 60 / m)) for tf, m in (("5min", 5), ("10min", 10), ("15min", 15), ("20min", 20))
                         for h in (70, 100, 130, 200)],
}
RULES = {"全部 0.01 手": lambda v: np.ones_like(v), "vr<1 0.02 手": lambda v: np.where(v < 1, 2, 1),
         "只做 vr<1": lambda v: np.where(v < 1, 1, 0)}


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    days = sorted(set(m1.index.date))
    first = [d for d in days if pd.Timestamp(d) < SPLIT]
    frames = {}
    out = []
    P = out.append
    P(f"{'設定':<24}{'倉位':<12}{'筆數':>5}{'每天':>6}{'勝率':>6}{'PF':>6}{'總損益$':>9}{'最大回撤$':>9}{'獲利/回撤':>9}"
      f"{'Sharpe':>8}{'前半':>7}{'後半':>7}{'最差一天$':>10}{'最好5天占':>9}{'需要資金$':>10}")
    keep = {}
    for sn, legs in SETS.items():
        rows = []
        for tf, mins, L in legs:
            F = frames.setdefault(tf, TFrame(m1, tf, mins))
            rows += trades(F, *signals(F, L), f"{tf}|{L}")
        T = pd.DataFrame(rows).sort_values("entry").reset_index(drop=True)
        T["xday"] = (T.exit - pd.Timedelta(minutes=1)).dt.date
        for rn, f in RULES.items():
            oz = f(T.vr.to_numpy())
            S = T[oz > 0].assign(oz=oz[oz > 0])
            S["pnl"] = S.usd * S.oz
            d = pd.Series(S.pnl.to_numpy(), index=S.xday).groupby(level=0).sum().reindex(days, fill_value=0.0)
            eq = d.cumsum().to_numpy()
            mdd = -(eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
            sh = lambda s: s.mean() / s.std() * np.sqrt(252)
            top5 = d.sort_values(ascending=False).head(5).sum() / d.sum()
            P(f"{sn:<24}{rn:<12}{len(S):>5}{len(S) / len(days):>6.2f}{np.mean(S.pnl > 0):>6.0%}"
              f"{S.pnl[S.pnl > 0].sum() / -S.pnl[S.pnl < 0].sum():>6.2f}{d.sum():>+9,.0f}{mdd:>9,.0f}{d.sum() / mdd:>9.1f}"
              f"{sh(d):>+8.2f}{sh(d.loc[first]):>+7.2f}{sh(d.drop(first)):>+7.2f}{d.min():>+10,.0f}{top5:>9.0%}{mdd / 0.2:>10,.0f}")
            keep[(sn, rn)] = (S, d)
        P("")
    for key in (("原本三組 M5/M15/H1（100h）", "全部 0.01 手"), ("原本三組 M5/M15/H1（100h）", "只做 vr<1"),
                ("原本三組 M5/M15/H1（100h）", "vr<1 0.02 手")):
        S, d = keep[key]
        P(f"\n【{key[0]} / {key[1]}】")
        S = S.assign(月=S.exit.dt.strftime("%m"))
        P("  逐月 $：" + "  ".join(f"{k}月 {v:+,.0f}" for k, v in S.groupby("月").pnl.sum().items()))
        P("  依時框：" + "  ".join(f"{tf} {len(s)} 筆 {s.pnl.sum():+,.0f}$（每筆 {s.pnl.mean():+.1f}）"
                                  for tf, s in S.groupby("tf")))
        P(f"  單筆最大 +{S.pnl.max():.0f} / {S.pnl.min():.0f}$；停損中位 ${(S.risk_usd * S.oz).median():.1f}；"
          f"持有中位 {S.hours.median():.1f}h；拿掉最好 10 天 {d.sum() - d.sort_values(ascending=False).head(10).sum():+,.0f}$")
    txt = "\n".join(out)
    print(txt)
    open("results_part47.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
