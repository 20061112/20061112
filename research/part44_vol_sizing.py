"""第四十四部分：16 組集成（M5/M10/M15/M20 × 回看 70/100/130/200 小時）的交易數，以及「波動低時加大倉位」。

訊號 / 出場同第 43 部分（OU |z| 穿越 3、順勢市價、停損 1.5 ATR、40 根、週五平倉、點差 + 隔夜）。
波動比 vr = 訊號當下 ATR14 / ATR200（同週期）。倉位倍數規則：
  固定 1 倍（基準）
  vr < 1 → ×1.5 / ×2 / ×3，其餘 ×1
  vr < 1 → ×2，vr ≥ 1.32 → ×0.5
  vr < 1 → ×2，vr ≥ 1.32 → 不做
  連續：倍數 = 1 / vr（限制在 0.5 ~ 3）
  只做 vr < 1（其餘不做）
每組權重 1/16；報酬單位 = 「每組 1 倍時每筆 1R」。另外算同時持倉數與單日最大總風險（倍數合計 / 16）。
"""
import numpy as np
import pandas as pd
from load import load
from part39_ou_vs_er import daily, SPLIT
from part43_twoside_vol_tf import TFrame, signals, run, sharpe, HOURS

TFS = {"5min": 5, "10min": 10, "15min": 15, "20min": 20}
RULES = {
    "固定 1 倍（基準）": lambda v: np.ones_like(v),
    "vr<1 ×1.5": lambda v: np.where(v < 1, 1.5, 1.0),
    "vr<1 ×2": lambda v: np.where(v < 1, 2.0, 1.0),
    "vr<1 ×3": lambda v: np.where(v < 1, 3.0, 1.0),
    "vr<1 ×2、vr≥1.32 ×0.5": lambda v: np.where(v < 1, 2.0, np.where(v >= 1.32, 0.5, 1.0)),
    "vr<1 ×2、vr≥1.32 不做": lambda v: np.where(v < 1, 2.0, np.where(v >= 1.32, 0.0, 1.0)),
    "連續 1/vr（0.5~3）": lambda v: np.clip(1 / v, 0.5, 3.0),
    "只做 vr<1": lambda v: np.where(v < 1, 1.0, 0.0),
}


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    days = sorted(set(m1.index.date))
    nd = len(days)
    lines = []
    P = lines.append
    books = []
    for tf, mins in TFS.items():
        F = TFrame(m1, tf, mins)
        for h in HOURS:
            T = run(F, *signals(F, int(h * 60 / mins)), "trend")
            T["cfg"], T["tf"], T["hrs"] = f"{tf}|{h}", tf, h
            idx = F.idx
            T["entry"] = idx[idx.get_indexer(T.time) + 1]
            books.append(T)
    A = pd.concat(books).sort_values("entry").reset_index(drop=True)
    A.to_csv("part44_trades.csv", index=False)

    P("=" * 120)
    P(f"1. 交易數（{nd} 個交易日）")
    P(f"   16 組合計 {len(A)} 筆 = 每天 {len(A) / nd:.2f} 筆（每筆只用 1/16 部位）")
    P("   各時框：" + "  ".join(f"{tf} {(A.tf == tf).sum() / nd:.2f}/天" for tf in TFS))
    P("   各回看：" + "  ".join(f"{h}h {(A.hrs == h).sum() / nd:.2f}/天" for h in HOURS))
    # 事件：同方向、進場時間相差 2 小時內視為同一波
    ev, last_t, last_d = 0, None, None
    for r in A.itertuples():
        if last_t is None or r.side != last_d or (r.entry - last_t) > pd.Timedelta(hours=2):
            ev += 1
        last_t, last_d = r.entry, r.side
    P(f"   合併成「同方向、2 小時內」的獨立事件：{ev} 次 = 每天 {ev / nd:.2f} 次；平均每次事件有 {len(A) / ev:.1f} 組同時進場")
    P(f"   有交易的日子 {A.xtime.dt.date.nunique()} / {nd} 天")
    # 同時持倉
    tl = pd.concat([pd.Series(1, index=A.entry), pd.Series(-1, index=A.xtime + pd.Timedelta(minutes=1))]).sort_index().cumsum()
    P(f"   同時持倉組數：最多 {int(tl.max())} 組，有持倉時的中位數 {int(tl[tl > 0].median())} 組")

    P("\n" + "=" * 120)
    P("2. 波動低時加大倉位（每組權重 1/16；總 R = 16 組平均後的累積）")
    P(f"   訊號 vr 分佈：<1 佔 {np.mean(A.vr < 1):.0%}、1~1.32 佔 {np.mean((A.vr >= 1) & (A.vr < 1.32)):.0%}、≥1.32 佔 {np.mean(A.vr >= 1.32):.0%}")
    P(f"   每筆 R：vr<1 {A[A.vr < 1].R.mean():+.3f}  1~1.32 {A[(A.vr >= 1) & (A.vr < 1.32)].R.mean():+.3f}  ≥1.32 {A[A.vr >= 1.32].R.mean():+.3f}")
    first = [d for d in days if pd.Timestamp(d) < SPLIT]
    P(f"   {'規則':<24}{'平均倍數':>8}{'總R':>8}{'最大回撤':>8}{'獲利/回撤':>9}{'Sharpe':>8}{'前半':>7}{'後半':>7}"
      f"{'最差一天':>9}{'最大同時風險':>12}")
    for lab, f in RULES.items():
        w = f(A.vr.to_numpy())
        B = A.assign(R=A.R * w / 16, w=w)
        s = daily(B[B.w > 0], days)
        eq = s.cumsum().to_numpy()
        mdd = -(eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
        live = B[B.w > 0]
        tl = pd.concat([pd.Series(live.w.to_numpy() / 16, index=live.entry),
                        pd.Series(-live.w.to_numpy() / 16, index=live.xtime + pd.Timedelta(minutes=1))]).sort_index().cumsum()
        P(f"   {lab:<24}{w[w > 0].mean():>8.2f}{s.sum():>+8.1f}{mdd:>8.1f}{s.sum() / mdd:>9.1f}{sharpe(s):>+8.2f}"
          f"{sharpe(s.loc[first]):>+7.2f}{sharpe(s.drop(first)):>+7.2f}{s.min():>+9.2f}{tl.max():>12.2f}")
    P("   （最大同時風險 = 同一時間所有未平倉單的倍數合計 / 16，單位 R；1.0 = 同時虧完等於虧 1 個「全倉 R」）")
    txt = "\n".join(lines)
    print(txt)
    open("results_part44.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
