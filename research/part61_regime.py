"""第六十一部分：IB 突破核心（L1）與版本 A —— 事前可知的「行情狀態」能否分辨好壞日子 → results_part61.txt

交易：與第六十部分相同的 M1 資料（2024/9 ~ 2026/10），滾動門檻
  L1 = IB 突破 + M5 收盤確認（不做其他篩選，含週一）
  A  = 版本 A
行情狀態（全部只用「前一天收盤以前」的日線，開盤前就知道；日線 = 每個交易日 01:00 ~ 23:59）
  eff20    : 前 20 天平均日效率 |收−開| / (高−低)
  peff     : 前一天的日效率
  vr       : ATR10 / ATR60（日線真實波幅），近期波動相對長期
  trend20  : |前 20 天累積漲跌| / (ATR20 × √20)，近一個月的趨勢強度
  rng10    : 前 10 天平均日波幅（美元）/ 收盤價 × 100（% 波幅）
做法：三分位門檻只用 2024/11 ~ 2025/12（研究期）的日子決定，原封不動套到 2026（驗證期）。
      看「研究期最差的那一組」在 2026 是否也最差；以及排除它之後兩段的 PF。
"""
import os
import subprocess
import sys
import numpy as np
import pandas as pd

DATA = "data/pseudo/XAUUSD_M1_2024_2026.csv"
FEATS = ["eff20", "peff", "vr", "trend20", "rng10"]


def daily_features():
    from load import load
    m = load(DATA)
    day = (m.index - pd.Timedelta(hours=1)).normalize()
    g = m.groupby(day)
    D = pd.DataFrame(dict(o=g.open.first(), h=g.high.max(), l=g.low.min(), c=g.close.last()))
    pc = D.c.shift()
    tr = pd.concat([D.h - D.l, (D.h - pc).abs(), (D.l - pc).abs()], axis=1).max(axis=1)
    eff = (D.c - D.o).abs() / (D.h - D.l)
    F = pd.DataFrame(index=D.index)
    F["eff20"] = eff.rolling(20).mean().shift()
    F["peff"] = eff.shift()
    F["vr"] = (tr.rolling(10).mean() / tr.rolling(60).mean()).shift()
    F["trend20"] = ((D.c - D.c.shift(20)).abs() / (tr.rolling(20).mean() * np.sqrt(20))).shift()
    F["rng10"] = ((D.h - D.l).rolling(10).mean() / D.c * 100).shift()
    return F


def pf(p):
    return p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-9)


def worker():
    from part17_balance import gen, wf
    from final_A import walk_forward, CFG
    L1 = wf(gen("close5", 0.05, (2, 10), 30, True), 1.0, 1.0, 0.0)
    A, _ = walk_forward(gen(**CFG))
    L1.to_pickle("data/pseudo/part61_L1.pkl")
    A.to_pickle("data/pseudo/part61_A.pkl")


def main():
    env = dict(os.environ, XAU_M1_PATH=DATA)
    r = subprocess.run([sys.executable, __file__, "--worker"], env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr)
        raise SystemExit(1)
    F = daily_features()
    out = []
    P = out.append
    P("第六十一部分：事前行情狀態 vs IB 突破（研究期 2024/11~2025/12 定三分位門檻，2026 驗證）")
    for name in ("L1", "A"):
        T = pd.read_pickle(f"data/pseudo/part61_{name}.pkl")
        T = T[T.day >= "2024-11-01"].join(F, on="day")
        T["期間"] = np.where(T.day < "2026-01-01", "研究", "驗證")
        res = T[T.期間 == "研究"]
        P(f"\n===== {name}：研究期 {len(res)} 筆 PF {pf(res.pnl):.2f} 每筆 {res.pnl.mean():+.2f}；"
          f"驗證期 {(T.期間 == '驗證').sum()} 筆 PF {pf(T.pnl[T.期間 == '驗證']):.2f} 每筆 {T.pnl[T.期間 == '驗證'].mean():+.2f} =====")
        days_res = F.loc[(F.index >= "2024-11-01") & (F.index < "2026-01-01")]
        for f in FEATS:
            q = days_res[f].quantile([1 / 3, 2 / 3]).to_numpy()
            T["g"] = pd.cut(T[f], [-np.inf, *q, np.inf], labels=["低", "中", "高"])
            rows = []
            for (per, g), x in T.groupby(["期間", "g"], observed=True):
                rows.append(dict(期間=per, 組=g, 筆數=len(x), PF=round(pf(x.pnl), 2), 每筆=round(x.pnl.mean(), 2),
                                 每筆R=round(x.R.mean(), 3), 總=round(x.pnl.sum())))
            tab = pd.DataFrame(rows).pivot(index="組", columns="期間", values=["筆數", "PF", "每筆R", "總"])
            P(f"\n  {f}（研究期門檻 {q[0]:.3f} / {q[1]:.3f}）")
            P("  " + tab.to_string().replace("\n", "\n  "))
        # 期間平均狀態
    P("\n各期間的行情狀態平均（日）：")
    per = pd.cut(F.index, pd.to_datetime(["2024-11-01", "2025-01-01", "2025-07-01", "2026-01-01", "2026-11-01"]),
                 labels=["2024/11~12", "2025H1", "2025H2", "2026"], right=False)
    P(F.groupby(per, observed=True)[FEATS].mean().round(3).to_string())
    open("results_part61.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker()
        raise SystemExit
    main()
