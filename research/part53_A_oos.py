"""第五十三部分：版本 A 的真正樣本外驗證（2025 年 M5 資料）。

版本 A 只用到：60 分 IB 的高低、M5 收盤確認、15 分鐘 TPO 字母（每個字母的高低）、日線 ATR10、M1/M5 高低判斷停損、當天最後收盤出場。
這些全部可以從 M5 精確還原，所以把每根 M5 展開成 5 根「偽 M1」（OHLC 都等於那根 M5），原封不動跑 final_A 的程式。
  - M5 收盤確認：只看每 5 分鐘最後一根（= M5 收盤），完全相同
  - 停損：用之後每根 M5 的高低判斷（M1 版本也是同一段價格，只是粒度較細）
  - 規則、門檻（每月只用過去資料滾動重算）都沒有改

步驟
  0. 原始 2026 M1（基準，應重現 FINAL_STRATEGY_A：735 筆、+9,326）
  1. 2026 M1 → 合成 M5 → 偽 M1：檢查 M5 近似誤差
  2. 2025/5/13 ~ 2026/10/9 的 M5 檔 → 偽 M1：2025/7 ~ 12 是開發時完全沒看過的資料
執行：python3 part53_A_oos.py（會在子程序中設定 XAU_M1_PATH 重跑）
"""
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from load import load

OUT_DIR = "data/pseudo"
RES = "results_part53.txt"


def write_tab(df, path):
    o = pd.DataFrame({"<DATE>": df.index.strftime("%Y.%m.%d"), "<TIME>": df.index.strftime("%H:%M:%S"),
                      "<OPEN>": df.open.round(3), "<HIGH>": df.high.round(3), "<LOW>": df.low.round(3), "<CLOSE>": df.close.round(3),
                      "<TICKVOL>": df.tickvol.astype(int), "<VOL>": 0, "<SPREAD>": df.spread.astype(int)})
    o.to_csv(path, sep="\t", index=False)


def m1_to_m5(d):
    g = d.resample("5min", label="left", closed="left")
    m5 = pd.DataFrame(dict(open=g.open.first(), high=g.high.max(), low=g.low.min(), close=g.close.last(),
                           tickvol=g.tickvol.sum(), spread=g.spread.max()))
    return m5.dropna()


def expand(m5):
    """每根 M5 → 5 根偽 M1（時間 T、T+1、…、T+4），OHLC 都等於那根 M5。"""
    rep = m5.loc[m5.index.repeat(5)].copy()
    rep.index = rep.index + pd.to_timedelta(np.tile(np.arange(5), len(m5)), unit="min")
    rep["tickvol"] = (rep.tickvol / 5).round()
    return rep


def run_one(path, label):
    env = dict(os.environ)
    if path:
        env["XAU_M1_PATH"] = path
    r = subprocess.run([sys.executable, __file__, "--worker", label], env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr); raise SystemExit(1)
    return r.stdout


# ------------------------------------------------------------------ worker：在指定資料上跑版本 A
def worker(label):
    from part14_deep import DAYS
    from part17_balance import gen
    from final_A import walk_forward, CFG
    X = gen(**CFG)
    T, used = walk_forward(X)
    if label.startswith("2_"): T.to_csv("part53_trades_2025_2026.csv", index=False)

    def block(S, lab):
        if not len(S):
            return f"  {lab:22s} 無交易"
        p = S.pnl; w, l = p[p > 0], p[p <= 0]
        eq = S.sort_values("exit_time").pnl.cumsum(); dd = (eq - eq.cummax()).min()
        days = [d for d in DAYS if S.day.min() <= d <= S.day.max() and d.dayofweek != 0]
        daily = pd.Series(0.0, index=days).add(S.groupby("day").pnl.sum(), fill_value=0)
        lose = (p <= 0).astype(int); streak = lose.groupby((lose != lose.shift()).cumsum()).sum().max()
        top5 = S.groupby("day").pnl.sum().nlargest(5).sum() / p.sum() if p.sum() > 0 else np.nan
        return (f"  {lab:22s} {len(S):4d}筆 {len(S) / max(len(days), 1):4.1f}筆/天 勝率 {len(w) / len(p):5.1%} 均賺 {w.mean():+6.1f} 均賠 {l.mean():+6.1f} "
                f"每筆 {p.mean():+6.2f}（{S.R.mean():+.2f}R） PF {w.sum() / max(-l.sum(), 1e-9):.2f} 總 {p.sum():+7.0f} 回撤 {dd:+6.0f} "
                f"日Sharpe {daily.mean() / daily.std() * np.sqrt(250):+.2f} 連虧 {streak} 最好5天占 {top5:.0%}")

    lines = [f"[{label}] 期間 {T.day.min().date()} ~ {T.day.max().date()}", block(T, "全部")]
    for y, g in T.groupby(T.day.dt.year):
        lines.append(block(g, f"{y} 年"))
    lines.append("  逐月（每筆 1 盎司，美元）：")
    for m, g in T.groupby(T.day.dt.to_period("M")):
        p = g.pnl
        lines.append(f"    {m}  {len(g):3d}筆  總 {p.sum():+7.0f}  PF {p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-9):5.2f}  勝率 {(p > 0).mean():5.1%}")
    lines.append("  每月門檻（滾動）：" + "  ".join(f"{r.month}: ib {r.thr_ib:.3f} tail {r.thr_tail:.3f} ext {r.thr_ext:.2f}" for r in used.itertuples()))
    print("\n".join(lines))


def ladder():
    """複雜度階梯（同第二十一部分 C）：每一步只多加一條規則，逐年列出。"""
    from part17_balance import gen, wf
    steps = [("L0 只有 IB 突破（觸價、停損另一側、收盤出）", ("touch", 0.0, True), (1.0, 1.0, 0.0)),
             ("L1 + M5 收盤確認", ("close5", 0.05, True), (1.0, 1.0, 0.0)),
             ("L2 + 不做週一", ("close5", 0.05, False), (1.0, 1.0, 0.0)),
             ("L3 + 小 IB 篩選", ("close5", 0.05, False), (1.0, 1.0, 0.4)),
             ("L5 = 版本 A（+ tail + ext）", ("close5", 0.05, False), (0.9, 0.9, 0.4))]
    print("  複雜度階梯（2025/5~2026/10 M5 資料，滾動門檻）：")
    for lab, (cf, bf, mon), (qt, qe, qi) in steps:
        Y = wf(gen(cf, bf, (2, 10), 30, mon), qt, qe, qi)
        cells = []
        for y, g in Y.groupby(Y.day.dt.year):
            p = g.pnl
            cells.append(f"{y}: {len(g)}筆 每筆 {p.mean():+6.2f}（{g.R.mean():+.2f}R） PF {p[p > 0].sum() / -p[p <= 0].sum():.2f} 總 {p.sum():+6.0f}")
        print(f"  {lab:32s} " + " | ".join(cells))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        ladder() if sys.argv[2] == "ladder" else worker(sys.argv[2]); raise SystemExit
    os.makedirs(OUT_DIR, exist_ok=True)
    m1 = load("data/XAUUSD_M1_2026.csv")
    p1 = f"{OUT_DIR}/pseudo_from_m1_2026.csv"; write_tab(expand(m1_to_m5(m1)), p1)
    m5 = load("data/XAUUSD_M5_2025_2026.csv")
    p2 = f"{OUT_DIR}/pseudo_from_m5_2025_2026.csv"; write_tab(expand(m5), p2)
    out = open(RES, "w")
    def P(s=""): print(s); out.write(s + "\n"); out.flush()
    P("第五十三部分：版本 A 的真正樣本外驗證（規則、門檻算法完全不變；單位 美元/盎司 = 0.01 手美元；每筆扣進場 K 棒點差）")
    P(run_one(None, "0_M1_2026原始"))
    P(run_one(p1, "1_M1轉M5_2026"))
    P(run_one(p2, "2_M5_2025_2026"))
    P(run_one(p2, "ladder"))
    out.close()
