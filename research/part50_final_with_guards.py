"""第五十部分：最終版 + 兩個防護（波動暴增暫停 ATR 比 > 2.2、單日虧損 −15R 停止開新單），RR 3 / 4 完整統計與逐筆明細。
0.02 口，點差 + 0.3 滑價。輸出 final_guarded_trades_rr3.csv / _rr4.csv、results_part50.txt。"""
import numpy as np
import pandas as pd
import part49_safeguards as P

lines = []


def p(s=""):
    print(s, flush=True)
    lines.append(s)


def streak(v):
    b = c = 0
    for x in v:
        c = c + 1 if x < 0 else 0
        b = max(b, c)
    return b


def stats(x, lab):
    pnl = x.net * 2
    yrs = (x.進場時間.max() - x.進場時間.min()).days / 365
    m_min = P.minute_mdd(x)
    e_tr = pnl.groupby(x.出場時間).sum().sort_index().cumsum(); m_tr = (e_tr.cummax() - e_tr).max()
    d = pnl.groupby(x.進場時間.dt.date).sum(); e_d = d.cumsum(); m_d = (e_d.cummax() - e_d).max()
    mon = pnl.groupby(x.進場時間.dt.to_period("M")).sum()
    w, l = pnl[pnl > 0], pnl[pnl <= 0]
    tr = (x.進場時間 < "2025-01-01").values

    def rat(z, pz):
        dz = pz.groupby(z.進場時間.dt.date).sum(); ez = dz.cumsum(); mz = (ez.cummax() - ez).max()
        return pz.sum() / mz
    yr = " / ".join(f"{pnl[(x.進場時間.dt.year == y).values].sum():+,.0f}" for y in (2023, 2024, 2025, 2026))
    p(f"--- {lab} ---")
    p(f"  筆數 {len(x)}（每天 {len(x) / x.進場時間.dt.date.nunique():.1f}），勝率 {(pnl > 0).mean():.1%}，每筆 {pnl.mean():+.2f}，平均賺 {w.mean():+.2f} / 賠 {l.mean():+.2f}，PF {w.sum() / -l.sum():.2f}")
    p(f"  總損益 {pnl.sum():+,.0f}，每年 {pnl.sum() / yrs:+,.0f}，每月平均 {mon.mean():+,.0f}（中位數 {mon.median():+,.0f}），正報酬月 {(mon > 0).sum()}/{len(mon)}")
    p(f"  MDD：含盤中浮虧 {m_min:,.0f} / 逐筆已實現 {m_tr:,.0f} / 日結 {m_d:,.0f}；年獲利÷MDD（含浮虧）{pnl.sum() / yrs / m_min:.2f}、（日結）{pnl.sum() / yrs / m_d:.2f}")
    p(f"  最差一天 {d.min():+,.0f}、最好一天 {d.max():+,.0f}、最差月 {mon.min():+,.0f}（{mon.idxmin()}）、最長連虧 {streak(x.sort_values('出場時間').net.values)} 筆")
    p(f"  訓練 / 驗證比值 {rat(x[tr], pnl[tr]):.1f} / {rat(x[~tr], pnl[~tr]):.1f}；逐年 2023/2024/2025/2026：{yr}")
    p(f"  MDD ≤ 20% 需本金（含浮虧）≥ {m_min / 0.2:,.0f}")
    return mon


p("第五十部分：最終版 + 兩個防護（0.02 口，點差 + 0.3 滑價，2023/8 ~ 2026/10）")
for t in (3, 4):
    x = P.run(t)
    p(f"\n======== RR {t} ========")
    stats(x, "原本")
    g = P.daily_stop(x[~(x.volrel > 2.2).values].reset_index(drop=True), 15)
    p(f"  （波動暴增暫停擋掉 {(x.volrel > 2.2).sum()} 筆、分布在 {x[x.volrel > 2.2].進場時間.dt.date.nunique()} 天；單日 −15R 再擋掉 {(~(x.volrel > 2.2)).sum() - len(g)} 筆）")
    mon = stats(g, "加上兩個防護")
    out = g.assign(口數=0.02, 損益=(g.net * 2).round(2), R=g.R.round(3))[["進場時間", "出場時間", "dir", "entry", "risk", "R", "口數", "損益"]]
    out = out.rename(columns={"dir": "方向(1多-1空)", "entry": "進場價", "risk": "停損距離"})
    out["累積損益"] = out.sort_values("出場時間").損益.cumsum().reindex(out.index).round(2)
    out.to_csv(f"final_guarded_trades_rr{t}.csv", index=False, encoding="utf-8-sig")
    p("  逐月（加防護後）：" + "  ".join(f"{k}:{v:+,.0f}" for k, v in mon.items()))
open("results_part50.txt", "w").write("\n".join(lines) + "\n")
