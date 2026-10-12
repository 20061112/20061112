"""第四十六部分（四）：滾動 IB 版本 A（2023~2026 全資料）× M15 最終版 的組合。
IB：part46_ib_trades.csv（每筆 1 盎司，已扣點差）；M15：final_m15_trades.csv。兩者都再扣每筆 0.3 美元滑價，以 0.02 口計。
指標：日損益相關係數、各自與組合的 總損益 / 日結 MDD / 年獲利÷MDD / 逐年。"""
import numpy as np
import pandas as pd

SLIP = 0.3
I = pd.read_csv("part46_ib_trades.csv", parse_dates=["day", "t_in", "exit_time"])
M = pd.read_csv("final_m15_trades.csv", parse_dates=["進場時間", "出場時間"])
ib = ((I.pnl - SLIP) * 2).groupby(I.day.dt.normalize()).sum()
m15 = ((M.淨利_每盎司 - SLIP) * 2).groupby(M.進場時間.dt.normalize()).sum()
start = max(ib.index.min(), m15.index.min())
idx = pd.date_range(start, max(ib.index.max(), m15.index.max()), freq="D")
D = pd.DataFrame({"IB": ib.reindex(idx, fill_value=0), "M15": m15.reindex(idx, fill_value=0)})
D = D[(D.IB != 0) | (D.M15 != 0) | (D.index.dayofweek < 5)]
lines = ["第四十六部分（四）：滾動 IB 版本 A × M15 最終版（0.02 口，點差 + 0.3 滑價）"]
lines.append(f"IB：{len(I)} 筆（{I.day.min().date()} ~ {I.day.max().date()}），每筆 1 盎司 {I.pnl.mean():+.2f}（扣滑價 {I.pnl.mean() - SLIP:+.2f}），勝率 {(I.pnl > 0).mean():.0%}")
lines.append("IB 逐年（每筆 1 盎司，扣滑價）：" + "；".join(f"{y} {len(g)} 筆 {(g.pnl - SLIP).sum():+,.0f}（每筆 {(g.pnl - SLIP).mean():+.2f}）" for y, g in I.groupby(I.day.dt.year)))
lines.append("  ※ 版本 A 是在 2026 年資料上建立的 → 2023~2025 對它是樣本外")
act = D[(D.IB != 0) & (D.M15 != 0)]
lines.append(f"日損益相關係數（兩者都有交易的日子）：{act.IB.corr(act.M15):+.2f}；全部日子：{D.IB.corr(D.M15):+.2f}")
yrs = (D.index.max() - D.index.min()).days / 365


def st(s, lab):
    e = s.cumsum(); m = (e.cummax() - e).max()
    mon = s.groupby(s.index.to_period("M")).sum()
    yr = " / ".join(f"{s[s.index.year == y].sum():+,.0f}" for y in (2023, 2024, 2025, 2026))
    lines.append(f"  {lab:28s} 總 {s.sum():+8,.0f}  每年 {s.sum() / yrs:+7,.0f}  日結MDD {m:6,.0f}  年獲利/MDD {s.sum() / yrs / m:5.2f}  "
                 f"最差月 {mon.min():+7,.0f}  正月 {(mon > 0).sum()}/{len(mon)} | 逐年 {yr}")


lines.append("\n=== 各自與組合（日結）===")
st(D.IB, "IB 版本 A（0.02 口）")
st(D.M15, "M15 最終版（0.02 口）")
st(D.IB + D.M15, "兩者各 0.02 口")
st((D.IB + D.M15) / 2, "兩者各 0.01 口（總曝險不變）")
for w in (0.25, 0.75):
    st(w * D.IB + (1 - w) * D.M15, f"IB {w:.0%} / M15 {1 - w:.0%}（合計 0.02 口）")

# 只看 2023~2025（IB 的樣本外）
lines.append("\n=== 只看 2023/8 ~ 2025/12（IB 版本 A 的真正樣本外）===")
D2 = D[D.index < "2026-01-01"]
yrs = (D2.index.max() - D2.index.min()).days / 365
lines.append(f"日損益相關係數：{D2.IB.corr(D2.M15):+.2f}")
st(D2.IB, "IB 版本 A（0.02 口）")
st(D2.M15, "M15 最終版（0.02 口）")
st((D2.IB + D2.M15) / 2, "兩者各 0.01 口")
st(0.25 * D2.IB + 0.75 * D2.M15, "IB 25% / M15 75%")
print("\n".join(lines))
open("results_part46_combine.txt", "w").write("\n".join(lines) + "\n")
