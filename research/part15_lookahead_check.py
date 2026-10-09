"""未來函數檢查。
1. 截斷測試：隨機抽交易，把當天資料砍到「進場那根 K 棒為止」，未來的 K 棒全部刪掉，
   重新偵測訊號、重算特徵，必須和完整資料算出的一模一樣。
2. 延後進場：改成下一根 K 棒開盤進場，看績效是否崩掉（崩掉代表依賴同根 K 棒內不可知的資訊）。
3. 不排除「K 棒數不足的短交易日」（事先無法知道哪天會是短日）。
"""
import numpy as np, pandas as pd
import part14_deep as P14
import part15_ib_params as P15
from part15_letters import features
from part14_deep import exits, both, SPLIT

out = open("results_part15.txt", "a")
def P(x=""): print(x); out.write(x + "\n"); out.flush()
P("\n" + "=" * 110 + "\n未來函數檢查")

rng = np.random.default_rng(7)
for confirm, buf in [("touch", 0.0), ("close5", 0.05)]:
    full, _ = P15.signals(60, 60, confirm=confirm, buf=buf)
    full = full.reset_index(drop=True)
    sample = full.iloc[rng.choice(len(full), 150, replace=False)]
    bad = 0
    for r in sample.itertuples():
        i_sig = r.i if confirm == "touch" else r.i - 1        # 訊號那根 K 棒
        t, O, H, L, C, S = P14.ARR[r.day]
        saved = P14.ARR[r.day]
        cut = i_sig + 1
        trunc = (t[:cut], O[:cut], H[:cut], L[:cut], C[:cut], S[:cut])
        P14.ARR[r.day] = trunc; P15.ARR[r.day] = trunc
        try:
            s2, _ = P15.signals(60, 60, confirm=confirm, buf=buf)
        finally:
            P14.ARR[r.day] = saved; P15.ARR[r.day] = saved
        m = s2[(s2.day == r.day) & (s2.h == r.h)]
        f_full = features(r, 15)
        if len(m) != 1:
            bad += 1; continue
        m = m.iloc[0]
        P14.ARR[r.day] = trunc
        try:
            f_cut = features(m, 15) if confirm == "touch" else None
        finally:
            P14.ARR[r.day] = saved
        same = (m.side == r.side) and np.isclose(m.lvl, r.lvl) and (m.t_in == r.t_in)
        if f_cut is not None:
            same &= all(np.isclose(f_full[k], f_cut[k]) for k in f_full)
        bad += (not same)
    P(f"  1. 截斷測試 [{confirm}]：抽 150 筆，砍掉進場後所有資料重算，訊號{'與特徵' if confirm == 'touch' else ''}不一致 {bad} 筆")
P("     （門檻 thr 用 1~5 月分位數，是「參數」不是逐筆資訊；短日排除見 3）")

P("\n  2. 延後進場：訊號後下一根 K 棒開盤進場（其餘不變）")
for confirm, buf in [("touch", 0.0), ("close5", 0.05)]:
    s, _ = P15.signals(60, 60, confirm=confirm, buf=buf)
    base = exits(s)
    d2 = s.copy()
    k0 = d2.i if confirm == "touch" else d2.i - 1
    nxt = []
    for r, k in zip(d2.itertuples(), k0):
        t, O, H, L, C, S = P14.ARR[r.day]
        nxt.append((min(k + 1, len(O) - 1), O[min(k + 1, len(O) - 1)]))
    d2["i"] = [a for a, _ in nxt]; d2["lvl"] = [b for _, b in nxt]
    P(f"    {confirm:6s} 原始   {both(base)}")
    P(f"    {confirm:6s} 延後1根 {both(exits(d2))}")

P("\n  3. 短交易日：build_days 只保留 ≥1000 根 M1 的日子")
import tpo
d = tpo.load_m1()
cnt = d.groupby("day").size()
short = cnt[cnt < 1000]
P(f"    被排除的短日 {len(short)} 天：" + ", ".join(f"{k.date()}({v})" for k, v in short.items()))
out.close()
