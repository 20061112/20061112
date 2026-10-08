"""第四部分重現：python3 run_part4.py（候選表需先由 mr_features 產生，本腳本會自動重建）"""
import pandas as pd
from load import load
from backtest import resample, stats
from mr_features import candidate_table
import combos4
import stability4

m15 = load("data/XAUUSD_M15_full.csv")
m5 = resample(load("data/XAUUSD_M1_full.csv"), "5min")
candidate_table(m15).to_pickle("cand4_m15.pkl")
candidate_table(m5).to_pickle("cand4_m5.pkl")
pd.set_option("display.width", 250)

for tf, k in (("m15", 4), ("m5", 3)):
    T = pd.read_pickle(f"cand4_{tf}.pkl")
    print(f"\n=== 特徵穩定性 {tf.upper()}（2R 出場，{k} 段）===")
    print(stability4.stability(T, "pnl_R", k).to_string(index=False))

for tf, folds in (("m15", combos4.FOLDS),
                  ("m5", [("6/29~7/31", "2026-06-01", "2026-08-01"), ("8月", "2026-08-01", "2026-09-01"),
                          ("9/1~10/8", "2026-09-01", "2026-10-09")])):
    T = pd.read_pickle(f"cand4_{tf}.pkl")
    combos4.T, combos4.DAYS, combos4.FOLDS = T, T.setup.dt.normalize().nunique(), folds
    B, C, D = T.er5_20 >= 2, T.bbw_pct <= .5, T.close_outside == 1
    rules = {"全部型態": T.index == T.index, "B ER5/ER20>=2": B, "C 帶寬<=50%": C, "D 外軌外收盤": D,
             "B+C": B & C, "B+D": B & D, "C+D": C & D, "三選二": (B.astype(int) + C.astype(int) + D.astype(int)) >= 2}
    print(f"\n=== 規則組合 {tf.upper()} ===")
    print(pd.DataFrame([combos4.report(k, v) for k, v in rules.items()]).to_string(index=False))

a, b = pd.read_pickle("cand4_m15.pkl"), pd.read_pickle("cand4_m5.pkl")
days = b.setup.dt.normalize().nunique()
a = a[(a.close_outside == 1) & (a.entry_time >= b.entry_time.min())]
b = b[(b.close_outside == 1) & (b.bbw_pct <= .5)]
P = pd.concat([a.assign(tf="M15"), b.assign(tf="M5")])
s = stats(P)
print(f"\n=== 組合：M15 外軌收盤 + M5 外軌收盤且帶寬<=50%（6/29~10/8，{days} 個交易日）===")
print(f"  {s['n']} 筆，每日 {s['n'] / days:.2f}，期望 {s['expR']:+.3f}R，t {s['t']:.2f}，PF {s['pf']:.2f}")
print(P.groupby(P.entry_time.dt.to_period("M")).pnl_R.agg(["size", "mean"]).round(2).T.to_string())
P.to_csv("part4_combo_trades.csv", index=False)
