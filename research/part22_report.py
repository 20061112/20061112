import pandas as pd, numpy as np
from part22_smc import summarize
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
ev = pd.read_pickle("part22_events.pkl")
cols = ["N","b1","b1_pl","b1_t","b05","b05_pl","b05_t","r2","r2_pl","r2_t","mfe-mae","mfe-mae_pl"]
for tf in ("15min","1h"):
    e = ev[ev.tf==tf]
    print(f"\n===== {tf}：每種價位第一次觸碰 =====")
    print(summarize(e, "kind")[cols].round(3).to_string())
    print(f"\n----- {tf}：順結構 (aligned=1) vs 逆結構 -----")
    print(summarize(e, ["kind","aligned"])[["N","b1","b1_pl","b1_t","r2","r2_pl","r2_t"]].round(3).to_string())
    print(f"\n----- {tf}：逐年 b1（真實 - 對照）-----")
    s = summarize(e, ["kind","year"])
    print((s.b1 - s.b1_pl).unstack().round(3).to_string())
e = ev[(ev.tf=="15min")&(ev.kind=="FVG")].copy()
e["wbin"] = pd.cut(e.width,[0,0.1,0.25,0.5,1,99])
print("\n----- M15 FVG 依缺口大小(ATR) -----")
print(summarize(e,"wbin")[["N","b1","b1_pl","b1_t","r2","r2_pl","r2_t"]].round(3).to_string())
for tf in ("15min","1h"):
    e = ev[ev.tf==tf].copy()
    e["agebin"] = pd.cut(e.age,[0,4,16,64,1001])
    print(f"\n----- {tf} 全部價位依觸碰前等待根數 -----")
    print(summarize(e,"agebin")[["N","b1","b1_pl","b1_t","r2","r2_pl","r2_t"]].round(3).to_string())
print("\n未結比例 b1:", ev.b1.isna().mean().round(3))
