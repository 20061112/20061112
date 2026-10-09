"""第二十三部分：彙整 part23_grid.csv / part23_fwd.csv → results_part23.txt。"""
import pandas as pd

G = pd.read_csv("part23_grid.csv")
F = pd.read_csv("part23_fwd.csv")
out = []
P = out.append
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 400)
fmt = lambda x: f"{x:+.3f}" if isinstance(x, float) else x


def region(df, by):
    g = df.groupby(by)
    r = pd.DataFrame({
        "組數": g.size(),
        "平均筆數": g.n.mean().round(0),
        "每日": g.per_day.mean().round(2),
        "全期R": g.avgR.mean().round(3),
        "H1": g.h1.mean().round(3),
        "H2": g.h2.mean().round(3),
        "兩半皆正%": g.apply(lambda x: ((x.h1 > 0) & (x.h2 > 0)).mean() * 100, include_groups=False).round(0),
    })
    return r


C = G[G.kind == "cycle"]
P("=== 1. 前瞻報酬（突破方向，ATR14 單位；各參數組平均）cycle vs base ===")
f = F.groupby(["tf", "kind", "dir", "k"]).agg(n=("n", "mean"), fwd=("fwd", "mean"), t=("t", "mean")).round(3)
P(f.unstack("k").to_string())

P("\n=== 2. 循環 vs 對照（不要求先發散），依週期 × 方向（所有出場/停損平均）===")
P(region(G, ["tf", "kind", "dir"]).to_string())

P("\n=== 3. 循環：週期 × SMA/EMA ===")
P(region(C, ["tf", "type", "dir"]).to_string())

P("\n=== 4. 循環：週期 × 均線組（SMA/EMA 合併）===")
P(region(C, ["tf", "set", "dir"]).to_string())

P("\n=== 5. 循環：出場 × 停損 ===")
P(region(C, ["tf", "exit", "stop"]).to_string())

P("\n=== 6. 循環：發散門檻 / 密集門檻 / 間隔 ===")
P(region(C, ["tf", "div"]).to_string())
P(region(C, ["tf", "sq"]).to_string())
P(region(C, ["tf", "gap"]).to_string())

P("\n=== 7. SMA vs EMA 配對比較（同週期、同長度、同參數，只換均線種類）===")
key = ["tf", "set", "kind", "div", "sq", "gap", "exit", "stop", "dir"]
S = G[G.type == "sma"].set_index(key)
E = G[G.type == "ema"].set_index(key)
j = S[["avgR", "n"]].join(E[["avgR", "n"]], lsuffix="_sma", rsuffix="_ema", how="inner").reset_index()
j = j[j.kind == "cycle"]
j["diff"] = j.avgR_ema - j.avgR_sma
P(j.groupby("tf").agg(SMA平均R=("avgR_sma", "mean"), EMA平均R=("avgR_ema", "mean"),
                      SMA筆數=("n_sma", "mean"), EMA筆數=("n_ema", "mean"),
                      EMA較好比例=("diff", lambda x: (x > 0).mean())).round(3).to_string())

P("\n=== 8. 兩半都正、且每半 t > 1 的參數組（循環，筆數 ≥ 40）===")
ok = C[(C.n >= 40) & (C.t1 > 1) & (C.t2 > 1)].sort_values("t", ascending=False)
cols = ["tf", "type", "set", "div", "sq", "gap", "exit", "stop", "dir", "n", "per_day", "avgR", "t", "win", "pf", "h1", "h2", "hold"]
P(f"{len(ok)} / {len(C[C.n >= 40])} 組")
P(ok[cols].head(40).round(3).to_string(index=False))

P("\n=== 9. 對照組也一樣篩（不要求先發散）===")
Bs = G[G.kind == "base"]
okb = Bs[(Bs.n >= 40) & (Bs.t1 > 1) & (Bs.t2 > 1)].sort_values("t", ascending=False)
P(f"{len(okb)} / {len(Bs[Bs.n >= 40])} 組")
P(okb[cols].head(20).round(3).to_string(index=False))

open("results_part23.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
