"""第二十二部分：相鄰兩根 K 棒的大小與方向關係（實體 / 含影線全長）。
資料：data/XAUUSD_M1_2023_2026.csv（2023/6/1 ~ 2026/10/8），重取樣為 M1/M5/M15/H1/H4/D1。
一開始就把期間切成 發現期 2023/6~2024/12、驗證期 2025/1~2026/10，所有「找到的東西」都要在兩段同向才算。
"""
import numpy as np
import pandas as pd
from load import load

SPLIT = pd.Timestamp("2025-01-01")
TFS = [("M1", "1min", 1), ("M5", "5min", 5), ("M15", "15min", 15), ("H1", "1h", 60), ("H4", "4h", 240), ("D1", "1D", 1440)]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "tickvol": "sum", "spread": "mean"}
rng = np.random.default_rng(0)


def bars(m1, rule, mins):
    df = m1 if rule == "1min" else m1.resample(rule, label="left", closed="left").agg(AGG).dropna()
    df = df.copy()
    df["body"] = (df.close - df.open).abs()
    df["rng"] = df.high - df.low
    df["dir"] = np.sign(df.close - df.open)
    df["atr"] = df.rng.rolling(20).mean()          # 含當根，用在「這根收盤後」的標準化
    df["cost"] = df.spread * 0.01                  # 來回成本（MT5 bid 報價，點差 0.01 美元/點）
    gap = df.index.to_series().diff().dt.total_seconds().div(60)
    # 相鄰兩根才算一對：日內時框容許 2 根內（跨日休市那一小時算斷開），H4 / D1 容許跨週末
    lim = {1440: 4 * 1440, 240: 3 * 1440}.get(mins, 2 * mins)
    df["ok"] = (gap <= lim) & (gap > 0)
    return df


def pct(x):
    return f"{100 * x:5.1f}%"


def compare(a, b):
    """回傳 a>b / a<b / 相等 的比例。"""
    return (a > b).mean(), (a < b).mean(), (a == b).mean()


def part_a(df, name, out):
    """無條件：這根 vs 前一根 的大小、方向，以及 2x2 聯合機率（含隨機打亂基準）。"""
    ok = df.ok.to_numpy() & (df.dir.to_numpy() != 0) & (df.dir.shift().to_numpy() != 0)
    d = df[ok]
    p = df.shift()[ok]
    same = (d.dir == p.dir).to_numpy()
    res = {"tf": name, "n": len(d), "doji%": (df.dir == 0).mean()}
    for k in ("body", "rng"):
        big = (d[k] > p[k]).to_numpy()
        sml = (d[k] < p[k]).to_numpy()
        res[f"{k}_big"] = big.mean()
        res[f"{k}_sml"] = sml.mean()
        res[f"{k}_tie"] = 1 - big.mean() - sml.mean()
        res[f"{k}_big_same"] = (big & same).mean()
        res[f"{k}_big_opp"] = (big & ~same).mean()
        res[f"{k}_sml_same"] = (sml & same).mean()
        res[f"{k}_sml_opp"] = (sml & ~same).mean()
        res[f"{k}_same|big"] = same[big].mean()
        res[f"{k}_same|sml"] = same[sml].mean()
    res["same"] = same.mean()
    # 隨機打亂基準（K 棒順序打亂 20 次，保留每根 K 棒本身形狀）
    nulls = []
    v = df[df.dir != 0][["body", "rng", "dir"]].to_numpy()
    for _ in range(20):
        s = v[rng.permutation(len(v))]
        nulls.append([(s[1:, 2] == s[:-1, 2]).mean(),
                      ((s[1:, 0] > s[:-1, 0]) & (s[1:, 2] == s[:-1, 2])).mean(),
                      ((s[1:, 0] > s[:-1, 0]) & (s[1:, 2] != s[:-1, 2])).mean()])
    res["null_same"], res["null_body_big_same"], res["null_body_big_opp"] = np.mean(nulls, axis=0)
    out.append(res)


def table_a(rows):
    print("\n## A1. 這根 vs 前一根：變大 / 變小 / 方向（每個時框全部相鄰兩根，排除十字線）")
    print(f"{'時框':4s} {'樣本':>9s} {'十字':>6s} | {'實體變大':>8s} {'實體變小':>8s} {'相等':>6s} | {'全長變大':>8s} {'全長變小':>8s} {'相等':>6s} | {'同向(順)':>8s} {'反向(逆)':>8s} {'打亂基準':>8s}")
    for r in rows:
        print(f"{r['tf']:4s} {r['n']:9d} {pct(r['doji%'])} | {pct(r['body_big'])}  {pct(r['body_sml'])}  {pct(r['body_tie'])} | "
              f"{pct(r['rng_big'])}  {pct(r['rng_sml'])}  {pct(r['rng_tie'])} | {pct(r['same'])}  {pct(1 - r['same'])}  {pct(r['null_same'])}")
    for k, lab in (("body", "實體"), ("rng", "全長(含影線)")):
        print(f"\n## A2. 2x2 聯合機率：{lab}變大/變小 × 同向/反向（四格加總 = 100% − 相等）")
        print(f"{'時框':4s} | {'變大+順':>8s} {'變大+逆':>8s} {'變小+順':>8s} {'變小+逆':>8s} | {'變大時順勢率':>10s} {'變小時順勢率':>10s}")
        for r in rows:
            print(f"{r['tf']:4s} | {pct(r[k + '_big_same'])}  {pct(r[k + '_big_opp'])}  {pct(r[k + '_sml_same'])}  {pct(r[k + '_sml_opp'])} | "
                  f"{pct(r[k + '_same|big']):>10s}   {pct(r[k + '_same|sml']):>10s}")
    print("\n  打亂基準（實體）：變大+順 / 變大+逆 = " + "  ".join(f"{r['tf']} {pct(r['null_body_big_same'])}/{pct(r['null_body_big_opp'])}" for r in rows))


def pair_frame(df):
    """每一列 = 一根 K 棒 t，附上 t-1、t-2 的資訊與下一根 t+1 的結果（全部在 t 收盤時已知，除了 nxt_*）。"""
    f = pd.DataFrame(index=df.index)
    f["dir"] = df.dir
    f["body_atr"] = df.body / df.atr.shift()            # 這根實體 / 前 20 根平均全長
    f["rng_atr"] = df.rng / df.atr.shift()
    f["body_ratio"] = df.body / df.rng.replace(0, np.nan)
    # 收盤在 K 棒的位置，換成「順著這根方向」：1 = 收在最極端，0 = 收在另一端
    clv = (df.close - df.low) / df.rng.replace(0, np.nan)
    f["clv_dir"] = np.where(df.dir > 0, clv, 1 - clv)
    f["body_grow"] = np.sign(df.body - df.body.shift())
    f["rng_grow"] = np.sign(df.rng - df.rng.shift())
    f["same_prev"] = (df.dir == df.dir.shift()) & (df.dir != 0)
    f["ok_prev"] = df.ok
    nxt_ok = df.ok.shift(-1).fillna(False).astype(bool)
    f["nxt_dir"] = df.dir.shift(-1).where(nxt_ok)
    f["nxt_cont"] = (df.dir.shift(-1) == df.dir).where(nxt_ok & (df.dir.shift(-1) != 0) & (df.dir != 0))
    f["nxt_ret"] = ((df.close.shift(-1) - df.close) * df.dir / df.atr).where(nxt_ok)   # 順這根方向，單位 ATR
    f["nxt_ret_usd"] = ((df.close.shift(-1) - df.close) * df.dir).where(nxt_ok)
    f["nxt_body_grow"] = np.sign(df.body.shift(-1) - df.body).where(nxt_ok)
    f["nxt_rng_grow"] = np.sign(df.rng.shift(-1) - df.rng).where(nxt_ok)
    f["cost"] = df.cost
    f["cost_atr"] = df.cost / df.atr
    f["atr"] = df.atr
    # 連續同向根數（含這根）
    run = np.zeros(len(df), int)
    dv, okv = df.dir.to_numpy(), df.ok.to_numpy()
    for i in range(len(df)):
        run[i] = run[i - 1] + 1 if (i and okv[i] and dv[i] != 0 and dv[i] == dv[i - 1]) else (1 if dv[i] != 0 else 0)
    f["run"] = run
    # 連續實體放大根數（含這根）
    g = f.body_grow.to_numpy()
    er = np.zeros(len(df), int)
    for i in range(1, len(df)):
        er[i] = er[i - 1] + 1 if (okv[i] and g[i] > 0) else 0
    f["expand_run"] = er
    f["period"] = np.where(f.index < SPLIT, "發現", "驗證")
    f["year"] = f.index.year
    return f[f.dir != 0].dropna(subset=["atr"])


def tstat(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan


def cell(sub):
    c = sub.nxt_cont.dropna()
    return f"n={len(c):7d} 順勢率 {pct(c.mean())} 下一根順勢報酬 {sub.nxt_ret.mean():+.3f}ATR (t{tstat(sub.nxt_ret):+5.1f})"


def part_b(f, name):
    print(f"\n### {name}")
    base = f[f.ok_prev]
    print(f"  全部：{cell(base)}  下一根實體變大 {pct((base.nxt_body_grow > 0).mean())}")
    lab = {(1, True): "這根實體變大+順", (1, False): "這根實體變大+逆", (-1, True): "這根實體變小+順", (-1, False): "這根實體變小+逆"}
    for (g, s), l in lab.items():
        sub = base[(base.body_grow == g) & (base.same_prev == s)]
        a, b = sub[sub.period == "發現"], sub[sub.period == "驗證"]
        print(f"  {l}: {cell(sub)} | 下一根實體再變大 {pct((sub.nxt_body_grow > 0).mean())} | 發現 {pct(a.nxt_cont.mean())} / 驗證 {pct(b.nxt_cont.mean())}")


def by_bins(f, col, bins, labels, name):
    print(f"\n  依「{name}」分組（下一根）：")
    q = pd.cut(f[col], bins, labels=labels)
    for lab in labels:
        sub = f[q == lab]
        if len(sub) < 200:
            continue
        a, b = sub[sub.period == "發現"], sub[sub.period == "驗證"]
        print(f"    {lab:>10s}: {cell(sub)} | 發現 {pct(a.nxt_cont.mean())} {a.nxt_ret.mean():+.3f} / 驗證 {pct(b.nxt_cont.mean())} {b.nxt_ret.mean():+.3f} | 下根實體變大 {pct((sub.nxt_body_grow > 0).mean())}")


def main():
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    print(f"資料 {m1.index[0]} ~ {m1.index[-1]}，M1 {len(m1):,} 根；發現期 < {SPLIT.date()} ≤ 驗證期")
    rows, frames = [], {}
    for name, rule, mins in TFS:
        df = bars(m1, rule, mins)
        part_a(df, name, rows)
        frames[name] = pair_frame(df)
    table_a(rows)

    print("\n\n# B. 用「這根相對前一根」的狀態，預測下一根（全部在這根收盤時已知）")
    print("  順勢率 = 下一根與這根同色；順勢報酬 = 下一根收盤 − 這根收盤，乘上這根方向，除以 ATR20")
    for name, f in frames.items():
        part_b(f, name)

    print("\n\n# C. 這根的大小 / 形狀 → 下一根")
    for name, f in frames.items():
        print(f"\n### {name}")
        base = f[f.ok_prev]
        by_bins(base, "body_atr", [0, .25, .5, 1, 1.5, 2, 3, np.inf], ["<0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", ">3"], "實體 / ATR20")
        by_bins(base, "clv_dir", [-.01, .5, .7, .85, .95, 1.01], ["收在後半", "0.5-0.7", "0.7-0.85", "0.85-0.95", "收在極端"], "收盤位置（順方向，1=收在最高/最低）")
        by_bins(base, "run", [0, 1, 2, 3, 4, 5, 7, 100], ["1", "2", "3", "4", "5", "6-7", "8+"], "連續同色根數")
        by_bins(base, "expand_run", [-1, 0, 1, 2, 3, 100], ["0(變小)", "1", "2", "3", "4+"], "連續實體放大根數")

    for name, f in frames.items():
        f.to_pickle(f"/tmp/claude-0/-home-user-20061112/0370e1da-5eaa-5078-98c4-a5f5f016d428/scratchpad/p22_{name}.pkl")


if __name__ == "__main__":
    main()
