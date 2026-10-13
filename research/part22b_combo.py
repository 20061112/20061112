"""第二十二部分 b：M15「A 順勢突破」增加筆數 — 視窗 N 拉長、k 細掃、多個 k 合併（M1 逐根，2023/6 ~ 2026/10）。

1. 視窗 N = 96 / 192 / 480 / 2000（k = 3）。
2. k 細掃 2.5 ~ 5（每 0.25），看 k = 3 的優勢是不是孤立的尖峰。
3. 合併 k = 3 / 4 / 5（及 3~5 每 0.5）：
   - 獨立帳戶：每個 k 各自下單，同時可持有多筆（總風險會疊加）；
   - 單一持倉：依時間排序，手上有單時新的訊號一律略過（同向 / 反向都略過）。
"""
import numpy as np
import pandas as pd
import part22_zigzag as z
import part22_m1check as m

D = z.prep(z.M15)


def signals(k, N):
    P = z.zigzag(D, k)
    E = z.classify(z.events(D, P, N))
    E["time"] = D.index[E.i.values]
    E["year"] = E.time.dt.year
    S = E[(E.setup == "A") & (E.time >= m.M1.index[0])].copy()
    R, X = m.sim_m1(D, S, 15, False, return_exit=True)
    S["R"], S["exit"] = R, X
    S["entry"] = S.time + pd.Timedelta(minutes=15)  # 事件 K 棒收盤
    S["k"] = k
    return S.dropna(subset=["R"])


def stats(name, S):
    r = S.R.values
    mon = (m.M1.index[-1] - m.M1.index[0]).days / 30.4
    eq = np.cumsum(r)
    dd = (np.maximum.accumulate(eq) - eq).max()
    run = mx = 0
    for x in r:
        run = run + 1 if x < 0 else 0
        mx = max(mx, run)
    yr = S.groupby("year").R.mean()
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    print(f"  {name:28s} 每月 {len(r) / mon:5.1f} 筆 期望 {r.mean():+.3f}R t {r.mean() / r.std() * np.sqrt(len(r)):+.2f} "
          f"PF {gp / gl:.2f} 每月 {r.sum() / mon:+.2f}R 回撤 {dd:5.1f}R 連虧 {mx:2d} | "
          + " ".join(f"{y}:{v:+.2f}" for y, v in yr.items()))


def single_position(S):
    S = S.sort_values(["entry", "k"])
    keep, busy = [], pd.Timestamp.min
    for idx, r in S.iterrows():
        if r.entry < busy:
            continue
        keep.append(idx)
        busy = r.exit
    return S.loc[keep]


def dedupe_same_event(S):
    """不同 k 在同一根 K 棒、同方向的訊號只算一筆（常常是同一個高點）。"""
    return S.sort_values("k").drop_duplicates(subset=["time", "dir"])


if __name__ == "__main__":
    print("一、視窗 N（k = 3）")
    for N in (96, 192, 480, 2000):
        stats(f"N={N}", signals(3, N).sort_values("entry"))

    print("\n二、k 細掃（N = 192）")
    cache = {}
    for k in np.arange(2.5, 5.01, 0.25):
        cache[round(k, 2)] = signals(round(k, 2), 192)
        stats(f"k={k:.2f}", cache[round(k, 2)].sort_values("entry"))

    print("\n三、合併多個 k（N = 192）")
    for ks in ((3, 4, 5), (3, 3.5, 4, 4.5, 5), tuple(cache)):
        U = pd.concat([cache[k] for k in ks])
        name = "k=" + "/".join(f"{k:g}" for k in ks) if len(ks) <= 5 else "k=2.5~5 全部"
        stats(name + " 獨立帳戶", U.sort_values("entry"))
        stats(name + " 同訊號去重", dedupe_same_event(U).sort_values("entry"))
        stats(name + " 單一持倉", single_position(U))
        # 每個 k 用 1/len(ks) 風險：每月報酬與回撤（以單一 k 的 R 為單位）
        W = U.copy()
        W["R"] = W.R / len(ks)
        stats(name + " 均分風險", W.sort_values("entry"))
    print("\n  k=3 單獨，單一持倉（對照）")
    stats("k=3 單一持倉", single_position(cache[3.0]))

    print("\n四、合併 k=3/4/5 + 視窗 N=480")
    U480 = pd.concat([signals(k, 480) for k in (3, 4, 5)])
    best = dedupe_same_event(U480).sort_values("entry")
    stats("k=3/4/5 N=480 同訊號去重", best)
    stats("k=3/4/5 N=480 單一持倉", single_position(U480))
    best.to_csv("part22b_trades_k345_N480.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 5), dpi=110)
    k3 = cache[3.0].sort_values("entry")
    k345 = dedupe_same_event(pd.concat([cache[k] for k in (3, 4, 5)])).sort_values("entry")
    for S, lab, c in ((k3, "k=3 單獨（N=192）", "#9a9890"), (k345, "k=3/4/5 去重（N=192）", "#2a78d6"),
                      (best, "k=3/4/5 去重（N=480）", "#eb6834")):
        ax.plot(S.entry, S.R.cumsum(), lw=2, color=c, label=lab)
    ax.axhline(0, color="#52514e", lw=.8)
    ax.set_ylabel("累計 R（扣點差，M1 逐根）")
    ax.set_title("M15 ZigZag 順勢突破（A）：單一 k 與多 k 合併 2023/6 ~ 2026/10")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig("part22b_equity.png")
