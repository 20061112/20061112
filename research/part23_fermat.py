"""第二十三部分：用費馬原理定義「價格走的最短（光程最小）路徑」，看符合 / 不符合時之後會怎樣。

介質：過去 1440 根（1 天）的時間價格剖面（TPO）。n(P) = 每 1 ATR 價格區間停留的分鐘數 + 1（下限避免新價位 n=0）。
  停留越久 = 介質越稠密 = 光走得越慢（市場剖面說法：高量節點走得慢、低量節點走得快）。
  剖面只用視窗開始以前的資料，視窗內的 K 棒不會影響它（沒有前視）。
光程座標：F(P) = ∫ n dP（把價格軸依剖面拉伸）。單調路徑 A→B 的光程 = |F(B) − F(A)|，這就是費馬最短光程。

視窗 W 根（A = 視窗起點收盤、B = 現在收盤）：
  dP 版：Φ_P = |F(B)−F(A)| / Σ|F(c_i)−F(c_{i−1})|      — 實際光程中「必要」的比例（1 = 完全走最短路徑，越小越繞）
         對照：ER = |B−A| / Σ|Δc|                       — 不考慮介質的普通效率比
  dt 版：Φ_t = |F(B)−F(A)| / W                         — 介質預期需要的時間 / 實際花的時間
         （>1 = 比介質允許的還快，「超光速」；<1 = 停滯，花了比最短時間更多的時間）
         對照：速度 = |B−A| / (ATR·W)                    — 不考慮介質的普通速度
局部光線定律：瞬時速度應 ∝ 1/n(P)；檢查價格在稠密區是否真的走得慢，以及「違反」時之後會怎樣。
輸出：results_part23.txt
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from numba import njit
from load import load
import physics_features as pf

out = open("results_part23.txt", "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")
    out.flush()


@njit(cache=True)
def _fermat(hi, lo, cl, atr, ts, W, look, binw, eps):
    """對每個取樣點 t：剖面 = [t−W−look, t−W) 的 K 棒（每根 1 分鐘平均攤在高低區間），
    回傳 光程淨值、光程路徑長、目前價位的 n、視窗起點價位的 n。"""
    m = len(ts)
    onet = np.full(m, np.nan)
    opath = np.full(m, np.nan)
    n_now = np.full(m, np.nan)
    n_start = np.full(m, np.nan)
    for j in range(m):
        t = ts[j]
        s = t - W
        if s - look < 0:
            continue
        a = atr[s]
        bw = binw * a
        pmin, pmax = 1e18, -1e18
        for k in range(s - look, s + 1):
            pmin = min(pmin, lo[k])
            pmax = max(pmax, hi[k])
        for k in range(s, t + 1):
            pmin = min(pmin, cl[k])
            pmax = max(pmax, cl[k])
        pmin -= bw
        pmax += bw
        nb = int((pmax - pmin) / bw) + 2
        h = np.zeros(nb)
        for k in range(s - look, s):
            b0 = (lo[k] - pmin) / bw
            b1 = (hi[k] - pmin) / bw
            if b1 - b0 < 1e-9:
                h[int(b0)] += 1.0
                continue
            i0, i1 = int(b0), int(b1)
            for i in range(i0, i1 + 1):
                ov = min(b1, i + 1.0) - max(b0, float(i))
                if ov > 0:
                    h[i] += ov / (b1 - b0)
        # n：每 1 ATR 停留分鐘數 + eps；F 為累積光程（單位：分鐘）
        n = h / binw + eps
        F = np.zeros(nb + 1)
        for i in range(nb):
            F[i + 1] = F[i] + n[i] * binw  # 每格寬 binw ATR

        def Fx(x):
            u = (x - pmin) / bw
            i = int(u)
            if i < 0:
                return F[0]
            if i >= nb:
                return F[nb]
            return F[i] + (u - i) * (F[i + 1] - F[i])

        onet[j] = abs(Fx(cl[t]) - Fx(cl[s]))
        acc = 0.0
        prev = Fx(cl[s])
        for k in range(s + 1, t + 1):
            cur = Fx(cl[k])
            acc += abs(cur - prev)
            prev = cur
        opath[j] = acc
        n_now[j] = n[int((cl[t] - pmin) / bw)]
        n_start[j] = n[int((cl[s] - pmin) / bw)]
    return onet, opath, n_now, n_start


def daily_ic(x, y):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    ic = df.groupby(df.index.date).apply(lambda t: t.x.rank().corr(t.y.rank()) if len(t) > 10 else np.nan).dropna()
    ic.index = pd.to_datetime(ic.index)
    return ic.mean(), ic.mean() / ic.std() * np.sqrt(len(ic)), ic.groupby(ic.index.year).mean()


def ic_line(name, x, y):
    m, t, yr = daily_ic(x, y)
    P(f"  {name:<40s} IC {m:+.4f}  t {t:+6.2f}  | " + " ".join(f"{k}:{v:+.3f}" for k, v in yr.items()))


def resid_on(x, c):
    df = pd.DataFrame({"x": x, "c": c}).dropna()
    rx, rc = df.x.rank(), df.c.rank()
    return (rx - np.cov(rx, rc)[0, 1] / rc.var() * rc).reindex(x.index)


def qtab(df, col, ys, q=5):
    b = pd.qcut(df[col].rank(method="first"), q, labels=False) + 1
    g = df.groupby(b)
    t = pd.DataFrame({"中位": g[col].median()})
    for y in ys:
        t[y] = g[y].mean()
        t[f"t_{y}"] = g[y].mean() / g[y].std() * np.sqrt(g[y].count())
    t["n"] = g.size()
    return t.round(3)


d = pf.base(load("data/XAUUSD_M1_2023_2026.csv"))
Fw = pf.forward(d)
hi, lo, cl, atr = d.high.values, d.low.values, d.close.values, d.atr.values
seg = d.seg.values
P(f"資料 {d.index[0]} ~ {d.index[-1]}；報酬單位 ATR60(M1)；點差 ≈ 0.09 ATR")

for W in (30, 60):
    STEP = 15
    ts = np.arange(1440 + W + 1, len(d) - 61, STEP)
    ts = ts[seg[ts - W] == seg[ts]]  # 視窗內不跨休市
    onet, opath, n_now, n_start = _fermat(hi, lo, cl, atr, ts.astype(np.int64), W, 1440, 0.1, 1.0)
    idx = d.index[ts]
    net = cl[ts] - cl[ts - W]
    path = pd.Series(np.abs(np.diff(cl))).rolling(W).sum().values[ts - 1]
    X = pd.DataFrame({
        "PhiP": onet / opath, "ER": np.abs(net) / path,
        "Phit": onet / W, "speed": np.abs(net) / (atr[ts] * W),
        "netATR": net / atr[ts], "n_now": n_now, "n_start": n_start,
        "cont15": np.sign(net) * Fw[15].values[ts], "cont60": np.sign(net) * Fw[60].values[ts],
        "vol15": Fw["path15"].values[ts]}, index=idx)
    X["lnPhit"] = np.log(X.Phit)
    X["yr"] = X.index.year
    P("\n" + "=" * 110 + f"\n視窗 W = {W} 根，每 {STEP} 根取樣一次，共 {len(X):,} 筆\n" + "=" * 110)
    P("0. 價格平常有多「費馬」？")
    P(f"  Φ_P 中位 {X.PhiP.median():.3f}（普通 ER 中位 {X.ER.median():.3f}）；Φ_P 與 ER 的排名相關 {X.PhiP.corr(X.ER, method='spearman'):.3f}")
    P(f"  Φ_t 中位 {X.Phit.median():.2f}；Φ_t > 1（比介質允許的還快）的比例 {(X.Phit > 1).mean():.1%}；"
      f"Φ_t 與普通速度的排名相關 {X.Phit.corr(X.speed, method='spearman'):.3f}")
    P(f"  Φ_P ≥ 0.8（幾乎走最短光程）的比例 {(X.PhiP >= 0.8).mean():.1%}；隨機漫步下 W={W} 根的普通 ER 中位約 {np.sqrt(2 / np.pi / W):.2f}")

    big = X[X.netATR.abs() >= 1.0]
    P(f"\n1. dP 版：Φ_P（光程效率）vs 之後延續（只看視窗淨移動 ≥ 1 ATR 的 {len(big):,} 筆；>0 = 延續）")
    P(qtab(big, "PhiP", ["cont15", "cont60", "vol15"]).to_string())
    ic_line("Φ_P → 延續15", big.PhiP, big.cont15)
    ic_line("Φ_P → 延續60", big.PhiP, big.cont60)
    ic_line("對照 ER → 延續15", big.ER, big.cont15)
    ic_line("Φ_P 扣掉 ER → 延續15", resid_on(big.PhiP, big.ER), big.cont15)
    ic_line("Φ_P 扣掉 ER → 延續60", resid_on(big.PhiP, big.ER), big.cont60)

    P(f"\n2. dt 版：Φ_t（介質預期時間 / 實際時間）vs 之後延續")
    P(qtab(big, "Phit", ["cont15", "cont60", "vol15"]).to_string())
    ic_line("Φ_t → 延續15", big.Phit, big.cont15)
    ic_line("Φ_t → 延續60", big.Phit, big.cont60)
    ic_line("對照 普通速度 → 延續15", big.speed, big.cont15)
    ic_line("Φ_t 扣掉 普通速度 → 延續15", resid_on(big.Phit, big.speed), big.cont15)
    ic_line("Φ_t 扣掉 普通速度 → 延續60", resid_on(big.Phit, big.speed), big.cont60)

    big = big.assign(Phit_res=resid_on(big.Phit, big.speed))
    P("  Φ_t 扣掉普通速度後的五分位（同樣速度下，穿過的區域「多稠密」）：")
    P(qtab(big.dropna(subset=["Phit_res"]), "Phit_res", ["cont15", "cont60"]).drop(columns="中位").to_string())
    hb = pd.qcut(big.Phit_res.rank(method="first"), 5, labels=False) + 1
    P("  同上，延續60 > 0 的比例（IC 看的是排名，所以也看勝率與中位數）：")
    P(pd.DataFrame({"延續機率": big.groupby(hb).cont60.apply(lambda x: (x > 0).mean()),
                    "中位數": big.groupby(hb).cont60.median()}).round(3).to_string())
    P("  拆解：同樣速度下 Φ_t 高 = 剛剛穿過的是過去 1 天停留很久的價區（離開成交密集區）；"
      "Φ_t 低 = 穿過的是很少停留或全新的價區（真空 / 新價位）")
    nb = big.assign(新價位=big.n_now <= 1.5, 速度組=pd.qcut(big.speed.rank(method="first"), 3, labels=["慢", "中", "快"]),
                    Φt組=lambda x: x.groupby("速度組", observed=True).Phit.transform(lambda v: np.where(v > v.median(), "Φt高", "Φt低")))
    P(nb.pivot_table(index="速度組", columns="Φt組", values="cont60", aggfunc="mean", observed=True).round(3).to_string())
    P(f"  目前價位在過去 1 天剖面之外（新價位）的比例：{nb.新價位.mean():.1%}；新價位 cont60 {nb.cont60[nb.新價位].mean():+.3f}，"
      f"剖面內 {nb.cont60[~nb.新價位].mean():+.3f}")
    P("  只看剖面內（排除新價位）：Φ_t 扣掉速度 → 延續60")
    ic_line("    剖面內", resid_on(big.Phit[~nb.新價位], big.speed[~nb.新價位]), big.cont60[~nb.新價位])
    P("\n3. 符合 / 不符合費馬的四種狀態（Φ_P 以 0.5 為界、Φ_t 以 1 為界），之後 15 / 60 根延續（ATR）")
    big = big.assign(路徑=np.where(big.PhiP >= 0.5, "走最短光程", "繞路"), 時間=np.where(big.Phit >= 1, "超光速(快)", "停滯(慢)"))
    for y in ("cont15", "cont60"):
        g = big.groupby(["路徑", "時間"])[y]
        P(f"  {y}")
        P(pd.DataFrame({"平均": g.mean(), "t": g.mean() / g.std() * np.sqrt(g.count()), "n": g.count()}).round(3).to_string())
    fs = big[(big.PhiP >= 0.5) & (big.Phit >= 1)]
    P(f"  「走最短光程 + 超光速」那格裡，現價在過去 1 天剖面之外（新價位）的比例 {(fs.n_now <= 1.5).mean():.1%}，普通 ER 中位 {fs.ER.median():.2f}")
    P("  對照：把 Φ_P 換成普通 ER（≥0.5）、Φ_t 換成普通速度（≥ 中位），同樣的四格 cont60：")
    big2 = big.assign(路徑=np.where(big.ER >= 0.5, "ER高", "ER低"), 時間=np.where(big.speed >= big.speed.median(), "快", "慢"))
    g = big2.groupby(["路徑", "時間"]).cont60
    P(pd.DataFrame({"平均": g.mean(), "t": g.mean() / g.std() * np.sqrt(g.count()), "n": g.count()}).round(3).to_string())
    P("  逐年 cont60：")
    P(big.pivot_table(index="yr", columns=["路徑", "時間"], values="cont60", aggfunc="mean").round(3).to_string())

# ---------------------------------------------------------------- 局部光線定律
P("\n" + "=" * 110 + "\n4. 局部光線定律：價格在稠密區（n 大）真的走得比較慢嗎？\n" + "=" * 110)
STEP = 5
ts = np.arange(1440 + 61, len(d) - 61, STEP)
# 剖面到 t−30 為止（W=30 讓 _fermat 用 [t−30−1440, t−30)），n_now = 目前價位的 n
ts = ts[seg[ts - 30] == seg[ts]]
_, _, n_now, _ = _fermat(hi, lo, cl, atr, ts.astype(np.int64), 30, 1440, 0.1, 1.0)
idx = d.index[ts]
sp_past = pd.Series(np.abs(np.diff(cl, prepend=np.nan))).rolling(5).sum().values[ts] / atr[ts] / 5
sp_fut = Fw["path15"].values[ts] / 15
c5 = cl[ts] - cl[ts - 5]
L = pd.DataFrame({"n": n_now, "sp_past": sp_past, "sp_fut": sp_fut,
                  "cont15": np.sign(c5) * Fw[15].values[ts], "cont60": np.sign(c5) * Fw[60].values[ts]}, index=idx)
L["ln_n"] = np.log(L.n)
P(f"  取樣 {len(L):,} 筆（每 5 根）")
ic_line("n(目前價位) → 未來15根速度", L.n, L.sp_fut)
ic_line("n(目前價位) → 最近5根速度", L.n, L.sp_past)
L["nq"] = pd.qcut(L.n.rank(method="first"), 5, labels=False) + 1
g = L.groupby("nq")
P(pd.DataFrame({"n中位(分鐘/ATR)": g.n.median(), "未來15根速度": g.sp_fut.mean(), "最近5根速度": g.sp_past.mean()}).round(3).to_string())
P("  費馬預測：n 越大速度越慢（IC 應為負）。")
# 違反：在稠密區卻走得快（或在稀疏區卻走得慢）
L["viol"] = resid_on(np.log(L.sp_past.clip(lower=1e-4)), -L.ln_n)  # 速度扣掉 n 的預測
P("\n5. 違反局部定律的程度（實際速度 − n 預測的速度，正 = 比介質允許的快）vs 之後")
ic_line("違反度 → 延續15（最近5根方向）", L.viol, L.cont15)
ic_line("違反度 → 延續60", L.viol, L.cont60)
ic_line("違反度 → 未來15根速度", L.viol, L.sp_fut)
ic_line("對照 只用最近5根速度 → 延續15", L.sp_past, L.cont15)
ic_line("對照 只用最近5根速度 → 未來15根速度", L.sp_past, L.sp_fut)
P(qtab(L.dropna(subset=["viol"]), "viol", ["cont15", "cont60", "sp_fut"]).drop(columns="中位").to_string())
out.close()
