"""第五十一部分：什麼狀態下 PF 大幅提升 / 變很醜？能不能當加倉依據？＋ ATR 類熵（entropy）因子。

樣本：最終版 + 兩個防護（RR 3），實際下單；R 已扣點差 + 0.3 滑價。另用影子訊號（未經權益濾網）做穩健性檢查。
嚴謹度：
  1. 門檻只用訓練期（2023/8~2024/12）的五分位決定；驗證期（2025~2026/10）沿用。
  2. PF = 獲利 R 總和 / 虧損 R 總和。看最好 vs 最差五分位的 PF 比值，訓練與驗證都要同方向。
  3. 驗證期用「以天為單位的區塊自助法」（2,000 次）算 PF 差的 90% 信賴區間。
  4. 打亂對照：把因子在「天」之間隨機打亂 200 次，看隨機狀況下會出現多大的 PF 差（多重檢定的基準）。
  5. 通過的因子才拿去做加倉測試（好狀態 1.5 倍、壞狀態 0.5 倍），以固定風險 R 與 0.02 口美元兩種方式評估。

ATR 類熵（全部只用進場前已完成的資料，正規化到 0~1）：
  ATR熵_M15(20)   前 20 根 M15 的真實波幅 TR_k，p_k = TR_k / ΣTR，H = −Σ p ln p / ln 20
                  低 = 波動集中在少數幾根（急衝 / 消息），高 = 波動平均分布（穩定的波動）
  ATR熵_M1(60)    前 60 分鐘 M1 的 TR 同樣計算（更短期）
  ATR熵_日(10)    前 10 天日波幅
  ATR熵變化       ATR熵_M15 現在 − 20 根前
  排列熵_M15      前 48 根 M15 收盤的排列熵（order 3）：低 = 走勢有結構（趨勢 / 規律），高 = 雜亂
  排列熵_日       前 30 天日收盤的排列熵（order 3）
  方向熵_M15      前 20 根 M15 漲跌方向的二元熵：低 = 一面倒，高 = 漲跌各半
"""
import numpy as np
import pandas as pd
from itertools import permutations
import part49_safeguards as P
import part24_prev_bar_breakout as B

rng = np.random.default_rng(11)
W, IDX = P.W, P.IDX
H, Lo, C = W.H, W.Lo, W.C
O1 = W.O1
dayv = W.dayv

# ---------- 熵工具 ----------
def norm_entropy_rows(M):
    s = M.sum(1, keepdims=True)
    p = np.where(s > 0, M / np.where(s > 0, s, 1), 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0), 1)
    return h / np.log(M.shape[1])


def rolling_windows(a, w):
    a = np.asarray(a, float)
    out = np.full((len(a), w), np.nan)
    for k in range(w):
        out[w - 1:, k] = a[k:len(a) - w + 1 + k]
    return out


PATS = {pm: k for k, pm in enumerate(permutations(range(3)))}


def perm_entropy_series(x, w, order=3):
    x = np.asarray(x, float)
    n = len(x)
    codes = np.full(n, -1)
    for t in range(order - 1, n):
        seg = x[t - order + 1:t + 1]
        if np.isfinite(seg).all():
            codes[t] = PATS[tuple(np.argsort(seg, kind="stable"))]
    out = np.full(n, np.nan)
    m = len(PATS)
    for t in range(w + order - 2, n):
        c = codes[t - w + 1:t + 1]
        c = c[c >= 0]
        if len(c) < w * 0.8:
            continue
        cnt = np.bincount(c, minlength=m) / len(c)
        cnt = cnt[cnt > 0]
        out[t] = -(cnt * np.log(cnt)).sum() / np.log(m)
    return out


def binary_entropy(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# ---------- M15 ----------
pid = B.period_ids(15)
g = pd.DataFrame({"pid": pid, "H": H, "L": Lo, "O": O1, "C": C})
b = g.groupby("pid").agg(H=("H", "max"), L=("L", "min"), O=("O", "first"), C=("C", "last"))
tr15 = np.maximum(b.H - b.L, np.maximum((b.H - b.C.shift(1)).abs(), (b.L - b.C.shift(1)).abs())).values
ent15 = norm_entropy_rows(rolling_windows(tr15, 20))
ent15_chg = ent15 - np.r_[np.full(20, np.nan), ent15[:-20]]
pe15 = perm_entropy_series(b.C.values, 48)
up = (b.C.values > b.O.values).astype(float)
dirent15 = binary_entropy(pd.Series(up).rolling(20).mean().values)
# ---------- 日 ----------
dl = B.daily.copy()
dtr = (dl.H - dl.L).values
entd = pd.Series(norm_entropy_rows(rolling_windows(dtr, 10)), index=dl.index).shift(1)
ped = pd.Series(perm_entropy_series(dl.C.values, 30), index=dl.index).shift(1)
dd = pd.DataFrame({"pc": dl.C, "ma": dl.C.rolling(50).mean(), "atr": (dl.H - dl.L).rolling(10).mean(),
                   "er10": (dl.C - dl.C.shift(10)).abs() / dl.C.diff().abs().rolling(10).sum(),
                   "run20": dl.C - dl.C.shift(20), "slope": dl.C.rolling(50).mean() - dl.C.rolling(50).mean().shift(10),
                   "atrrel": (dl.H - dl.L).rolling(10).mean() / (dl.H - dl.L).rolling(10).mean().rolling(60).mean()}).shift(1)
dd["o"] = dl.O
rng15 = (b.H - b.L)
f15 = pd.DataFrame({"atr20": rng15.rolling(20).mean(), "atr5": rng15.rolling(5).mean(),
                    "bbw": (4 * b.C.rolling(20).std() / b.C.rolling(20).mean()).rolling(500, min_periods=100).rank(pct=True),
                    "ret16": b.C - b.C.shift(16)})
def features(T):
    i = T.i.values
    dirv = T.dir.values
    px = T.entry.values
    k = pid[i] - 1  # 前一根已完成的 M15（以 pid 序號；跨日時為前一日最後一根）
    k = np.maximum(k, 0)
    # ---------- M1 前 60 分鐘 ----------
    tr1 = np.maximum(H - Lo, np.maximum(np.abs(H - np.r_[C[0], C[:-1]]), np.abs(Lo - np.r_[C[0], C[:-1]])))
    ent1 = np.array([norm_entropy_rows(tr1[max(t - 60, 0):t][None, :])[0] if t >= 60 else np.nan for t in i])

    F = pd.DataFrame(index=T.index)
    F["ATR熵_M15(20)"] = ent15[k]
    F["ATR熵_M1(60)"] = ent1
    F["ATR熵_日(10)"] = entd.reindex(T.date).values
    F["ATR熵變化_M15"] = ent15_chg[k]
    F["排列熵_M15(48)"] = pe15[k]
    F["排列熵_日(30)"] = ped.reindex(T.date).values
    F["方向熵_M15(20)"] = dirent15[k]

    # 其他（沿用第三十四、四十四部分的定義）
    S_ = dd.reindex(T.date).reset_index(drop=True)
    F15 = f15.iloc[k].reset_index(drop=True)
    F["離50MA/ATR(dir)"] = (S_.pc - S_.ma).values / S_.atr.values * dirv
    F["斜率強度(dir)"] = S_.slope.values / S_.atr.values * dirv
    F["日ER10"] = S_.er10.values
    F["前20日漲/ATR(dir)"] = S_.run20.values / S_.atr.values * dirv
    F["日ATR/60日均"] = S_.atrrel.values
    F["今日漲跌/ATR(dir)"] = (px - S_.o.values) / S_.atr.values * dirv
    F["ATR15/ATR日"] = F15.atr20.values / S_.atr.values
    F["ATR15 5/20"] = F15.atr5.values / F15.atr20.values
    F["BB寬度百分位"] = F15.bbw.values
    F["4h動能(dir)"] = F15.ret16.values / F15.atr20.values * dirv
    F["小時"] = T.進場時間.dt.hour.values.astype(float)



    return F


def pf(r):
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return gp / gl if gl > 0 else np.nan


lines = []


def p(s=""):
    print(s, flush=True)
    lines.append(s)


def shadow_set():
    """影子訊號：A+B+斜率 → 上限 8，沒有權益濾網、沒有防護（RR 3）。"""
    rows = []
    for i_, L, side, a_, vr in P.EV:
        risk = 2.0 * a_
        r = P.G.sim_detail(i_, L, side, L - side * risk, L + side * 3 * risk)
        if r is None:
            continue
        j, px_, why = r
        net = (px_ - L) * side - P.SP[i_] - P.SLIP
        rows.append((IDX[i_], IDX[j], i_, j, side, L, net, net / risk, risk, vr))
    x = pd.DataFrame(rows, columns=["進場時間", "出場時間", "i", "j", "dir", "entry", "net", "R", "risk", "volrel"])
    return P.cap_total(x.sort_values("進場時間").reset_index(drop=True), 8).reset_index(drop=True)


def prep(x):
    x = x.copy()
    x["date"] = x.進場時間.dt.normalize()
    x["train"] = x.進場時間 < "2025-01-01"
    return x.reset_index(drop=True)


X0 = P.run(3)
T = prep(P.daily_stop(X0[~(X0.volrel > 2.2).values].reset_index(drop=True), 15))
SH = prep(shadow_set())


def analyze(T, label, THR=1.25):
    F = features(T)
    tr_m = T.train.values
    R = T.R.values
    days = T.date.values
    p(f"\n################ {label}：{len(T)} 筆；整體 PF 訓練 {pf(R[tr_m]):.2f}、驗證 {pf(R[~tr_m]):.2f} ################")
    p(f"每列：五分位 Q1→Q5 的 PF（訓練切點）；★ = 好/壞五分位同方向、PF 比兩段都 ≥ {THR}")
    res = []
    for f in F.columns:
        v = F[f].values.astype(float)
        vt = v[tr_m & np.isfinite(v)]
        if len(np.unique(vt)) < 5:
            continue
        qs = np.unique(np.quantile(vt, [0.2, 0.4, 0.6, 0.8]))
        q = np.digitize(v, qs)
        q[~np.isfinite(v)] = -1
        nb = len(qs) + 1
        pt = [pf(R[tr_m & (q == b_)]) for b_ in range(nb)]
        pv = [pf(R[~tr_m & (q == b_)]) for b_ in range(nb)]
        best_hi = pt[-1] > pt[0]
        hi_b, lo_b = (nb - 1, 0) if best_hi else (0, nb - 1)
        rt, rv = pt[hi_b] / pt[lo_b], pv[hi_b] / pv[lo_b]
        ok = rt >= THR and rv >= THR
        res.append(dict(f=f, pt=pt, pv=pv, rt=rt, rv=rv, ok=ok, hi=hi_b, lo=lo_b, q=q, best_hi=best_hi))
        p(f"{'★' if ok else ' '} {f:18s} 訓練 " + " ".join(f"{x:4.2f}" for x in pt) + "  | 驗證 " + " ".join(f"{x:4.2f}" for x in pv)
          + f"  | 好/壞 PF 比 訓練 {rt:4.2f} 驗證 {rv:4.2f}（{'大' if best_hi else '小'}較好）")
    oks = [r for r in res if r["ok"]]
    p(f"  → 通過 {len(oks)} / {len(res)} 個")
    if oks:
        p("  驗證期 好−壞 PF 差：90% 區間（天區塊自助 1,000 次）；打亂對照 = 整天的因子值隨機換到別天 200 次的 95% 分位")
    ud = np.unique(days)
    uv = np.unique(days[~tr_m])
    didx = {d: np.where(days == d)[0] for d in ud}
    for r_ in oks:
        q, hi, lo = r_["q"], r_["hi"], r_["lo"]
        diffs = []
        for _ in range(1000):
            ix = np.concatenate([didx[d] for d in rng.choice(uv, len(uv), replace=True)])
            a_, b_ = pf(R[ix][q[ix] == hi]), pf(R[ix][q[ix] == lo])
            if np.isfinite(a_) and np.isfinite(b_):
                diffs.append(a_ - b_)
        lo5, hi95 = np.quantile(diffs, [0.05, 0.95])
        qday = {d: q[didx[d]] for d in ud}
        nulls = []
        for _ in range(200):
            perm = rng.permutation(ud)
            qq = np.empty(len(q), int)
            for d_, d2 in zip(ud, perm):
                qq[didx[d_]] = np.resize(qday[d2], len(didx[d_]))
            m = ~tr_m
            a_, b_ = pf(R[m & (qq == hi)]), pf(R[m & (qq == lo)])
            nulls.append(a_ - b_ if np.isfinite(a_) and np.isfinite(b_) else 0)
        real = r_["pv"][hi] - r_["pv"][lo]
        sig = lo5 > 0 and real > np.quantile(nulls, .95)
        r_["sig"] = sig
        p(f"  {r_['f']:18s} 驗證 PF 差 {real:+.2f}，90% 區間 [{lo5:+.2f}, {hi95:+.2f}]；打亂 95% 分位 {np.quantile(nulls, .95):+.2f} → {'顯著' if sig else '不顯著'}")
    return F, res, oks


def evaluate(T, w, lab):
    tr_m = T.train.values
    days = T.date.values
    out = []
    for unit in ("R", "usd"):
        val = (T.R.values if unit == "R" else T.net.values * 2) * w
        s_ = pd.Series(val).groupby(days).sum(); e = s_.cumsum(); m = (e.cummax() - e).max()
        st = pd.Series(val[tr_m]).groupby(days[tr_m]).sum(); et = st.cumsum(); mt = (et.cummax() - et).max()
        sv = pd.Series(val[~tr_m]).groupby(days[~tr_m]).sum(); ev = sv.cumsum(); mv = (ev.cummax() - ev).max()
        out.append((val.sum(), m, val[tr_m].sum() / mt, val[~tr_m].sum() / mv))
    x = T.copy(); x["net"] = x.net * w
    mm = P.minute_mdd(x)
    yrs = (T.進場時間.max() - T.進場時間.min()).days / 365
    p(f"  {lab:40s} 平均倍數 {w.mean():.2f} | R：總 {out[0][0]:+6.0f} 比值 訓練 {out[0][2]:4.1f} 驗證 {out[0][3]:4.1f} | "
      f"美元：總 {out[1][0]:+8,.0f} MDD含浮虧 {mm:6,.0f} 年獲利/MDD {out[1][0] / yrs / mm:4.2f} 訓練 {out[1][2]:4.1f} 驗證 {out[1][3]:4.1f}")


p("第五十一部分：什麼狀態下 PF 大幅提升 / 變很醜（RR 3；R 已扣點差 + 0.3 滑價）")
FT, resT, oksT = analyze(T, "A. 最終版 + 防護（實際下單）")
FS, resS, oksS = analyze(SH, "B. 影子訊號（無權益濾網、無防護）")

# 加倉測試：在實際下單上，用「兩個樣本都通過」或「B 通過且 A 同方向」的因子；切點來自各自的訓練期（A）
p("\n=== 加倉測試（實際下單；好五分位 1.5 倍、壞五分位 0.5 倍；切點 = 實際下單的訓練期五分位）===")
evaluate(T, np.ones(len(T)), "不加倉（基準）")
cand = []
for rs in resS:
    ra = next(r for r in resT if r["f"] == rs["f"])
    if rs["ok"] and ra["best_hi"] == rs["best_hi"] and ra["rt"] > 1 and ra["rv"] > 1:
        cand.append(ra)
for r_ in [r for r in resT if r["ok"]] + cand:
    w = np.ones(len(T)); w[r_["q"] == r_["hi"]] = 1.5; w[r_["q"] == r_["lo"]] = 0.5
    evaluate(T, w, f"{r_['f']}（好 1.5 / 壞 0.5）")
    w2 = np.ones(len(T)); w2[r_["q"] == r_["lo"]] = 0.5
    evaluate(T, w2, f"{r_['f']}（只減倉壞狀態）")
    w3 = np.ones(len(T)); w3[r_["q"] == r_["hi"]] = 1.5
    evaluate(T, w3, f"{r_['f']}（只加倉好狀態）")
open("results_part51.txt", "w").write("\n".join(lines) + "\n")

# ---------- 補充：ATR熵變化（兩個樣本、兩段方向都一致：熵下降較好、熵上升最差）----------
p("\n=== 補充：ATR熵變化_M15（熵下降 = 波動開始集中到最近幾根；熵上升 = 波動被攤平）===")
for lab, TT in (("A 實際下單", T), ("B 影子訊號", SH)):
    FF = features(TT)
    v = FF["ATR熵變化_M15"].values
    trm = TT.train.values
    qs = np.quantile(v[trm & np.isfinite(v)], [0.2, 0.4, 0.6, 0.8])
    q = np.digitize(v, qs); q[~np.isfinite(v)] = -1
    R_ = TT.R.values; dd_ = TT.date.values
    worst = R_[~trm & (q == 4)]; rest = R_[~trm & (q >= 0) & (q < 4)]
    ud = np.unique(dd_[~trm]); didx = {d: np.where(dd_ == d)[0] for d in ud}
    diffs = []
    for _ in range(1000):
        ix = np.concatenate([didx[d] for d in rng.choice(ud, len(ud), replace=True)])
        a_, b_ = pf(R_[ix][(q[ix] >= 0) & (q[ix] < 4)]), pf(R_[ix][q[ix] == 4])
        if np.isfinite(a_) and np.isfinite(b_):
            diffs.append(a_ - b_)
    yr = TT.進場時間.dt.year.values
    p(f"  {lab}：熵上升最多的 20%（> {qs[-1]:+.3f}）PF 訓練 {pf(R_[trm & (q == 4)]):.2f} / 驗證 {pf(worst):.2f}；其餘 80% 訓練 {pf(R_[trm & (q < 4) & (q >= 0)]):.2f} / 驗證 {pf(rest):.2f}；"
      f"驗證 PF 差 90% 區間 [{np.quantile(diffs, .05):+.2f}, {np.quantile(diffs, .95):+.2f}]；逐年 PF（最差 20% / 其餘）"
      + " ".join(f"{y}:{pf(R_[(yr == y) & (q == 4)]):.2f}/{pf(R_[(yr == y) & (q < 4) & (q >= 0)]):.2f}" for y in (2023, 2024, 2025, 2026)))
    if TT is T:
        w = np.where(q == 4, 0.5, 1.0)
        evaluate(T, w, "ATR熵上升最多 20% → 減半")
        w = np.where(q == 4, 0.0, 1.0)
        evaluate(T, w, "ATR熵上升最多 20% → 不做")
        w = np.where(q == 0, 1.5, np.where(q == 4, 0.5, 1.0))
        evaluate(T, w, "ATR熵下降最多 ×1.5、上升最多 ×0.5")
open("results_part51.txt", "w").write("\n".join(lines) + "\n")
