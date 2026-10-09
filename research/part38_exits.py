"""第三十八部分：OU 均值回歸 — 等權 vs EWMA 權重回歸、出場方式、增加交易量的方法、滾動前推 + M1 執行。

訊號（同第 37 部分）：z = (p - μ*)/σ_eq，μ*、σ_eq 由最近 L 根 AR(1) 回歸估出（ou = 等權；ou_ew = EWMA 權重 span L）。
|z| 首次 ≥ thr 的那根收盤後，下一根開盤往 μ* 方向進場。風險單位 R = 1.5 ATR14（與停損倍數無關，方便比較）。

出場方式（generic_sim 參數）
  sl     停損 = 進場價 ∓ sl×ATR
  tgt    ("mu0",)    碰到進場時的 μ*（收盤越過）
         ("z", lvl)  動態 z 回到 lvl（1 = 回一半、0 = 回到目前均值、-1 = 衝過頭 1σ）
         ("R", m)    固定停利 m×R（觸價成交）
         None        不設目標
  hold   最多持有幾根（原週期）
  trail  移動停損：最有利價位回吐 trail×ATR
  be     浮盈達 1R 後停損移到進場價
同一根 K 棒：先檢查停損（含移動 / 保本），再檢查觸價停利，最後檢查收盤條件 → 保守。

增加交易量
  多週期：同一時間尺度在 M5（L×3）、M15（L）、H1（L÷4）同時做
  分層：thr 2 / 2.5 / 3 各自一筆（可同時持有）
  重複進場：只要 |z| ≥ thr 且空手就進（不必重新穿越），出場後冷卻 4 根
"""
import numpy as np
import pandas as pd
from load import load
from spring import resample, atr
from part37_wf import zscore, SPLIT

RISK = 1.5
EXITS = {
    "基準 μ0 sl1.5 H40": dict(sl=1.5, tgt=("mu0",), hold=40),
    "μ0 sl1.0 H40": dict(sl=1.0, tgt=("mu0",), hold=40),
    "μ0 sl2.0 H40": dict(sl=2.0, tgt=("mu0",), hold=40),
    "μ0 sl3.0 H40": dict(sl=3.0, tgt=("mu0",), hold=40),
    "μ0 sl1.5 H20": dict(sl=1.5, tgt=("mu0",), hold=20),
    "μ0 sl1.5 H80": dict(sl=1.5, tgt=("mu0",), hold=80),
    "z回一半 sl1.5 H40": dict(sl=1.5, tgt=("z", 1.0), hold=40),
    "z回均值(動態) sl1.5 H40": dict(sl=1.5, tgt=("z", 0.0), hold=40),
    "z衝過頭1σ sl1.5 H80": dict(sl=1.5, tgt=("z", -1.0), hold=80),
    "停利1R": dict(sl=1.5, tgt=("R", 1.0), hold=40),
    "停利2R": dict(sl=1.5, tgt=("R", 2.0), hold=40),
    "停利3R": dict(sl=1.5, tgt=("R", 3.0), hold=80),
    "移動停損2ATR 無目標 H80": dict(sl=1.5, tgt=None, hold=80, trail=2.0),
    "移動停損3ATR + μ0 H80": dict(sl=1.5, tgt=("mu0",), hold=80, trail=3.0),
    "保本(1R) + μ0 H40": dict(sl=1.5, tgt=("mu0",), hold=40, be=True),
    "只靠時間 H20 sl1.5": dict(sl=1.5, tgt=None, hold=20),
    "只靠時間 H40 sl5(災難停損)": dict(sl=5.0, tgt=None, hold=40),
}
BASE = "基準 μ0 sl1.5 H40"


def generic_sim(o, h, l, c, z, entries, p, cooldown=0, unit=1):
    """entries: 依時間排序的 (k 進場 K 棒, dir, atr, mu0)。z：每根可用的最新 z（已對齊）。
    unit：一根原週期 = 幾根執行 K 棒（M1 執行時 > 1）。回傳 list of dict。"""
    n = len(c)
    out = []
    busy = -1
    sl, tgt, hold = p["sl"], p.get("tgt"), p["hold"] * unit
    trail, be = p.get("trail"), p.get("be", False)
    for k, d, a, mu0 in entries:
        if k >= n or k <= busy or np.isnan(a) or np.isnan(mu0):
            continue
        e = o[k]
        if tgt and tgt[0] == "mu0" and d * (mu0 - e) <= 0:
            continue
        risk = RISK * a
        stop = e - d * sl * a
        tp = e + d * tgt[1] * risk if tgt and tgt[0] == "R" else None
        best = e
        x = j_out = None
        why = "time"
        for j in range(k, min(k + hold, n)):
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                x, j_out, why = stop, j, "stop"
                break
            if tp is not None and ((h[j] >= tp) if d == 1 else (l[j] <= tp)):
                x, j_out, why = tp, j, "tp"
                break
            if tgt and tgt[0] == "mu0" and d * (c[j] - mu0) >= 0:
                x, j_out, why = c[j], j, "tp"
                break
            if tgt and tgt[0] == "z" and not np.isnan(z[j]) and -d * z[j] <= tgt[1]:
                x, j_out, why = c[j], j, "tp"
                break
            best = max(best, h[j]) if d == 1 else min(best, l[j])
            if trail:
                stop = max(stop, best - trail * a) if d == 1 else min(stop, best + trail * a)
            if be and d * (best - e) >= risk:
                stop = max(stop, e) if d == 1 else min(stop, e)
        if j_out is None:
            j_out = min(k + hold, n) - 1
            x = c[j_out]
        busy = j_out + cooldown * unit
        out.append(dict(k=k, exit=j_out, dir=d, pnl_atr=(x - e) * d / a, why=why, bars=(j_out - k + 1) / unit))
    return out


class TF:
    def __init__(self, m1, tf):
        self.tf = tf
        self.mins = {"5min": 5, "15min": 15, "1h": 60}[tf]
        self.df = resample(m1, tf)
        self.A = atr(self.df).to_numpy()
        self.spr = self.df.spread.to_numpy(float) * 0.01
        self.arr = [self.df[k].to_numpy(float) for k in ("open", "high", "low", "close")]
        self.cache = {}

    def z(self, method, L):
        if (method, L) not in self.cache:
            self.cache[(method, L)] = zscore(self.df, method, L)
        return self.cache[(method, L)]

    def entries(self, method, L, thr, reentry=False):
        z, mu = self.z(method, L)
        za = np.abs(z)
        if reentry:
            i = np.where(za >= thr)[0]
        else:
            prev = np.r_[np.nan, za[:-1]]
            i = np.where((za >= thr) & (prev < thr))[0]
        i = i[i + 1 < len(z)]
        return [(t + 1, -int(np.sign(z[t])), self.A[t], mu[t]) for t in i]

    def run(self, method, L, thr, exit_name, reentry=False, cooldown=0):
        z, mu = self.z(method, L)
        E = self.entries(method, L, thr, reentry)
        res = generic_sim(*self.arr, z, E, EXITS[exit_name], cooldown=cooldown if reentry else 0)
        idx = self.df.index
        rows = []
        for r in res:
            k = r["k"]
            a = self.A[k - 1]
            rows.append(dict(cfg=f"{self.tf}|{method}|{L}|{thr}|{exit_name}" + ("|re" if reentry else ""),
                             tf=self.tf, method=method, L=L, thr=thr, exit=exit_name,
                             t_sig=idx[k - 1], t_entry=idx[k], t_exit=idx[r["exit"]] + pd.Timedelta(minutes=self.mins),
                             dir=r["dir"], atr=a, mu0=mu[k - 1], why=r["why"], bars=r["bars"],
                             R=(r["pnl_atr"] * a - self.spr[k]) / (RISK * a)))
        return rows


def stat(r, months=7.2):
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    if len(r) < 3:
        return dict(n=len(r))
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.inf
    eq = np.cumsum(r)
    dd = (eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
    return dict(n=len(r), per_m=len(r) / months, avg=r.mean(), t=r.mean() / r.std(ddof=1) * np.sqrt(len(r)),
                win=np.mean(r > 0), pf=pf, sum=r.sum(), dd=dd)


def fmt(s, lab, extra=""):
    if s["n"] < 3:
        return f"  {lab:<36} n={s['n']}"
    return (f"  {lab:<36} n={s['n']:4d}（{s['per_m']:4.1f}/月） avg={s['avg']:+.3f}R t={s['t']:+.1f} "
            f"win={s['win']:.0%} PF={s['pf']:.2f} 總和={s['sum']:+6.1f}R 回撤={s['dd']:+.1f}R{extra}")


def halves(T):
    a = T[T.t_sig < SPLIT].R.mean()
    b = T[T.t_sig >= SPLIT].R.mean()
    return f" | 前半 {a:+.3f} 後半 {b:+.3f}"


def m1_exec(m1, T, frames):
    """把交易表（含 tf/method/L/thr/exit）改用 M1 K 棒重跑；全組合同時最多 1 筆。"""
    M = m1.index
    o, h, l, c = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    spr = m1.spread.to_numpy(float) * 0.01
    zmap = {}
    R = np.full(len(T), np.nan)
    busy = pd.Timestamp("1970-01-01")
    for q, s in enumerate(T.itertuples()):
        if s.t_entry < busy:
            continue
        F = frames[s.tf]
        key = (s.tf, s.method, s.L)
        if key not in zmap:     # 原週期 K 棒收盤後才可用 → 對齊到 M1
            z, _ = F.z(s.method, s.L)
            zs = pd.Series(z, index=F.df.index + pd.Timedelta(minutes=F.mins))
            zmap[key] = zs[~zs.index.duplicated()].reindex(M, method="ffill").to_numpy()
        k0 = M.searchsorted(s.t_entry)
        if k0 >= len(M) or (M[k0] - s.t_entry) > pd.Timedelta(minutes=30):
            continue
        res = generic_sim(o, h, l, c, zmap[key], [(k0, s.dir, s.atr, s.mu0)], EXITS[s.exit], unit=F.mins)
        if not res:
            continue
        r = res[0]
        R[q] = (r["pnl_atr"] * s.atr - spr[k0]) / (RISK * s.atr)
        busy = M[r["exit"]] + pd.Timedelta(minutes=1)
    return R


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    frames = {tf: TF(m1, tf) for tf in ("5min", "15min", "1h")}
    F15 = frames["15min"]
    lines = []
    P = lines.append

    # ---------- A. 等權 vs EWMA 權重（M15、長視窗、高門檻） ----------
    P("=" * 130)
    P("A. OU 回歸視窗：等權（ou，像 SMA）vs EWMA 權重（ou_ew，span L）；M15，基準出場")
    for method, Ls in (("ou", (200, 300, 400, 600)), ("ou_ew", (200, 300, 400, 600, 800, 1200))):
        for L in Ls:
            for thr in (2.0, 2.5, 3.0):
                T = pd.DataFrame(F15.run(method, L, thr, BASE))
                P(fmt(stat(T.R), f"{method} L={L} thr={thr}", halves(T)))

    # ---------- B. 出場方式 ----------
    P("\n" + "=" * 130)
    P("B. 出場方式（M15；ou L200/400 + ou_ew L400/800；thr 2.5 / 3 → 8 組合併）；pnl 統一以 1.5 ATR 為 1R")
    B = []
    for ex in EXITS:
        rows = []
        for method, L in (("ou", 200), ("ou", 400), ("ou_ew", 400), ("ou_ew", 800)):
            for thr in (2.5, 3.0):
                rows += F15.run(method, L, thr, ex)
        T = pd.DataFrame(rows)
        B.append(T)
        why = T.why.value_counts(normalize=True)
        P(fmt(stat(T.R), ex, halves(T) + f" | 平均持有 {T.bars.mean():4.1f} 根 停損 {why.get('stop', 0):.0%} "
                                          f"目標 {why.get('tp', 0):.0%} 時間 {why.get('time', 0):.0%}"))
    P("\n  各出場方式在 8 組裡為正的組數（看穩定度）")
    for T in B:
        g = T.groupby("cfg").R.mean()
        P(f"    {T.exit.iloc[0]:<30} {int((g > 0).sum())}/{len(g)} 組為正，最差 {g.min():+.3f}R 最好 {g.max():+.3f}R")

    # ---------- C. 增加交易量 ----------
    P("\n" + "=" * 130)
    P("C. 增加交易量（基準出場；ou L400 thr3 為單一基準）")
    base = pd.DataFrame(F15.run("ou", 400, 3.0, BASE))
    P(fmt(stat(base.R), "單一：M15 ou L400 thr3", halves(base)))
    for L in (200, 400):
        for thr in (2.5, 3.0):
            mt = pd.concat([pd.DataFrame(frames["5min"].run("ou", L * 3, thr, BASE)),
                            pd.DataFrame(F15.run("ou", L, thr, BASE)),
                            pd.DataFrame(frames["1h"].run("ou", L // 4, thr, BASE))])
            P(fmt(stat(mt.sort_values("t_entry").R), f"多週期 M5 L{L * 3}/M15 L{L}/H1 L{L // 4} thr{thr}", halves(mt)))
            for tf in ("5min", "1h"):
                s = mt[mt.tf == tf]
                P(fmt(stat(s.R), f"    其中 {tf}", halves(s)))
    for L in (200, 400):
        lay = pd.concat([pd.DataFrame(F15.run("ou", L, thr, BASE)) for thr in (2.0, 2.5, 3.0)])
        P(fmt(stat(lay.sort_values("t_entry").R), f"分層 thr2/2.5/3 各一筆 ou L{L}", halves(lay)))
    for L in (200, 400):
        for thr in (2.5, 3.0):
            re = pd.DataFrame(F15.run("ou", L, thr, BASE, reentry=True, cooldown=4))
            P(fmt(stat(re.R), f"重複進場 ou L{L} thr{thr} 冷卻4根", halves(re)))
    allm = pd.concat([pd.DataFrame(F15.run(m, L, thr, BASE)) for m, L in (("ou", 200), ("ou", 400), ("ou_ew", 400), ("ou_ew", 800))
                      for thr in (2.5, 3.0)])
    P(fmt(stat(allm.sort_values("t_entry").R), "8 組 M15 全部一起做（可重疊）", halves(allm)))

    # ---------- D. 滾動前推 + M1 ----------
    P("\n" + "=" * 130)
    P("D. 滾動前推：池子 = M15 × {ou L200/300/400/600, ou_ew L400/600/800/1200} × thr{2.5,3} × 17 種出場 = 272 組")
    pool = []
    for method, Ls in (("ou", (200, 300, 400, 600)), ("ou_ew", (400, 600, 800, 1200))):
        for L in Ls:
            for thr in (2.5, 3.0):
                for ex in EXITS:
                    pool += F15.run(method, L, thr, ex)
    pool = pd.DataFrame(pool)
    months = pd.date_range("2026-03-01", "2026-10-01", freq="MS")

    def walk(pool, k):
        sel, chosen = [], []
        for m in months:
            past = pool[pool.t_exit < m]
            s = past.groupby("cfg").R.agg(["mean", "size"])
            s = s[s["size"] >= 20]
            s["score"] = s["mean"] * np.sqrt(s["size"])
            cf = list(s[s.score > 0].sort_values("score", ascending=False).index[:k])
            nxt = m + pd.offsets.MonthBegin(1)
            sel.append(pool[pool.cfg.isin(cf) & (pool.t_sig >= m) & (pool.t_sig < nxt)])
            chosen.append((m.strftime("%Y-%m"), cf))
        return pd.concat(sel).sort_values("t_entry").reset_index(drop=True), chosen

    for lab, pl, k in (("Top1（全部出場可選）", pool, 1), ("Top3", pool, 3),
                       ("Top1（只用基準出場）", pool[pool.exit == BASE], 1),
                       ("Top1（只選 ou 等權）", pool[pool.method == "ou"], 1),
                       ("Top1（只選 ou_ew）", pool[pool.method == "ou_ew"], 1)):
        S, ch = walk(pl, k)
        R1 = m1_exec(m1, S, frames)
        P(fmt(stat(S.R), lab + " 原週期"))
        P(fmt(stat(R1), lab + " M1 執行"))
        if lab.startswith("Top1（全部"):
            for mo, cf in ch:
                P(f"      {mo}: {cf}")
            S["R_m1"] = R1
            S.to_csv("part38_wf_trades.csv", index=False)
            r = S.R_m1.dropna().sort_values()
            P(f"      拿掉最好 3 筆：avg {r.iloc[:-3].mean():+.3f}R；逐月 " +
              " ".join(f"{mo}:{v:+.1f}" for mo, v in S.groupby(S.t_sig.dt.strftime('%m')).R_m1.sum().items()))
    txt = "\n".join(lines)
    print(txt)
    open("results_part38.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
