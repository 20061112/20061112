"""第五十八部分：系統性、穩健型的優化（不在同一份資料上找最佳參數，而是降低「參數 / 狀態 / 配置」的單點風險）。

基準：IB 核心 + 2R 保本 1 份 + M15 最終版 2 份（份 = 0.01 口），2023/8/24 ~ 2026/10。
測試（全部只用事前資訊；需要門檻的只用訓練期 2023/8~2024/12 決定）：
  1. 參數集成：M15 停損 {1.75, 2.0, 2.25} ATR × RR {2.5, 3, 3.5} 九組各下 2/9 份（不挑最佳的一組，把參數風險平均掉）
     同時列出九組各自的結果（看 2.0 / 3 是不是孤立高點）
  2. 固定風險定口數：每筆口數 ∝ 1 / 停損距離（平均口數不變），以及折衷版 ∝ 1/√停損距離 → 看各年份是否更平均
  3. IB 也加權益濾網（影子累積 R < 50 筆均線就不做）
  4. 整個組合的單日虧損上限：兩策略合計當天已實現虧損 ≥ X 美元就都不開新單（X = 訓練期組合日損益的 2% / 5% 分位）
  5. 配置比例：依訓練期日損益波動的倒數配置（風險平價），總份數 3 份不變
  6. 月區塊自助抽樣：組合 MDD 的分布（實盤該準備多少回撤）
成本：只扣點差，另列 +0.3 滑價。
"""
import numpy as np
import pandas as pd
import part56_noslip_layers_mtf as Q

rng = np.random.default_rng(58)
P, G, IDX, SP = Q.P, Q.G, Q.IDX, Q.SP
TRAIN_END = Q.TRAIN_END
START = pd.Timestamp("2023-08-24")
lines = []
Q.lines = lines


def p(s=""):
    print(s, flush=True)
    lines.append(s)


# ---------------- 建構 ----------------
def sim_e(E, slip, sm, t):
    rows = []
    for r in E.itertuples():
        risk = sm * r.a
        x = G.sim_detail(r.i, r.L, r.side, r.L - r.side * risk, r.L + r.side * t * risk)
        if x is None:
            continue
        j, px, why = x
        net = (px - r.L) * r.side - SP[r.i] - slip
        rows.append((IDX[r.i], IDX[j], r.i, j, r.side, r.L, net, net / risk, risk, r.volrel, r.dent))
    x = pd.DataFrame(rows, columns=["進場時間", "出場時間", "i", "j", "dir", "entry", "net", "R", "risk", "volrel", "dent"])
    return x.sort_values("進場時間").reset_index(drop=True)


def m15(slip, sm=2.0, t=3.0):
    E = Q.EVS[15]
    E = E[E.A & E.B & E.S]
    x = P.cap_total(sim_e(E, slip, sm, t), 8).reset_index(drop=True)
    x = Q.eq_filter(x)
    x = x[~(x.volrel > 2.2).values & ~(x.dent > Q.CUT[15]).values].reset_index(drop=True)
    return Q.daily_stop(x)


Eib = Q.CI.events()


def ib(slip, eq=False):
    X = Q.ib_prep(Q.CI.simulate(Eib, be=2), slip)
    X["進場時間"] = pd.to_datetime(X.t_in)
    X["出場時間"] = IDX[X.j.values]          # ib_prep 已把 j 換成全域 M1 索引
    X["risk"] = np.where(X.side == 1, X.lvl - X.ibl, X.ibh - X.lvl)
    X["R"] = X.net / X.risk
    X = X[X.進場時間 >= START].sort_values("進場時間").reset_index(drop=True)
    return Q.eq_filter(X) if eq else X


def show(lab, items):
    """items: [(df, q 純量或陣列)]"""
    parts = [Q.mpart(x, q) for x, q in items]
    daily = pd.concat([(x.net * q).groupby(x.進場時間.dt.normalize()).sum() for x, q in items], axis=1).fillna(0).sum(1)
    Q.show(lab, parts, daily)
    return daily


def ratio_by_year(daily):
    out = []
    for y in (2023, 2024, 2025, 2026):
        s = daily[pd.DatetimeIndex(daily.index).year == y]
        e = s.cumsum(); m = (e.cummax() - e).max()
        out.append(s.sum() / m if m > 0 else np.nan)
    return out


# ---------------- 4. 組合單日虧損上限 ----------------
def port_cap(items, X):
    rows = []
    for k, (x, q) in enumerate(items):
        qq = np.broadcast_to(q, len(x))
        for n_, (a, b, v) in enumerate(zip(x.進場時間, x.出場時間, x.net.values * qq)):
            rows.append((a, b, v, k, n_))
    rows.sort(key=lambda r: (r[0], r[3]))
    keep = [np.zeros(len(x), bool) for x, q in items]
    cur, real = None, []
    for a, b, v, k, n_ in rows:
        d = a.normalize()
        if d != cur:
            cur, real = d, []
        if sum(vv for (bb, vv) in real if bb <= a) <= -X:
            continue
        keep[k][n_] = True; real.append((b, v))
    return [(x[m].reset_index(drop=True), (np.broadcast_to(q, len(x))[m] if np.ndim(q) else q)) for (x, q), m in zip(items, keep)]



if __name__ == "__main__":
    BASE = {}
    for slip in (0.0, 0.3):
        BASE[slip] = (ib(slip), m15(slip))
    p("第五十八部分：系統性穩健型優化（基準 = IB(2R 保本) 1 份 + M15 2 份；只扣點差，括號或另段為 +0.3 滑價）")
    p("\n=== 0. 基準 ===")
    for slip in (0.0, 0.3):
        I, M = BASE[slip]
        show(f"基準（滑價 {slip}）", [(I, 1), (M, 2)])

    # ---------------- 1. 參數集成 ----------------
    p("\n=== 1. M15 參數集成（停損 1.75/2.0/2.25 × RR 2.5/3/3.5）===")
    ENS = {}
    for slip in (0.0, 0.3):
        p(f"-- 滑價 {slip}：九組各自（M15 2 份單獨）--")
        for sm in (1.75, 2.0, 2.25):
            for t in (2.5, 3.0, 3.5):
                x = m15(slip, sm, t)
                ENS[(slip, sm, t)] = x
                show(f"停損 {sm} RR {t}", [(x, 2)])
        I = BASE[slip][0]
        show(f"單一參數 + IB（滑價 {slip}）", [(I, 1), (BASE[slip][1], 2)])
        show(f"九組集成 + IB（滑價 {slip}）", [(I, 1)] + [(ENS[(slip, sm, t)], 2 / 9) for sm in (1.75, 2.0, 2.25) for t in (2.5, 3.0, 3.5)])

    # ---------------- 2. 固定風險定口數 ----------------
    p("\n=== 2. 口數依停損距離（平均口數不變）===")
    for slip in (0.0, 0.3):
        I, M = BASE[slip]
        for lab, pw in (("固定口數", 0.0), ("∝ 1/√停損", 0.5), ("∝ 1/停損（固定美元風險）", 1.0)):
            qi = (1 / I.risk.values) ** pw; qi = qi / qi.mean()
            qm = (1 / M.risk.values) ** pw; qm = 2 * qm / qm.mean()
            d = show(f"{lab}（滑價 {slip}）", [(I, qi), (M, qm)])
            p(f"      逐年 年獲利/年內MDD：" + " / ".join(f"{v:.1f}" for v in ratio_by_year(d)))

    # ---------------- 3. IB 權益濾網 ----------------
    p("\n=== 3. IB 加權益濾網 ===")
    for slip in (0.0, 0.3):
        I, M = BASE[slip]
        Ie = ib(slip, eq=True)
        show(f"IB 1 單獨（滑價 {slip}）", [(I, 1)])
        show(f"IB 1 + 權益濾網（滑價 {slip}）", [(Ie, 1)])
        show(f"IB(濾網) 1 + M15 2（滑價 {slip}）", [(Ie, 1), (M, 2)])


    p("\n=== 4. 組合單日虧損上限（兩策略合計已實現）===")
    for slip in (0.0, 0.3):
        I, M = BASE[slip]
        d0 = pd.concat([(I.net).groupby(I.進場時間.dt.normalize()).sum(), (M.net * 2).groupby(M.進場時間.dt.normalize()).sum()], axis=1).fillna(0).sum(1)
        tr = d0[d0.index < TRAIN_END]
        for qq in (0.02, 0.05):
            X = -tr.quantile(qq)
            show(f"上限 = 訓練期 {qq:.0%} 分位 {X:,.0f} 美元（滑價 {slip}）", port_cap([(I, 1), (M, 2)], X))

    # ---------------- 5. 風險平價 ----------------
    p("\n=== 5. 配置比例（總 3 份）===")
    I0, M0 = BASE[0.0]
    di = I0.net.groupby(I0.進場時間.dt.normalize()).sum()
    dm = M0.net.groupby(M0.進場時間.dt.normalize()).sum()
    idx = di.index.union(dm.index)
    di, dm = di.reindex(idx, fill_value=0), dm.reindex(idx, fill_value=0)
    trm = idx < TRAIN_END
    wi, wm = 1 / di[trm].std(), 1 / dm[trm].std()
    s = 3 / (wi + wm); wi, wm = wi * s, wm * s
    p(f"  訓練期日損益標準差（每份）：IB {di[trm].std():.1f}、M15 {dm[trm].std():.1f} → 風險平價 IB {wi:.2f} 份 / M15 {wm:.2f} 份")
    for slip in (0.0, 0.3):
        I, M = BASE[slip]
        for lab, a, b in (("IB 1 + M15 2", 1, 2), (f"風險平價 IB {wi:.2f} + M15 {wm:.2f}", wi, wm), ("IB 1.5 + M15 1.5", 1.5, 1.5), ("IB 2 + M15 1", 2, 1)):
            show(f"{lab}（滑價 {slip}）", [(I, a), (M, b)])

    # ---------------- 6. 自助抽樣 MDD ----------------
    p("\n=== 6. 月區塊自助抽樣（2,000 次；組合日損益，日結 MDD）===")
    for slip in (0.0, 0.3):
        I, M = BASE[slip]
        for lab, a, b in (("IB 1 + M15 2", 1, 2), ("IB 2 + M15 2", 2, 2)):
            d = pd.concat([(I.net * a).groupby(I.進場時間.dt.normalize()).sum(), (M.net * b).groupby(M.進場時間.dt.normalize()).sum()], axis=1).fillna(0).sum(1).sort_index()
            mon = pd.DatetimeIndex(d.index).to_period("M")
            blocks = [d[mon == m].values for m in mon.unique()]
            e = d.cumsum(); hist = (e.cummax() - e).max()
            mdds, worst_m, neg_years = [], [], []
            for _ in range(2000):
                seq = np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))])
                ee = np.cumsum(seq); mdds.append((np.maximum.accumulate(ee) - ee).max())
                pick = rng.integers(0, len(blocks), 12)
                neg_years.append(sum(blocks[k].sum() for k in pick) < 0)
            mdds = np.array(mdds)
            mm = np.array([b_.sum() for b_ in blocks])
            p(f"  {lab}（滑價 {slip}）：歷史日結 MDD {hist:,.0f}；抽樣 MDD 中位 {np.median(mdds):,.0f}、90% {np.quantile(mdds, .9):,.0f}、95% {np.quantile(mdds, .95):,.0f}、99% {np.quantile(mdds, .99):,.0f}；"
              f"月虧損機率 {(mm < 0).mean():.0%}、隨機 12 個月合計虧損機率 {np.mean(neg_years):.1%}")
    open("results_part58.txt", "w").write("\n".join(lines) + "\n")
