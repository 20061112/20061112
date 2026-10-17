"""第二十二部分：QQQ↔3 倍槓桿定投策略 —— 用 Nasdaq-100 期貨執行的組合驗證。

A. 重現報告數字（確認訊號實作正確）
B. 執行組合網格：結構（現股+期貨借款 / 現股+期貨賣股補保證金 / 全期貨）× 槓桿 × 重設頻率
C. 成本與融資敏感度（basis、借款利差、保證金緩衝）
D. 合約顆粒度：以「今天的 MNQ/NQ 合約大小」起算，不同帳戶規模的 17 段 10 年
E. 時機是否真的有用（沿用第十九部分的隨機對照）：固定槓桿基準、隨機打亂開關槓桿時段
F. 過擬合機率 PBO（沿用第二十一部分的 CSCV）：在執行組合裡挑最好的，樣本外還好嗎
G. 壓力測試：單日 −20%、每日結算現金流出
H. 逐年表現
I. 目前訊號（part22_live.json）
"""
import itertools
import json
import numpy as np
import pandas as pd
from part22_core import load, signals, run, metrics, year_starts, TODAY_MNQ, CONTRACTS, NDX_PER_QQQ

S = signals(load())
YS = year_starts(S)
I2010 = S.index.searchsorted(pd.Timestamp("2010-02-11"))
WIN = list(range(2000, 2017))
out = open("results_part22.txt", "w")


def P(s=""):
    print(s); out.write(s + "\n"); out.flush()


def windows(**kw):
    return [metrics(run(S, i0=YS[y], i1=YS[y + 10], **kw)) for y in WIN]


def pct(x, d=1):
    return f"{x * 100:+.{d}f}%"


REF = {"dca": windows(eng="dca"), "etf": windows(eng="etf")}
FULL = {k: metrics(run(S, eng=k)) for k in ("dca", "etf")}
F2010 = {k: metrics(run(S, eng=k, i0=I2010)) for k in ("dca", "etf")}

# ---------------------------------------------------------------- A
P("第二十二部分：QQQ↔3 倍槓桿定投 —— 期貨執行的組合驗證")
P(f"資料 {S.index[0].date()} ~ {S.index[-1].date()}（{len(S)} 日）；每週投入 1,000（布林加碼時 2,000）\n")
P("A. 重現報告數字")
P(f"  {'':22s}{'報告':>10s}{'本次':>10s}")
for lab, doc, mine in [("純週 DCA IRR 1999 起", 0.151, FULL["dca"]["irr"]), ("QQQ↔TQQQ IRR 1999 起", 0.269, FULL["etf"]["irr"]),
                       ("純週 DCA IRR 2010 起", 0.195, F2010["dca"]["irr"]), ("QQQ↔TQQQ IRR 2010 起", 0.360, F2010["etf"]["irr"]),
                       ("10 年平均總損益 DCA", 1.29, np.mean([m["tot"] for m in REF["dca"]])),
                       ("10 年平均總損益 策略", 4.56, np.mean([m["tot"] for m in REF["etf"]])),
                       ("10 年最差一段 策略", 0.83, np.min([m["tot"] for m in REF["etf"]])),
                       ("最慘相對本金 策略", -0.52, FULL["etf"]["worst"])]:
    P(f"  {lab:22s}{doc * 100:>9.1f}%{mine * 100:>9.1f}%")
P(f"  開槓桿時間占比 {S.on.mean() * 100:.1f}%，平均槓桿 {FULL['etf']['avg_lev']:.2f} 倍，每年換倉 {FULL['etf']['sw_per_yr']:.1f} 次")
P("  （2010 起為合成 3 倍 ETF；報告的真實 TQQQ 與合成版差 0.2%，以下全部用合成版比較）")

# ---------------------------------------------------------------- B
STRUCT = {"現股+期貨/借款": dict(eng="fut", fund="borrow"), "現股+期貨/賣股補": dict(eng="fut", fund="sell"),
          "全期貨": dict(eng="allfut")}
REB = {"每日": dict(rebal="daily"), "每週": dict(rebal="weekly"), "偏離±0.25": dict(rebal="band", band=0.25),
       "偏離±0.5": dict(rebal="band", band=0.5), "只在進場": dict(rebal="entry")}
LEVS = [2.0, 2.5, 3.0]
grid, rows = {}, []
for (sn, skw), L, (rn, rkw) in itertools.product(STRUCT.items(), LEVS, REB.items()):
    kw = dict(L=L, **skw, **rkw)
    full = run(S, **kw)
    mf, m10 = metrics(full), metrics(run(S, i0=I2010, **kw))
    w = windows(**kw)
    tots = np.array([x["tot"] for x in w])
    grid[(sn, L, rn)] = dict(kw=kw, twr=full["twr"])
    rows.append(dict(結構=sn, 槓桿=L, 重設=rn, irr99=mf["irr"], irr10=m10["irr"], 平均10年=tots.mean(), 最差10年=tots.min(),
                     勝ETF段=int((tots > np.array([x["tot"] for x in REF["etf"]])).sum()),
                     勝DCA段=int((tots > np.array([x["tot"] for x in REF["dca"]])).sum()),
                     最慘相對本金=mf["worst"], 平均槓桿=mf["avg_lev"], 期貨帳戶最低=mf["min_acct"], 成本占期末=mf["cost_pct"]))
G = pd.DataFrame(rows)
G.to_csv("part22_grid.csv", index=False)
P("\nB. 執行組合網格（basis 0.25%、借款利率 T-bill+1%、可分割合約；17 段 10 年從零定投）")
P(f"  基準：DCA IRR {pct(FULL['dca']['irr'])} / 2010 起 {pct(F2010['dca']['irr'])} / 10 年平均 {np.mean([m['tot'] for m in REF['dca']]) * 100:+.0f}%；"
  f"QQQ↔TQQQ {pct(FULL['etf']['irr'])} / {pct(F2010['etf']['irr'])} / {np.mean([m['tot'] for m in REF['etf']]) * 100:+.0f}%"
  f"（最差段 {np.min([m['tot'] for m in REF['etf']]) * 100:+.0f}%、最慘相對本金 {pct(FULL['etf']['worst'], 0)}）")
P(f"  {'結構':16s}{'L':>4s} {'重設':10s}{'IRR99':>8s}{'IRR10':>8s}{'10年均':>8s}{'10年最差':>8s}{'勝ETF':>6s}{'勝DCA':>6s}"
  f"{'最慘/本金':>9s}{'均槓桿':>7s}{'帳戶最低':>8s}{'成本/期末':>9s}")
for r in rows:
    P(f"  {r['結構']:16s}{r['槓桿']:>4.1f} {r['重設']:10s}{r['irr99'] * 100:>7.1f}%{r['irr10'] * 100:>7.1f}%{r['平均10年'] * 100:>7.0f}%"
      f"{r['最差10年'] * 100:>7.0f}%{r['勝ETF段']:>5d}/17{r['勝DCA段']:>3d}/17{r['最慘相對本金'] * 100:>8.0f}%{r['平均槓桿']:>7.2f}"
      f"{r['期貨帳戶最低'] * 100:>7.0f}%{r['成本占期末'] * 100:>8.2f}%")
g3 = G[G.槓桿 == 3.0]
P(f"\n  L=3 的 15 組：IRR 1999 起 {g3.irr99.min() * 100:.1f}% ~ {g3.irr99.max() * 100:.1f}%（QQQ↔TQQQ {FULL['etf']['irr'] * 100:.1f}%）；"
  f"10 年段勝過 QQQ↔TQQQ 的中位數 {g3.勝ETF段.median():.0f}/17；全部 45 組勝 DCA 段數最少 {G.勝DCA段.min()}/17")

# ---------------------------------------------------------------- C
P("\nC. 成本與融資敏感度（全期貨 L=3 每週重設；現股+期貨 L=3 每週重設）")
for lab, kw in [("全期貨", dict(eng="allfut", L=3.0)), ("現股+期貨/借款", dict(eng="fut", L=3.0, fund="borrow"))]:
    s = []
    for b in (0.0, 0.0025, 0.005, 0.01):
        s.append(f"basis {b * 100:.2f}% → {metrics(run(S, basis=b, **kw))['irr'] * 100:.1f}%")
    P(f"  {lab}：" + "；".join(s))
P("  現股+期貨/借款，借款利差：" + "；".join(
    f"T-bill+{sp * 100:.1f}% → {metrics(run(S, eng='fut', L=3.0, spread=sp))['irr'] * 100:.1f}%" for sp in (0.005, 0.01, 0.02, 0.03)))
P("  現股+期貨/賣股補，保證金緩衝：" + "；".join(
    f"{b:.1f}× → {metrics(run(S, eng='fut', fund='sell', L=3.0, buffer=b))['irr'] * 100:.1f}%" for b in (1.0, 1.5, 2.0, 3.0)))
P("  （QQQ↔TQQQ 隱含的融資 = 2 × (T-bill + 0.45%) + 0.95% 管理費；期貨約 2 × (T-bill + basis)，再加上每季轉倉成本）")

# ---------------------------------------------------------------- D
P("\nD. 合約顆粒度：以今天的合約大小起算（MNQ 名目約 {:,.0f}、NQ 約 {:,.0f} 美元，之後隨價格變動）".format(TODAY_MNQ, TODAY_MNQ * 10))
P("   17 段 10 年歷史路徑；L=3、每週重設；與『可分割合約』比較 10 年總損益與實際槓桿誤差")
ACC = [(10_000, 250), (30_000, 500), (60_000, 1_000), (150_000, 2_000), (500_000, 5_000)]
P(f"  {'起始/每週':>16s} {'結構':14s}{'合約':>6s}{'10年均':>9s}{'與理想差':>9s}{'開槓桿時實際槓桿':>18s}")
for (lump, wk), (sl, skw) in itertools.product(ACC, [("全期貨", dict(eng="allfut")), ("現股+期貨/賣股補", dict(eng="fut", fund="sell"))]):
    ideal = None
    for con in (None, "MNQ", "NQ"):
        tots, levs = [], []
        for y in WIN:
            kw = dict(L=3.0, base=wk, lump=lump, i0=YS[y], i1=YS[y + 10], **skw)
            if con:
                mult = CONTRACTS[con][0]
                kw.update(contract=con, start_notional=TODAY_MNQ * mult / 2.0)
            else:
                kw.update(start_notional=TODAY_MNQ)
            o = run(S, **kw)
            tots.append(metrics(o)["tot"])
            on = S.on.values[YS[y]:YS[y + 10]]
            levs.append(o["lev"][on])
        lv = np.concatenate(levs)
        t = np.mean(tots)
        ideal = t if con is None else ideal
        P(f"  {lump:>8,.0f}/{wk:<6,d} {sl:14s}{(con or '理想'):>6s}{t * 100:>8.0f}%{(t - ideal) * 100:>8.0f}%"
          f"   中位 {np.median(lv):.2f}，10–90% {np.percentile(lv, 10):.2f}~{np.percentile(lv, 90):.2f}")

# ---------------------------------------------------------------- E
P("\nE. 時機是否有用（第十九部分的隨機對照做法）")
avg_lev = metrics(run(S, eng="allfut", L=3.0))["avg_lev"]
base_kw = dict(eng="allfut", L=3.0)
act = metrics(run(S, **base_kw))
const = {}
for lv in (avg_lev, 1.5, 2.0, 2.5):
    c = metrics(run(S, eng="allfut", L=lv, rebal="daily", regime_arr=np.ones(len(S), bool)))
    cw = metrics(run(S, eng="allfut", L=lv, rebal="weekly", regime_arr=np.ones(len(S), bool)))
    const[lv] = c
    P(f"  固定 {lv:.2f} 倍期貨（每天/每週重設）：IRR {c['irr'] * 100:.1f}% / {cw['irr'] * 100:.1f}%，最慘相對本金 {pct(c['worst'], 0)} / {pct(cw['worst'], 0)}")
P(f"  策略（全期貨 L=3 每週，平均槓桿 {avg_lev:.2f}）：IRR {act['irr'] * 100:.1f}%，最慘相對本金 {pct(act['worst'], 0)}")
on = S.on.values.astype(int)
edges = np.flatnonzero(np.diff(np.r_[0, on, 0]))
on_runs = list(edges[1::2] - edges[::2])
off_runs = list(np.diff(np.r_[0, edges])[0::2])
off_runs = [r for r in off_runs if r > 0]
rng = np.random.default_rng(22)
rand_irr, rand_w = [], []
for _ in range(400):
    a, b = rng.permutation(on_runs), rng.permutation(off_runs)
    seq = []
    for k in range(max(len(a), len(b))):
        if k < len(b): seq += [False] * b[k]
        if k < len(a): seq += [True] * a[k]
    reg = np.array(seq[:len(S)] + [False] * max(0, len(S) - len(seq)))
    m = metrics(run(S, regime_arr=reg, **base_kw))
    rand_irr.append(m["irr"]); rand_w.append(m["worst"])
rand_irr, rand_w = np.array(rand_irr), np.array(rand_w)
P(f"  隨機打亂開/關槓桿的時段（保留每段長度與總時間，400 次）：IRR 中位 {np.median(rand_irr) * 100:.1f}%"
  f"（5–95% {np.percentile(rand_irr, 5) * 100:.1f}% ~ {np.percentile(rand_irr, 95) * 100:.1f}%），"
  f"最慘相對本金中位 {pct(np.median(rand_w), 0)}")
P(f"  → 策略的 IRR 勝過 {(rand_irr < act['irr']).mean() * 100:.1f}% 的隨機時段；最慘相對本金好過 {(rand_w < act['worst']).mean() * 100:.1f}%")
on_r, off_r = S.ret.values[1:][S.on.values[:-1]], S.ret.values[1:][~S.on.values[:-1]]
P(f"  開槓桿時段 QQQ 日報酬：平均 {on_r.mean() * 252 * 100:.1f}%/年、波動 {on_r.std() * np.sqrt(252) * 100:.1f}%；"
  f"關槓桿時段：{off_r.mean() * 252 * 100:.1f}%/年、波動 {off_r.std() * np.sqrt(252) * 100:.1f}%")

# ---------------------------------------------------------------- F
P("\nF. 過擬合機率（第二十一部分的 CSCV；45 組執行方式的每日時間加權報酬，16 塊、12,870 種切法）")
keys = list(grid)
M = np.array([grid[k]["twr"] for k in keys]).T[260:]
M = np.log1p(M)
qqq = np.log1p(S.ret.values[260:])
blocks = np.array_split(np.arange(len(M)), 16)
bs = np.array([M[b].sum(0) for b in blocks]); bq = np.array([(M[b] ** 2).sum(0) for b in blocks]); bn = np.array([len(b) for b in blocks])
qs = np.array([qqq[b].sum() for b in blocks]); qq2 = np.array([(qqq[b] ** 2).sum() for b in blocks])
etf_twr = np.log1p(run(S, eng="etf")["twr"][260:])
es = np.array([etf_twr[b].sum() for b in blocks]); eq2 = np.array([(etf_twr[b] ** 2).sum() for b in blocks])


def sharpe(s, q2, n):
    m = s / n; v = q2 / n - m ** 2
    return m / np.sqrt(np.maximum(v, 1e-18)) * np.sqrt(252)


logits, beat_q, beat_e, rank_pick = [], [], [], []
pick = keys.index(("全期貨", 3.0, "每週"))
for ins in itertools.combinations(range(16), 8):
    i = np.zeros(16, bool); i[list(ins)] = True; o = ~i
    s_in = sharpe(bs[i].sum(0), bq[i].sum(0), bn[i].sum()); s_out = sharpe(bs[o].sum(0), bq[o].sum(0), bn[o].sum())
    best = int(np.argmax(s_in))
    w = (s_out < s_out[best]).sum() / (len(keys) - 1)
    w = min(max(w, 1e-6), 1 - 1e-6)
    logits.append(np.log(w / (1 - w)))
    beat_q.append(s_out[best] > sharpe(qs[o].sum(), qq2[o].sum(), bn[o].sum()))
    beat_e.append(s_out[best] > sharpe(es[o].sum(), eq2[o].sum(), bn[o].sum()))
    rank_pick.append((s_out < s_out[pick]).sum() / (len(keys) - 1))
logits = np.array(logits)
P(f"  PBO = {(logits <= 0).mean() * 100:.0f}%（在 45 種執行方式裡挑樣本內最好的，樣本外落到後半的機率）")
P(f"  樣本內最佳在樣本外：Sharpe 勝 QQQ {np.mean(beat_q) * 100:.0f}%、勝 QQQ↔TQQQ {np.mean(beat_e) * 100:.0f}%")
P(f"  預設選擇（全期貨 L=3 每週）在樣本外的平均百分位 {np.mean(rank_pick) * 100:.0f}%")
sh_all = sharpe(bs.sum(0), bq.sum(0), bn.sum())
P(f"  全期 Sharpe：45 組 {sh_all.min():.2f} ~ {sh_all.max():.2f}，QQQ {sharpe(qs.sum(), qq2.sum(), bn.sum()):.2f}，"
  f"QQQ↔TQQQ {sharpe(es.sum(), eq2.sum(), bn.sum()):.2f}")
P("  全期 Sharpe 前 5 名：" + "；".join(f"{keys[j][0]} L{keys[j][1]} {keys[j][2]} {sh_all[j]:.2f}" for j in np.argsort(-sh_all)[:5]))
for L_ in LEVS:
    idx = [j for j, k in enumerate(keys) if k[1] == L_]
    P(f"  L={L_}：Sharpe {sh_all[idx].min():.2f} ~ {sh_all[idx].max():.2f}")

# ---------------------------------------------------------------- G
P("\nG. 壓力測試")
P("  G1 單日 −20%（1987 型）發生在持有槓桿的一天，從那天起之後的價格都下移 20%：")
for dt in ("2020-02-19", "2021-11-19", "2024-07-10"):
    i = S.index.searchsorted(pd.Timestamp(dt))
    if not S.on.values[i - 1]:
        continue
    for lab, kw in [("QQQ↔TQQQ", dict(eng="etf")), ("全期貨 L=3", dict(eng="allfut", L=3.0)),
                    ("現股+期貨/借款 L=3", dict(eng="fut", L=3.0)), ("現股+期貨/賣股補 L=3", dict(eng="fut", fund="sell", L=3.0)),
                    ("全期貨 L=2", dict(eng="allfut", L=2.0))]:
        base_o, sh_o = run(S, **kw), run(S, shock=(i, 0.20), **kw)
        d1 = sh_o["V"].iloc[i] / base_o["V"].iloc[i - 1] - 1
        P(f"    {dt} {lab:20s}當日淨值 {pct(d1, 0)}；全期 IRR {metrics(base_o)['irr'] * 100:.1f}% → {metrics(sh_o)['irr'] * 100:.1f}%")
P("    （CME 股指期貨有 7%/13%/20% 熔斷與夜盤 ±7% 漲跌停；實際可能在跌到 −7% 時就被鎖住無法出場）")

P("  G2 每日結算的現金流出（現股+期貨/借款 L=3）：期貨帳戶相對總淨值")
o = run(S, eng="fut", L=3.0)
lev = pd.Series(o["lev"], index=S.index)
fut_frac = (lev - 1).clip(lower=0)
flow = fut_frac.shift(1) * S.ret
for n_ in (1, 5, 20):
    x = flow.rolling(n_).sum()
    P(f"    最差 {n_:>2d} 日結算 {x.min() * 100:+.1f}% 淨值（{x.idxmin().date()}）；99% 分位 {x.quantile(0.01) * 100:+.1f}%")
P(f"    期貨帳戶最低曾到淨值的 {o['min_acct'] * 100:.0f}%（需借款或賣股補上）")
mrq = (S.mrate * fut_frac * 1.0)[S.on]
P(f"    開槓桿時期貨保證金需求（估）中位 {mrq.median() * 100:.1f}% 淨值、最高 {mrq.max() * 100:.1f}%")

# ---------------------------------------------------------------- H
P("\nH. 逐年時間加權報酬（1999 起同一帳戶）")
yr = {}
for lab, kw in [("QQQ", dict(eng="dca")), ("QQQ↔TQQQ", dict(eng="etf")), ("全期貨L3週", dict(eng="allfut", L=3.0)),
                ("全期貨L2.5週", dict(eng="allfut", L=2.5)), ("現股+期貨L3週", dict(eng="fut", L=3.0))]:
    tw = pd.Series(run(S, **kw)["twr"], index=S.index)
    yr[lab] = (1 + tw).groupby(S.index.year).prod() - 1
Y = pd.DataFrame(yr)
P("  年份 " + "".join(f"{c:>14s}" for c in Y.columns))
for y, r in Y.iterrows():
    P(f"  {y} " + "".join(f"{v * 100:>13.0f}%" for v in r.values))
P(f"  全期貨L3 勝 QQQ↔TQQQ 的年份：{(Y['全期貨L3週'] > Y['QQQ↔TQQQ']).sum()}/{len(Y)}；"
  f"勝 QQQ 的年份：{(Y['全期貨L3週'] > Y['QQQ']).sum()}/{len(Y)}")

# ---------------------------------------------------------------- I
last = S.iloc[-1]
c = S.close
m200 = c.rolling(200).mean().iloc[-1]
tr = pd.concat([S.close.diff().abs()], axis=1)
live = dict(date=str(S.index[-1].date()), qqq_close=float(last.close), ma200=round(float(m200), 2),
            above_ma200=bool(last.close > m200), atr20_pct=round(float(last.atrp) * 100, 3),
            regime_on_after_close=bool(last.on),
            note="z/bz 欄為『截至前一日』；下一個交易日的判斷請用 signals() 重算")
df = load()
pc = df.close.shift(1)
trr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
la = np.log(trr.rolling(20).mean() / df.close)
live["atr_z_today"] = round(float((la.iloc[-1] - la.rolling(252).mean().iloc[-1]) / la.rolling(252).std().iloc[-1]), 3)
live["boll_z100_today"] = round(float((df.close.iloc[-1] - df.close.rolling(100).mean().iloc[-1]) / df.close.rolling(100).std().iloc[-1]), 3)
live["exit_tomorrow"] = bool(live["regime_on_after_close"] and (not live["above_ma200"] or live["atr_z_today"] >= 0.5))
live["mnq_notional_est"] = round(2.0 * NDX_PER_QQQ * float(last.close))
live["contracts_formula"] = "全期貨：MNQ 口數 = round(目標槓桿 × 部位淨值 / (2 × NDX))；現股+期貨：round((L × 淨值 − QQQ 市值) / (2 × NDX))"
json.dump(live, open("part22_live.json", "w"), ensure_ascii=False, indent=2)
P("\nI. 目前訊號（part22_live.json）")
for k, v in live.items():
    P(f"  {k}: {v}")

# ---------------------------------------------------------------- chart
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(11, 8))
for lab, kw, col in [("QQQ weekly DCA", dict(eng="dca"), "#888"), ("QQQ<->TQQQ", dict(eng="etf"), "#d62728"),
                     ("All-futures L=3 weekly", dict(eng="allfut", L=3.0), "#1f77b4"), ("QQQ + futures overlay L=3 weekly", dict(eng="fut", L=3.0), "#2ca02c")]:
    o = run(S, **kw)
    cum = pd.Series(0.0, index=S.index)
    for d, a in o["flows"]:
        cum[d] += a
    ax[0].plot(o["V"] / cum.cumsum(), label=lab, color=col, lw=1)
ax[0].set_yscale("log"); ax[0].set_title("Value / cumulative contributions (from 1999)"); ax[0].legend(); ax[0].grid(alpha=.3)
x = np.arange(len(WIN))
tot_e = [m["tot"] * 100 for m in REF["etf"]]
tot_d = [m["tot"] * 100 for m in REF["dca"]]
tot_f = [m["tot"] * 100 for m in windows(eng="allfut", L=3.0)]
ax[1].bar(x - .27, tot_d, .27, label="QQQ weekly DCA", color="#888")
ax[1].bar(x, tot_e, .27, label="QQQ<->TQQQ", color="#d62728")
ax[1].bar(x + .27, tot_f, .27, label="All-futures L=3", color="#1f77b4")
ax[1].set_xticks(x); ax[1].set_xticklabels(WIN, rotation=45); ax[1].set_title("10-year DCA total P&L (%) by start year"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("part22_equity.png", dpi=110)
out.close()
