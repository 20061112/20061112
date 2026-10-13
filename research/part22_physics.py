"""第二十二部分：幾何光學 / 電磁學類比指標 — 逐一檢查「算出來的值」與盤面有沒有關係（不做成策略）。
資料：data/XAUUSD_M1_2023_2026.csv（2023/6/1 ~ 2026/10/8，118 萬根 M1）。
統計方式：
  - 連續型指標：每天算一次 Spearman IC（指標 vs 前瞻報酬），用「日 IC 平均 / 標準誤」算 t，並逐年列出；
  - 事件型指標：分組比較結果，逐年列出，並和「只用價格的簡單對照組」比較，看物理量有沒有額外資訊。
輸出：results_part22.txt
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from load import load
import physics_features as pf

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
out = open("results_part22.txt", "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")
    out.flush()


def daily_ic(x, y, every=1):
    df = pd.DataFrame({"x": x, "y": y}).iloc[::every].dropna()
    g = df.groupby(df.index.date)
    ic = g.apply(lambda t: t.x.rank().corr(t.y.rank()) if len(t) > 20 else np.nan).dropna()
    ic.index = pd.to_datetime(ic.index)
    t = ic.mean() / ic.std() * np.sqrt(len(ic))
    yr = ic.groupby(ic.index.year).mean()
    return ic.mean(), t, (ic > 0).mean(), yr


def ic_line(name, x, y, every=1):
    m, t, pos, yr = daily_ic(x, y, every)
    ys = " ".join(f"{k}:{v:+.3f}" for k, v in yr.items())
    P(f"  {name:<34s} IC {m:+.4f}  t {t:+6.2f}  正IC日 {pos:.0%}  | {ys}")
    return m, t


def resid_on(x, c):
    """x 對控制變數 c 做排名迴歸後的殘差（看 x 在 c 之外的增量資訊）。"""
    df = pd.DataFrame({"x": x, "c": c}).dropna()
    rx, rc = df.x.rank(), df.c.rank()
    b = np.cov(rx, rc)[0, 1] / rc.var()
    return (rx - b * rc).reindex(x.index)


def quint(x, y, q=5, every=1, label=""):
    df = pd.DataFrame({"x": x, "y": y}).iloc[::every].dropna()
    df["b"] = pd.qcut(df.x.rank(method="first"), q, labels=False) + 1
    g = df.groupby("b").y
    tab = pd.DataFrame({"x中位": df.groupby("b").x.median(), "平均": g.mean(), "t": g.mean() / g.std() * np.sqrt(g.count()),
                        "n": g.count()})
    P(f"  {label}\n" + tab.round(3).to_string())
    return tab


d = pf.base(load("data/XAUUSD_M1_2023_2026.csv"))
F = pf.forward(d)
P(f"資料 {d.index[0]} ~ {d.index[-1]}，{len(d):,} 根 M1；所有報酬以 ATR60(M1) 為單位。")
P("參考：中位點差 / ATR60 ≈", f"{(d.spread * 0.01 / d.atr).median():.2f}", "（每筆交易來回成本大約是這麼多 ATR）")

# ================================================================== 1. 折射率 n 與費馬最優路徑
P("\n" + "=" * 100 + "\n一、折射率 n = σ/√V（每單位成交量的波動）與費馬原理（最優執行路徑）\n" + "=" * 100)
n = pf.refractive(d, 15)
ln = np.log(n.replace(0, np.nan))
P("1-1 n 是否「可預測」（費馬路徑的前提：要能事先知道哪一段介質比較稠密）")
for lag in (15, 60, 240, 1440):
    P(f"  log n(t) vs log n(t+{lag:4d}) 相關 = {ln.corr(ln.shift(-lag)):.3f}")
mod = d.index.hour * 60 + d.index.minute
prof = ln.groupby(mod).transform("mean")
resid = ln - prof
P(f"  一天內時間點（分鐘）就能解釋 log n 變異的 {1 - resid.var() / ln.var():.1%}；"
  f"去掉時段效應後 15 根後相關仍有 {resid.corr(resid.shift(-15)):.3f}")
P("\n1-2 n 與後續行情")
ic_line("n → 未來15根報酬(方向)", n, F[15], every=15)
ic_line("n → 未來15根路徑長(波動)", n, F["path15"], every=15)
ic_line("n(去時段) → 未來15根路徑長", resid, F["path15"], every=15)
quint(n, F["path15"], every=15, label="n 五分位 vs 未來15根路徑長（ATR）")

P("\n1-3 執行模擬：每天每個整點下一張母單，60 分鐘內完成（成本用平方根衝擊模型 Σ σ_i·q_i·√(q_i/V_i)，"
  "σ、V 用事後真實值）")
P("    比較排程：TWAP（平均分）/ 量能曲線（過去 20 天同分鐘的平均量）/ 費馬（q ∝ V̂/σ̂² = 1/n̂²，"
  "n̂ 用過去 20 天同分鐘剖面 × 最近 30 分鐘水準修正）/ 先知（用真實 σ、V，理論下限）")
tv = d.tickvol.astype(float)
sg = (d.high - d.low).clip(lower=0.01)
day = pd.Index(d.index.date)
Vt = pd.DataFrame({"d": day, "m": mod, "v": tv.values, "s": sg.values})
pv = Vt.pivot_table(index="d", columns="m", values="v")
ps = Vt.pivot_table(index="d", columns="m", values="s")
# 剖面：過去 20 天同分鐘平均，再沿分鐘方向做 5 分鐘平滑（單一分鐘的 20 天平均仍很吵，平方後會讓權重過度集中）
pv_hat = pv.rolling(20, min_periods=10).mean().shift(1).T.rolling(5, center=True, min_periods=1).mean().T
ps_hat = ps.rolling(20, min_periods=10).mean().shift(1).T.rolling(5, center=True, min_periods=1).mean().T
res = []
for h in range(3, 22):
    cols = list(range(h * 60, h * 60 + 60))
    pre = list(range(h * 60 - 30, h * 60))
    if not set(cols + pre) <= set(pv.columns):
        continue
    V, S = pv[cols].values, ps[cols].values
    Vh, Sh = pv_hat[cols].values, ps_hat[cols].values
    lv = np.nanmean(pv[pre].values, 1) / np.nanmean(pv_hat[pre].values, 1)  # 今天的水準修正
    ls = np.nanmean(ps[pre].values, 1) / np.nanmean(ps_hat[pre].values, 1)
    decay = np.exp(-np.arange(60) / 30.0)[None, :]  # 今天偏離剖面的程度隨時間衰減（1-1：15 根後相關 0.45）
    ok = ~(np.isnan(V).any(1) | np.isnan(S).any(1) | np.isnan(Vh).any(1) | np.isnan(lv) | np.isnan(ls))
    V, S, Vh, Sh, lv, ls = V[ok], S[ok], Vh[ok], Sh[ok], lv[ok], ls[ok]

    def cost(q):
        q = q / q.sum(1, keepdims=True)
        return (S * q * np.sqrt(q / V)).sum(1)

    c = {"TWAP": cost(np.ones_like(V)), "量能曲線": cost(Vh), "費馬(剖面)": cost(Vh / Sh ** 2),
         "費馬(剖面+水準)": cost(Vh * lv[:, None] ** decay / (Sh * ls[:, None] ** decay) ** 2),
         "費馬(只用水準)": cost(lv[:, None] ** decay / ls[:, None] ** (2 * decay)), "先知": cost(V / S ** 2)}
    dates = pv.index[ok]
    for k, v in c.items():
        res.append(pd.DataFrame({"h": h, "d": dates, "k": k, "c": v}))
R = pd.concat(res)
W = R.pivot_table(index=["d", "h"], columns="k", values="c")
rel = W.div(W["TWAP"], axis=0) - 1
yr = pd.to_datetime(rel.index.get_level_values(0)).year
P("  相對 TWAP 的成本變化（平均，負 = 省）：")
P((rel.groupby(yr).mean() * 100).round(2).to_string())
P(f"  費馬(剖面+水準) 優於 TWAP 的母單比例：{(W['費馬(剖面+水準)'] < W['TWAP']).mean():.1%}")
P(f"  費馬(剖面+水準) 優於量能曲線的母單比例：{(W['費馬(剖面+水準)'] < W['量能曲線']).mean():.1%}")
P("  注意：成本函數是模型（平方根衝擊），不是真實成交；這裡只驗證「n 的預測夠不夠準，能不能讓排程接近先知」。")

# ================================================================== 2. 司乃耳定律 / 全反射
P("\n" + "=" * 100 + "\n二、司乃耳定律：碰到壓力/支撐時，全反射（反彈）還是折射（突破）？\n" + "=" * 100)
SN = pf.snell_events(d)
SN = SN.dropna(subset=["outcome"])
SN["yr"] = SN.time.dt.year
SN["brk"] = (SN.outcome == 1).astype(float)
SN["rev"] = (SN.outcome == -1).astype(float)
SN["TIR"] = SN.R > 1
P(f"事件：第一次碰到 120 根（且至少 10 根前形成）的高/低點，共 {len(SN):,} 筆；"
  f"結果 = 30 根內先到介面外 1.5 ATR（突破）或內側 1.5 ATR（反彈）")
P(f"  整體：突破 {SN.brk.mean():.1%}，反彈 {SN.rev.mean():.1%}，30 根內都沒到 {1 - SN.brk.mean() - SN.rev.mean():.1%}")
g = SN.groupby("TIR")
P("\n2-1 物理預測（R>1 全反射 / R<1 折射）")
P(pd.DataFrame({"n": g.size(), "突破率": g.brk.mean(), "反彈率": g.rev.mean(), "30根後(ATR,突破方向)": g.fwd30.mean()}).round(3).to_string())
P("  逐年 突破率（R<1 折射 / R>1 全反射）：")
P(SN.pivot_table(index="yr", columns="TIR", values="brk", aggfunc="mean").round(3).to_string())
P("\n2-2 拆開看：R 五分位，以及 R 的兩個成分（入射速度 v、外/內側稠密度比）各自的五分位")
for col, lab in (("R", "司乃耳比 R = n1·sinθ1/n2"), ("v", "入射速度 v（ATR/根）"), ("dens_ratio", "內側/外側 成交量密度比（= n2/n1；大 = 介面外是成交真空）")):
    SN["q"] = pd.qcut(SN[col].rank(method="first"), 5, labels=False) + 1
    gg = SN.groupby("q")
    P(f"  {lab}")
    P(pd.DataFrame({"中位": gg[col].median(), "突破率": gg.brk.mean(), "反彈率": gg.rev.mean(),
                    "30根後": gg.fwd30.mean(), "n": gg.size()}).round(3).to_string())
SN["vac"] = pd.qcut(SN.dens_ratio.rank(method="first"), 5, labels=False) + 1
P("  逐年：介面外真空（密度比 Q5）vs 其他 的突破率 / 30 根後")
P(pd.concat([SN.pivot_table(index="yr", columns=SN.vac == 5, values="brk", aggfunc="mean").add_prefix("突破率_Q5="),
             SN.pivot_table(index="yr", columns=SN.vac == 5, values="fwd30", aggfunc="mean").add_prefix("30根後_Q5=")], axis=1).round(3).to_string())
x5, x0 = SN.fwd30[SN.vac == 5], SN.fwd30[SN.vac < 5]
P(f"  Q5 − 其他 30根後差 {x5.mean() - x0.mean():+.3f} ATR，t {(x5.mean() - x0.mean()) / np.sqrt(x5.var() / len(x5) + x0.var() / len(x0)):+.2f}")
P("\n2-3 R 在控制 v 之後還有沒有資訊？（v 分三組，每組內比較 R 高低兩半的突破率）")
SN["vq"] = pd.qcut(SN.v.rank(method="first"), 3, labels=["慢", "中", "快"])
SN["Rhi"] = SN.groupby("vq", observed=True).R.transform(lambda x: x > x.median())
P(SN.pivot_table(index="vq", columns="Rhi", values="brk", aggfunc="mean", observed=True).round(3).to_string())
P(SN.pivot_table(index="vq", columns="Rhi", values="fwd30", aggfunc="mean", observed=True).round(3).to_string())

# ================================================================== 3. 焦散
P("\n" + "=" * 100 + "\n三、焦散：多個尺度的停損（轉折點）擠在同一價位，被一起觸發時會不會爆發？\n" + "=" * 100)
CE = pf.caustic_events(d)
CE["yr"] = CE.time.dt.year
CE["Fb"] = pd.cut(CE.F, [0, 1, 2, 3, 5, 1000], labels=["1", "2", "3", "4-5", "6+"])
P(f"事件：每根 K 棒第一次突破過去 1 天內任一尺度（5/15/60/240 根）轉折點，共 {len(CE):,} 筆；"
  f"F = 突破價位往外 0.5 ATR 內、同側還沒被破的轉折點數（同時被觸發的停損束）")
g = CE.groupby("Fb", observed=True)
P(pd.DataFrame({"n": g.size(), "突破方向最大延伸MFE": g.mfe.mean(), "反向MAE": g.mae.mean(), "15根後(突破方向)": g.fwd.mean(),
                "15根區間": g.rng.mean(), "MFE>3ATR機率": g.mfe.apply(lambda x: (x > 3).mean()),
                "t(15根後)": g.fwd.apply(lambda x: x.mean() / x.std() * np.sqrt(len(x)))}).round(3).to_string())
P("  逐年 15根區間（F=1 vs F≥4，數字越大 = 爆發越大）：")
CE["Fhi"] = np.where(CE.F >= 4, "F>=4", np.where(CE.F == 1, "F=1", "mid"))
P(CE.pivot_table(index="yr", columns="Fhi", values="rng", aggfunc="mean").round(3).to_string())
P("  逐年 15根後(突破方向)：")
P(CE.pivot_table(index="yr", columns="Fhi", values="fwd", aggfunc="mean").round(3).to_string())
P("  對照：F 大是否只是因為「附近波動本來就小、轉折點才會擠在一起」？以突破前 15 根區間（ATR）分組再比較 15 根區間：")
pre = ((d.high.rolling(15).max() - d.low.rolling(15).min()) / d.atr).values
CE["pre"] = pre[CE.i.values - 1]
CE["preq"] = pd.qcut(CE.pre.rank(method="first"), 3, labels=["窄", "中", "寬"])
P(CE.pivot_table(index="preq", columns="Fhi", values="rng", aggfunc="mean", observed=True).round(3).to_string())

# ================================================================== 4. 阻抗 / 反射係數
P("\n" + "=" * 100 + "\n四、特徵阻抗 Z = 路徑/成交量 與反射係數 Γ = (Z2−Z1)/(Z2+Z1)\n" + "=" * 100)
IM = pf.impedance(d)
sgn = np.sign(IM.move)
for h in (5, 15, 60):
    IM[f"cont{h}"] = sgn * F[h]  # >0 = 延續（穿透），<0 = 反轉（反射）
P("4-1 Γ 與「最近 5 根那一波」之後是延續還是反轉（>0 延續、<0 反射）")
ic_line("Γ → 延續(5根)", IM.Gamma, IM.cont5, every=5)
ic_line("Γ → 延續(15根)", IM.Gamma, IM.cont15, every=15)
ic_line("Γ → 延續(60根)", IM.Gamma, IM.cont60, every=60)
ic_line("對照 Γ_price(只用價格) → 延續15", IM.Gamma_price, IM.cont15, every=15)
quint(IM.Gamma, IM.cont15, every=15, label="Γ 五分位 vs 15 根延續（ATR）")
P("\n4-2 |Γ|≈0（阻抗匹配、T≈1）時趨勢是否比較能延續？只看最近 5 根移動 ≥ 1.5 ATR 的波")
big = IM[IM.move.abs() >= 1.5].iloc[::5].dropna(subset=["Gamma", "cont15"])
big = big.assign(m=pd.cut(big.Gamma.abs(), [0, 0.1, 0.25, 0.5, 1.0], include_lowest=True))
g = big.groupby("m", observed=True).cont15
P(pd.DataFrame({"平均延續15": g.mean(), "t": g.mean() / g.std() * np.sqrt(g.count()), "n": g.count()}).round(3).to_string())
P("\n4-3 Γ 與未來波動")
ic_line("Γ → 未來15根路徑長", IM.Gamma, F["path15"], every=15)
ic_line("對照 Γ_price → 未來15根路徑長", IM.Gamma_price, F["path15"], every=15)
ic_line("Γ 扣掉 Γ_price 後 → 未來15根路徑長", resid_on(IM.Gamma, IM.Gamma_price), F["path15"], every=15)
vratio = d.rvol.rolling(5).mean() / d.rvol.rolling(30).mean().shift(5)
ic_line("對照 量比 V5/V30 → 未來15根路徑長", vratio, F["path15"], every=15)
ic_line("Γ 扣掉 量比 後 → 未來15根路徑長", resid_on(IM.Gamma, vratio), F["path15"], every=15)
ic_line("對照 只用最近5根相對量 → 未來15根路徑長", d.rvol.rolling(5).mean(), F["path15"], every=15)
P("\n4-4 時段交接（歐盤 10:00、美盤 15:30/16:30 伺服器時間前後 30 分鐘）Γ 的效果")
hm = d.index.hour * 60 + d.index.minute
sess = ((hm >= 570) & (hm < 630)) | ((hm >= 900) & (hm < 1020))
ic_line("Γ → 延續15（交接時段）", IM.Gamma[sess], IM.cont15[sess], every=5)
ic_line("Γ → 延續15（其他時段）", IM.Gamma[~sess], IM.cont15[~sess], every=15)

# ================================================================== 5. 坡印廷向量
P("\n" + "=" * 100 + "\n五、坡印廷向量 S = E × H（價格偏離 × 委託流）\n" + "=" * 100)
PY = pf.poynting(d)
P("5-1 各分量對未來報酬的 IC（正 = 延續、負 = 回歸）")
for h in (15, 60):
    ic_line(f"E(偏離) → 未來{h}", PY.E, F[h], every=h)
    ic_line(f"H(委託流) → 未來{h}", PY.H, F[h], every=h)
    ic_line(f"S=E·H → 未來{h}（方向乘上 sign E）", PY.S, np.sign(PY.E) * F[h], every=h)
    ic_line(f"對照 |E| → 未來{h}（同上方向）", PY.E.abs(), np.sign(PY.E) * F[h], every=h)
ic_line("H 扣掉 E 後 → 未來15", resid_on(PY.H, PY.E), F[15], every=15)
ic_line("H 扣掉 E 後 → 未來60", resid_on(PY.H, PY.E), F[60], every=60)
P("\n5-2 E、H 交叉表：未來 60 根報酬（ATR）。H 與 E 同號 = 能流往偏離方向；異號 = 價格偏離但委託流反向")
X = pd.concat([PY, F[60]], axis=1).iloc[::60].dropna()
X["Eb"] = pd.qcut(X.E, 5, labels=False) + 1
X["Hb"] = pd.qcut(X.H, 5, labels=False) + 1
P(X.pivot_table(index="Eb", columns="Hb", values=60, aggfunc="mean").round(3).to_string())
P("\n5-3 背離：價格創 240 根新高/新低（30 根內第一次），但 S 比上一次新高/低時小一半以上（能流衰竭）")
hh = d.high >= d.high.rolling(240).max()
ll = d.low <= d.low.rolling(240).min()
ev = []
for s, mask in ((1, hh), (-1, ll)):
    first = mask & ~mask.shift(1).rolling(30).max().fillna(0).astype(bool)
    idx = np.where(first.values)[0]
    Sv = PY.S.values
    Ev = PY.E.values
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 240:
            continue
        ev.append((b, s, Sv[b], Sv[a], Ev[b], Ev[a]))
EV = pd.DataFrame(ev, columns=["i", "side", "S", "Sprev", "E", "Eprev"])
EV["fwd60"] = EV.side * F[60].values[EV.i]
EV["fwd15"] = EV.side * F[15].values[EV.i]
EV["yr"] = d.index[EV.i].year
EV["divS"] = EV.S < 0.5 * EV.Sprev
EV["divE"] = EV.side * EV.E < 0.5 * EV.side * EV.Eprev  # 對照：只用價格偏離的背離
EV = EV.dropna(subset=["fwd60"])
for c, lab in (("divS", "S 背離（能流）"), ("divE", "對照：E 背離（只用價格）")):
    g = EV.groupby(c)
    P(f"  {lab}")
    P(pd.DataFrame({"n": g.size(), "15根後(順新高方向)": g.fwd15.mean(), "60根後": g.fwd60.mean(),
                    "t60": g.fwd60.apply(lambda x: x.mean() / x.std() * np.sqrt(len(x)))}).round(3).to_string())
P("  逐年 60 根後（S 背離 True/False）：")
P(EV.pivot_table(index="yr", columns="divS", values="fwd60", aggfunc="mean").round(3).to_string())
P("  S 背離 但 E 沒背離（能流獨有的資訊）：")
sub = EV[EV.divS & ~EV.divE]
P(f"    n={len(sub)}  60根後 {sub.fwd60.mean():+.3f}  t {sub.fwd60.mean() / sub.fwd60.std() * np.sqrt(len(sub)):+.2f}")

# ================================================================== 6. 規範場
P("\n" + "=" * 100 + "\n六、規範場曲率（三角套利環路積分）\n" + "=" * 100)
P("  需要至少三個能組成閉環的報價（例如 XAUUSD、XAUEUR、EURUSD，或 EURUSD、USDJPY、EURJPY）的同步資料；")
P("  repo 只有 XAUUSD 單一商品，而這個環境的網路政策擋掉了外部行情來源（stooq / Yahoo / Dukascopy 都回 403），因此本項無法測試。")
out.close()
