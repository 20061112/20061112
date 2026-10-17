"""第二十八部分：VIX z-score 很高（恐慌）時加倉 —— 逆勢加碼測試。

VIX z = (ln VIX − 252 日平均) / 252 日標準差，用前一天收盤。z 高 = 恐慌（VIX 高），不是 VIX 低。
QQQ 全期貨、1999 起完整樣本（含 2000~2002、2008），每天重設以便槓桿變化即時生效；基準也每天重設。

A. 整個部位加槓桿：VIX z ≥ t 時（此時策略多半在 1 倍），把部位拉到 1 + extra 倍，抱 H 個交易日
   可選「等高峰過」：VIX 已從 5 日高點回落才加
B. 定投金額依 z 放大：當週投入 × (1 + k × max(z, 0))，上限 4 倍
C. 恐慌補倉（20% 現金）改用 VIX z ≥ t
每組都做安慰劑：同樣次數的隨機加倉日 100 次。前後半 = 1999~2012 / 2013~2026。
"""
import itertools
import numpy as np
import pandas as pd
from part22_core import load, signals, run, metrics, year_starts

out = open("results_part28.txt", "w")


def P(s=""):
    print(s); out.write(s + "\n"); out.flush()


def cagr(x):
    return (1 + x).prod() ** (252 / len(x)) - 1


DF = load()
S = signals(DF)
YS = year_starts(S)
WIN = list(range(2000, 2017))
SPLIT = pd.Timestamp("2013-01-01")
vix = pd.read_csv("data/VIX_History.csv")
vix.index = pd.to_datetime(vix.DATE, format="%m/%d/%Y")
v = vix.CLOSE.astype(float)
lv = np.log(v)
vz_full = (lv - lv.rolling(252).mean()) / lv.rolling(252).std()
VZ = vz_full.reindex(S.index.union(vz_full.index)).ffill(limit=3).reindex(S.index).shift(1).values
VIXc = v.reindex(S.index.union(v.index)).ffill(limit=3).reindex(S.index)
PAST_PEAK = (VIXc < VIXc.rolling(5).max()).shift(1).fillna(False).values
KW = dict(eng="allfut", L=3.0, rebal="daily")
ON = S.on.values
YEARS = (2000, 2001, 2002, 2008, 2020, 2022)


def ev(full=True, **extra):
    kw = {**KW, **extra}
    o = run(S, **kw)
    m = metrics(o)
    tw = pd.Series(o["twr"], index=S.index)
    m["h1"], m["h2"] = cagr(tw[tw.index < SPLIT]), cagr(tw[tw.index >= SPLIT])
    m["yr"] = (1 + tw).groupby(S.index.year).prod() - 1
    m["inv"] = o["inv"]
    if full:
        m["w"] = np.array([metrics(run(S, i0=YS[y], i1=YS[y + 10], **kw))["tot"] for y in WIN])
    return m


B = ev()


def show(lab, m, pl=None):
    d1, d2 = m["h1"] - B["h1"], m["h2"] - B["h2"]
    tag = "同向+" if d1 > 0.002 and d2 > 0.002 else ("同向−" if d1 < -0.002 and d2 < -0.002 else "  —  ")
    yrs = " ".join(f"{m['yr'][y] * 100:+4.0f}" for y in YEARS)
    s = (f"  {lab:34s}{m['irr'] * 100:6.1f}%{m['worst'] * 100:6.0f}%{m['mdd'] * 100:6.0f}%{d1 * 100:+7.1f}%{d2 * 100:+7.1f}% {tag}"
         f"{(m['w'] > B['w']).sum():3d}/17 │ {yrs}")
    if pl is not None:
        s += f" │ 安慰劑 {pl:3.0f}%"
    P(s)


def overlay(trig, H, extra):
    """trig[i] 為真時開始（只用到 i−1），之後 H 天部位至少 1 + extra 倍（原本已 3 倍就不變）。"""
    reg, L = ON.copy(), np.where(ON, 3.0, 1.0)
    left, n = 0, 0
    for i in range(len(S)):
        if left == 0 and trig[i] and not ON[i]:
            left, n = H, n + 1
        if left > 0:
            if not ON[i]:
                reg[i], L[i] = True, 1.0 + extra
            left -= 1
    return reg, L, n


def placebo(trig, H, extra, act, n_draw=100):
    rng = np.random.default_rng(28)
    cand = np.flatnonzero(~ON & (np.arange(len(S)) > 260))
    k = max(int(trig.sum()), 1)
    vals = []
    for _ in range(n_draw):
        t = np.zeros(len(S), bool); t[rng.choice(cand, min(k, len(cand)), replace=False)] = True
        reg, L, _ = overlay(t, H, extra)
        vals.append(ev(full=False, regime_arr=reg, L_arr=L)["irr"])
    return (np.array(vals) < act).mean() * 100


HDR = (f"  {'變體':34s}{'IRR':>7s}{'最慘':>6s}{'回撤':>6s}{'前半差':>8s}{'後半差':>8s} 兩半  勝段 │ "
       + " ".join(f"{y:>4d}" for y in YEARS))
P("第二十八部分：VIX z 很高（恐慌）時加倉（QQQ 全期貨、1999 起、每天重設）")
P("  最慘 = 最慘相對本金；回撤 = 時間加權最大回撤；右側為各年報酬（%）；安慰劑 = 勝過幾 % 的隨機加倉日\n")
P(HDR)
show("基準（每天重設）", B)
dist = pd.Series(VZ[~ON & ~np.isnan(VZ)])
P(f"  （關槓桿期間 VIX z 分布：50% {dist.median():.2f}、90% {dist.quantile(.9):.2f}、97% {dist.quantile(.97):.2f}）")

P("\nA. 恐慌時整個部位加槓桿（觸發日＝前一天 VIX z ≥ t）")
for t, H, extra, peak in itertools.product((1.5, 2.0, 2.5), (20, 60), (1.0, 2.0), (False, True)):
    trig = (VZ >= t) & (PAST_PEAK if peak else True)
    trig = np.nan_to_num(trig).astype(bool)
    reg, L, n = overlay(trig, H, extra)
    m = ev(regime_arr=reg, L_arr=L)
    pl = placebo(trig, H, extra, m["irr"], 60) if extra == 1.0 else None
    show(f"z≥{t} 抱{H}天 {1 + extra:.0f}倍{' 等回落' if peak else ''}（{n}次）", m, pl)

P("\nB. 定投金額依 VIX z 放大：× (1 + k × max(z, 0))，上限 4 倍（取代布林加碼）")
for k in (0.5, 1.0, 2.0):
    mult = np.clip(1 + k * np.clip(np.nan_to_num(VZ), 0, None), 1, 4)
    m = ev(boll=False, amt_mult=mult)
    show(f"k = {k}（總投入 {m['inv'] / B['inv']:.2f} 倍）", m)
mult = np.where(np.nan_to_num(VZ) >= 1.5, 3.0, 1.0)
m = ev(boll=False, amt_mult=mult)
show(f"z ≥ 1.5 投入 3 倍（總投入 {m['inv'] / B['inv']:.2f} 倍）", m)

P("\nC. 恐慌補倉（20% 現金）改用 VIX z")
for t in (1.0, 1.5, 2.0, 2.5):
    show(f"VIX z ≥ {t}（原 ATR z > 2）", ev(panic_arr=np.nan_to_num(VZ) >= t))

P("\nD. 1 倍時多加槓桿 vs 固定加槓桿：關槓桿期間一律 2 倍（不看 VIX）當對照")
L = np.where(ON, 3.0, 2.0)
show("關槓桿期間都 2 倍（不看 VIX）", ev(regime_arr=np.ones(len(S), bool), L_arr=L))
out.close()
