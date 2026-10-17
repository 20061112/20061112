"""第二十九部分：「去掉最好的 10 天就很難看」—— 大漲日出現後加倉有沒有用？

日報酬 z 兩種定義（都用到當天收盤，隔天才動作）：
  z252 = (r − 252 日平均) / 252 日標準差      ← 相對長期
  z20  = r / 前 20 日標準差                   ← 相對近期波動（平靜期的 +2% 也算大）
A. 事實：QQQ 去掉最好 / 最差的 N 天；最好的日子落在哪裡（MA200 上下、策略開不開槓桿、ATR z、離最差日幾天）
B. 大漲日之後 QQQ 的前瞻報酬與波動（1 / 5 / 20 / 60 天）
C. 大漲日後加倉：策略在 1 倍時拉到 2 或 3 倍，抱 H 天；可限 MA200 之上；安慰劑 = 同次數隨機日子
D. 對照：大跌日後加倉（均值回歸）
QQQ 全期貨、1999 起、每天重設；前後半 1999~2012 / 2013~2026。
"""
import itertools
import numpy as np
import pandas as pd
from part22_core import load, signals, run, metrics, year_starts

out = open("results_part29.txt", "w")


def P(s=""):
    print(s); out.write(s + "\n"); out.flush()


def cagr(x):
    return (1 + x).prod() ** (252 / len(x)) - 1


DF = load()
S = signals(DF)
YS = year_starts(S)
WIN = list(range(2000, 2017))
SPLIT = pd.Timestamp("2013-01-01")
KW = dict(eng="allfut", L=3.0, rebal="daily")
ON = S.on.values
r = S.ret
Z252 = ((r - r.rolling(252).mean().shift(1)) / r.rolling(252).std().shift(1))
Z20 = r / r.rolling(20).std().shift(1)
ma200 = S.close.rolling(200).mean()
ABOVE = (S.close > ma200)
YEARS = (2000, 2002, 2008, 2009, 2020, 2022)


def ev(full=True, **extra):
    kw = {**KW, **extra}
    o = run(S, **kw)
    m = metrics(o)
    tw = pd.Series(o["twr"], index=S.index)
    m["h1"], m["h2"] = cagr(tw[tw.index < SPLIT]), cagr(tw[tw.index >= SPLIT])
    m["yr"] = (1 + tw).groupby(S.index.year).prod() - 1
    if full:
        m["w"] = np.array([metrics(run(S, i0=YS[y], i1=YS[y + 10], **kw))["tot"] for y in WIN])
    return m


B = ev()


def overlay(trig, H, extra):
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


def placebo(trig, H, extra, act, n_draw=60):
    rng = np.random.default_rng(29)
    cand = np.flatnonzero(~ON & (np.arange(len(S)) > 260))
    k = max(int((trig & ~ON).sum()), 1)
    vals = []
    for _ in range(n_draw):
        t = np.zeros(len(S), bool); t[rng.choice(cand, min(k, len(cand)), replace=False)] = True
        reg, L, _ = overlay(t, H, extra)
        vals.append(ev(full=False, regime_arr=reg, L_arr=L)["irr"])
    return (np.array(vals) < act).mean() * 100


def show(lab, m, pl=None):
    d1, d2 = m["h1"] - B["h1"], m["h2"] - B["h2"]
    tag = "同向+" if d1 > 0.002 and d2 > 0.002 else ("同向−" if d1 < -0.002 and d2 < -0.002 else "  —  ")
    s = (f"  {lab:38s}{m['irr'] * 100:6.1f}%{m['worst'] * 100:6.0f}%{m['mdd'] * 100:6.0f}%{d1 * 100:+7.1f}%{d2 * 100:+7.1f}% {tag}"
         f"{(m['w'] > B['w']).sum():3d}/17 │ " + " ".join(f"{m['yr'][y] * 100:+4.0f}" for y in YEARS))
    if pl is not None:
        s += f" │ 安慰劑 {pl:3.0f}%"
    P(s)


# ---------------------------------------------------------------- A
P("第二十九部分：大漲日之後加倉（QQQ 全期貨、1999 起、每天重設）\n")
P("A. 去掉最好 / 最差的日子（QQQ 價格報酬，1999/3 ~ 2026/10，年化）")
yrs = len(r) / 252
base_ann = (1 + r).prod() ** (1 / yrs) - 1
srt = r.sort_values()
for n in (5, 10, 20, 40):
    no_best = (1 + r.drop(srt.index[-n:])).prod() ** (1 / yrs) - 1
    no_worst = (1 + r.drop(srt.index[:n])).prod() ** (1 / yrs) - 1
    both = (1 + r.drop(srt.index[-n:]).drop(srt.index[:n])).prod() ** (1 / yrs) - 1
    P(f"  N={n:2d}：全部 {base_ann * 100:5.1f}%｜去掉最好 {no_best * 100:5.1f}%｜去掉最差 {no_worst * 100:5.1f}%｜兩邊都去掉 {both * 100:5.1f}%")
best = srt.index[-20:]
worst = srt.index[:20]
dist_to_worst = [min(abs(S.index.get_loc(b) - S.index.get_loc(w)) for w in srt.index[:50]) for b in best]
P(f"\n  最好的 20 天：在 MA200 之下 {(~ABOVE[best]).mean() * 100:.0f}%｜策略當時在 1 倍 {(~S.on.shift(1).fillna(False)[best].astype(bool)).mean() * 100:.0f}%｜"
  f"當時 ATR z 中位 {S.z[best].median():.2f}｜離最差 50 天之一 ≤ 10 個交易日的 {np.mean(np.array(dist_to_worst) <= 10) * 100:.0f}%")
P(f"  最差的 20 天：在 MA200 之下 {(~ABOVE[worst]).mean() * 100:.0f}%｜策略當時在 1 倍 {(~S.on.shift(1).fillna(False)[worst].astype(bool)).mean() * 100:.0f}%")
P("  最好的 10 天：" + "、".join(f"{d.date()} {r[d] * 100:+.1f}%" for d in srt.index[-10:][::-1]))

# ---------------------------------------------------------------- B
P("\nB. 大漲日之後 QQQ 的前瞻（平均報酬 / 年化波動），對照全部日子")
fwd = {h: S.close.shift(-h) / S.close - 1 for h in (1, 5, 20, 60)}
vol20 = r.rolling(20).std().shift(-20) * np.sqrt(252)


def fw(mask, lab):
    cells = " ".join(f"{h:2d}天 {fwd[h][mask].mean() * 100:+5.2f}%" for h in fwd)
    P(f"  {lab:28s}n={int(mask.sum()):4d}｜{cells}｜之後 20 天波動 {vol20[mask].mean() * 100:4.0f}%")


fw(pd.Series(True, index=S.index), "全部日子")
for zn, Z in (("z252", Z252), ("z20", Z20)):
    for t in (2.0, 3.0):
        fw(Z >= t, f"{zn} ≥ {t}")
        fw((Z >= t) & ABOVE, f"{zn} ≥ {t} 且 MA200 之上")
    fw(Z <= -2.0, f"{zn} ≤ −2（大跌日）")

# ---------------------------------------------------------------- C / D
HDR = f"  {'變體':38s}{'IRR':>7s}{'最慘':>6s}{'回撤':>6s}{'前半差':>8s}{'後半差':>8s} 兩半  勝段 │ " + " ".join(f"{y:>4d}" for y in YEARS)
P("\nC. 大漲日後加倉：策略在 1 倍時拉到 1+extra 倍，抱 H 天（隔天收盤進）")
P(HDR)
show("基準", B)
for (zn, Z), t, H, extra, cond in itertools.product((("z252", Z252), ("z20", Z20)), (2.0, 3.0), (5, 20, 60), (1.0, 2.0), ("任何", "MA200上")):
    trig = (Z >= t)
    if cond == "MA200上":
        trig = trig & ABOVE
    trig = trig.shift(1).fillna(False).astype(bool).values
    reg, L, n = overlay(trig, H, extra)
    m = ev(regime_arr=reg, L_arr=L)
    pl = placebo(trig, H, extra, m["irr"]) if extra == 1.0 else None
    show(f"{zn}≥{t:.0f} {cond} 抱{H}天 {1 + extra:.0f}倍（{n}次）", m, pl)

P("\nD. 對照：大跌日後加倉")
P(HDR)
for (zn, Z), t, H in itertools.product((("z252", Z252), ("z20", Z20)), (2.0, 3.0), (5, 20, 60)):
    trig = (Z <= -t).shift(1).fillna(False).astype(bool).values
    reg, L, n = overlay(trig, H, 1.0)
    m = ev(regime_arr=reg, L_arr=L)
    show(f"{zn}≤−{t:.0f} 抱{H}天 2倍（{n}次）", m, placebo(trig, H, 1.0, m["irr"]))
out.close()
