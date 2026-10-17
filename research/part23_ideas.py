"""第二十三部分：把 repo 其他研究的概念搬到 QQQ↔3 倍槓桿策略上（全期貨 L=3、每週重設為基準）。

1. 第二十一部分 ER（效率比）濾網 → 開槓桿前要求趨勢「走得直」，針對震盪年（2005、2011、2015）
2. 第十八部分 分批加碼 → 槓桿分級：依 ATR z 或開槓桿後的週數，2 倍 → 3 倍
3. 第二十部分 虧損期對策 → 開槓桿不到 N 天就被洗出場的話，冷卻 K 週再進場
4. 第十五~十八部分 盤中停損 → 期貨幾乎 24 小時交易：持有槓桿時 QQQ 盤中跌破前收 −X% 就把額外的 2 倍出場
5. 第十部分 凍結參數樣本外（XAUUSD 全部策略在這關失敗）→ 只用 1999~2012 選訊號參數，2013~2026 檢驗
每個想法都報告前後兩半（1999~2012 / 2013~2026）相對基準的差異，要求兩半同向才算有用。
"""
import itertools
import numpy as np
import pandas as pd
from part22_core import load, signals, run, metrics, year_starts

DF = load()
S = signals(DF)
YS = year_starts(S)
WIN = list(range(2000, 2017))
I2010 = S.index.searchsorted(pd.Timestamp("2010-02-11"))
SPLIT = pd.Timestamp("2013-01-01")
CHOP = (2005, 2011, 2015)
out = open("results_part23.txt", "w")


def P(s=""):
    print(s); out.write(s + "\n"); out.flush()


def regime(S, entry_ok=None, cooldown=None, z_on=0.5):
    up, z, ws = S.up.values, S.z.values, S.ws.values
    on, start, block, res = False, 0, -1, np.zeros(len(S), bool)
    for i in range(len(S)):
        ok = up[i] and z[i] == z[i]
        if on and (not ok or z[i] >= z_on):
            on = False
            if cooldown and i - start < cooldown[0]:
                block = i + 5 * cooldown[1]
        if ws[i] and not on and ok and z[i] < z_on and i >= block and (entry_ok is None or entry_ok[i]):
            on, start = True, i
        res[i] = on
    return res


def evaluate(S_, kw):
    o = run(S_, **kw)
    m = metrics(o)
    m10 = metrics(run(S_, i0=I2010, **kw))
    w = [metrics(run(S_, i0=YS[y], i1=YS[y + 10], **kw))["tot"] for y in WIN]
    tw = pd.Series(o["twr"], index=S_.index)
    h1, h2 = tw[tw.index < SPLIT], tw[tw.index >= SPLIT]
    cagr = lambda x: (1 + x).prod() ** (252 / len(x)) - 1
    yr = (1 + tw).groupby(tw.index.year).prod() - 1
    return dict(irr=m["irr"], irr10=m10["irr"], avg10=np.mean(w), min10=np.min(w), worst=m["worst"],
                h1=cagr(h1), h2=cagr(h2), chop=yr.loc[list(CHOP)].mean(), w=np.array(w))


BASE_KW = dict(eng="allfut", L=3.0)
B = evaluate(S, BASE_KW)


def show(lab, r):
    d1, d2 = r["h1"] - B["h1"], r["h2"] - B["h2"]
    same = "同向" if np.sign(d1) == np.sign(d2) and abs(d1) > 0.002 and abs(d2) > 0.002 else "  —"
    P(f"  {lab:30s}{r['irr'] * 100:6.1f}%{r['irr10'] * 100:7.1f}%{r['avg10'] * 100:7.0f}%{r['min10'] * 100:6.0f}%"
      f"{r['worst'] * 100:7.0f}%{r['chop'] * 100:7.0f}%{d1 * 100:+7.1f}%{d2 * 100:+7.1f}%  {same}  "
      f"{(r['w'] > B['w']).sum():2d}/17")


HDR = (f"  {'變體':30s}{'IRR99':>7s}{'IRR10':>7s}{'10年均':>7s}{'最差':>6s}{'最慘/本金':>7s}{'震盪年':>7s}"
       f"{'前半差':>7s}{'後半差':>7s}  兩半  勝基準段")
P("第二十三部分：repo 其他研究的概念套用到 QQQ↔3 倍槓桿策略（全期貨 L=3、每週重設）")
P("  震盪年 = 2005、2011、2015 的平均報酬；前/後半差 = 1999~2012、2013~2026 年化報酬減基準；同向 = 兩半都改善或都變差（>0.2%）\n")
P(HDR)
show("基準", B)

# 1. ER 濾網
P("\n1. ER 效率比濾網（第二十一部分）：開槓桿時要求 ER(n) ≥ 門檻")
c = S.close
for n, thr in itertools.product((40, 60, 120), (0.1, 0.2, 0.3)):
    er = ((c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()).shift(1).fillna(0).values
    show(f"ER({n}) ≥ {thr}", evaluate(S, dict(regime_arr=regime(S, entry_ok=er >= thr), **BASE_KW)))

# 2. 槓桿分級
P("\n2. 槓桿分級（第十八部分的分批加碼）")
z = S.z.fillna(9).values
for zg in (-0.5, 0.0, 0.25):
    show(f"z < {zg} 才 3 倍，否則 2 倍", evaluate(S, dict(L_arr=np.where(z < zg, 3.0, 2.0), rebal="daily", **BASE_KW)))
on = S.on.values
age = np.zeros(len(S), int)
for i in range(1, len(S)):
    age[i] = age[i - 1] + 1 if on[i] and on[i - 1] else 0
for wk in (2, 4, 8):
    show(f"開槓桿前 {wk} 週 2 倍，之後 3 倍", evaluate(S, dict(L_arr=np.where(age < 5 * wk, 2.0, 3.0), **BASE_KW)))

# 3. 冷卻期
P("\n3. 被洗出場後冷卻（第二十部分的虧損期對策）：開槓桿不到 N 天就出場 → K 週內不再進場")
for short, blk in itertools.product((10, 20), (2, 4, 8)):
    show(f"<{short} 天出場 → 冷卻 {blk} 週", evaluate(S, dict(regime_arr=regime(S, cooldown=(short, blk)), **BASE_KW)))

# 4. 盤中停損
P("\n4. 盤中停損（期貨 24 小時）：持有槓桿時 QQQ 跌破前收 −X%，額外 2 倍在 −X%（跳空則開盤價）出場，多扣 0.1% 滑價")
pc = DF.close.shift(1).values
o_r, l_r = DF.open.values / pc - 1, DF.low.values / pc - 1
for X in (0.03, 0.04, 0.05, 0.07):
    reg = S.on.values.copy()
    fr = S.ret.values.copy()
    up, zz, ws = S.up.values, S.z.values, S.ws.values
    cur, n_stop = False, 0
    for i in range(len(S)):
        ok = up[i] and zz[i] == zz[i]
        if cur and l_r[i] <= -X:
            fill = min(o_r[i], -X) - 0.001
            fr[i] = (S.ret.values[i] + 2 * fill) / 3
            cur = False; n_stop += 1
        elif cur and (not ok or zz[i] >= 0.5):
            cur = False
        elif ws[i] and not cur and ok and zz[i] < 0.5:
            cur = True
        reg[i] = cur
    r = evaluate(S, dict(regime_arr=reg, fut_ret=fr, **BASE_KW))
    show(f"−{X * 100:.0f}% 停損（觸發 {n_stop} 次）", r)

# 5. 凍結參數樣本外
P("\n5. 凍結參數樣本外（第十部分）：MA × z 回看窗 × 開槓桿門檻 共 27 組，只用 1999~2012 的年化報酬挑最好的")
res = []
for ma, zw, zo in itertools.product((150, 200, 250), (126, 252, 504), (0.0, 0.5, 1.0)):
    S2 = signals(DF, ma=ma, zwin=zw, z_on=zo)
    tw = pd.Series(run(S2, **BASE_KW)["twr"], index=S2.index)
    h1, h2 = tw[tw.index < SPLIT], tw[tw.index >= SPLIT]
    cagr = lambda x: (1 + x).prod() ** (252 / len(x)) - 1
    res.append(dict(ma=ma, zw=zw, zo=zo, h1=cagr(h1), h2=cagr(h2)))
R = pd.DataFrame(res)
R["rank1"] = R.h1.rank(ascending=False); R["rank2"] = R.h2.rank(ascending=False)
best = R.loc[R.h1.idxmax()]
base = R[(R.ma == 200) & (R.zw == 252) & (R.zo == 0.5)].iloc[0]
P(f"  前半最佳：MA{int(best.ma)}、z 窗 {int(best.zw)}、門檻 {best.zo} → 前半 {best.h1 * 100:.1f}%，後半 {best.h2 * 100:.1f}%"
  f"（後半排名 {int(best.rank2)}/27）")
P(f"  報告參數（MA200、252、0.5）：前半 {base.h1 * 100:.1f}%（第 {int(base.rank1)}），後半 {base.h2 * 100:.1f}%（第 {int(base.rank2)}）")
P(f"  27 組後半：中位 {R.h2.median() * 100:.1f}%、最差 {R.h2.min() * 100:.1f}%；QQQ 後半 "
  f"{((1 + S.ret[S.index >= SPLIT]).prod() ** (252 / (S.index >= SPLIT).sum()) - 1) * 100:.1f}%；"
  f"前後半排名相關 {R.rank1.corr(R.rank2, method='spearman'):.2f}")
P(f"  後半勝過 QQQ 的組數：{(R.h2 > (1 + S.ret[S.index >= SPLIT]).prod() ** (252 / (S.index >= SPLIT).sum()) - 1).sum()}/27")
out.close()
