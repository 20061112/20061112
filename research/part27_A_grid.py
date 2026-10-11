"""第二十七部分：版本 A（TPO + IB 突破）出訊號後，用網格取代單筆進場。

訊號：完全照 final_A.py（M5 收盤突破、F1~F4、門檻每月滾動前推），但資料改用 2023/6 ~ 2026/10。
  → 2023/6 ~ 2025/12 是版本 A 開發時**完全沒用過**的資料（版本 A 只用 2026 年）。
每個訊號：E = 進場價、St = IB 另一側（停損）、R = |E − St|；當天收盤全平；停損時全平。
變體（g = R / n）：
  A        原版：1 單位
  AVG_n    攤平：E 進 1 單位，另在 E − j·g（j = 1..n−1）掛買單，全部抱到停損 / 收盤
  GRID_n   底倉 + 回檔網格：E 進 1 單位（抱著），E − j·g 掛買單、賺一格（+g）就平、價格回來再買
  GONLY_n  只有回檔網格（沒有底倉）
  PYR      順勢加碼：E + j·0.5R（j = 1..3）各加 1 單位，共用停損 St
  PYR_BE   同上，第一筆加碼成交後停損移到 E（保本）
（以上以做多描述，做空對稱。）
成交：限價單穿價成交於掛單價，開盤跳空越過則以開盤價（較好）；停損單/停損以掛單價，跳空以開盤價（較差）。
每單位進場扣該根 K 棒點差。M1 內部路徑：陽線 O→L→H→C、陰線 O→H→L→C；下跌段先處理掛單成交再檢查停損（保守）。
損益單位：該訊號的 R（原版一筆停損 = −1R），可跨年比較；每日加總後看日 Sharpe、t、最大回撤。
"""
import numpy as np, pandas as pd
import tpo
tpo.load_m1.__defaults__ = ("data/XAUUSD_M1_2023_2026.csv",)
from part14_deep import ARR
from part17_balance import gen
from final_A import walk_forward, CFG

out = open("results_part27.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

T, _ = walk_forward(gen(**CFG))


def sim(O, H, L, C, S, i0, E, St, s0, orders, base=True, be=False):
    """做多座標。orders: [(種類 'lim'/'stp', 價位, 停利價或 None, 是否重複)]。回傳美元損益。"""
    n = len(orders)
    hold = [False] * n; armed = [True] * n; ent = [0.0] * n
    stop = St; cash = 0.0; pos_base = 1 if base else 0
    if base:
        cash -= E + s0
    for k in range(i0, len(C)):
        pts = (O[k], L[k], H[k], C[k]) if C[k] >= O[k] else (O[k], H[k], L[k], C[k])
        for idx, x in enumerate(pts):
            first = idx == 0
            for m, (kind, lv, tp, rep) in enumerate(orders):            # 1) 限價買單（往下）
                if kind == "lim" and armed[m] and not hold[m] and x <= lv:
                    px = x if first else lv
                    if px <= stop: continue                              # 跳空到停損以下就不成交
                    hold[m] = True; armed[m] = False; ent[m] = px; cash -= px + S[k]
            if x <= stop:                                                # 2) 停損：全平
                px = x if first else stop
                return cash + px * (pos_base + sum(hold))
            for m, (kind, lv, tp, rep) in enumerate(orders):            # 3) 停利 / 順勢買單（往上）
                if hold[m] and tp is not None and x >= tp:
                    px = x if first else tp
                    cash += px; hold[m] = False; armed[m] = rep
                elif kind == "stp" and armed[m] and not hold[m] and x >= lv:
                    px = x if first else lv
                    hold[m] = True; armed[m] = False; ent[m] = px; cash -= px + S[k]
                    if be: stop = max(stop, E)
    return cash + C[-1] * (pos_base + sum(hold))


def variant(name):
    if name == "A": return [], True, False
    if name.startswith("AVG"):
        nn = int(name[3:]); return [("lim", 1 - j / nn, None, False) for j in range(1, nn)], True, False
    if name.startswith("GRID"):
        nn = int(name[4:]); return [("lim", 1 - j / nn, 1 - (j - 1) / nn, True) for j in range(1, nn)], True, False
    if name.startswith("GONLY"):
        nn = int(name[5:]); return [("lim", 1 - j / nn, 1 - (j - 1) / nn, True) for j in range(1, nn)], False, False
    if name.startswith("PYR"):
        return [("stp", 1 + 0.5 * j, None, False) for j in range(1, 4)], True, name.endswith("BE")
    raise ValueError(name)


# 最大曝險（全部掛單成交後被停損的損失，單位 R）
EXPO = {"A": 1, "AVG3": 2, "AVG5": 3, "GRID3": 2, "GRID5": 3, "GONLY5": 2, "PYR": 7, "PYR_BE": 7}
NAMES = list(EXPO)

res = {nm: np.zeros(len(T)) for nm in NAMES}
for t_i, r in enumerate(T.itertuples()):
    t, O, H, L, C, S = ARR[r.day]
    sd = r.side
    if sd == 1:
        o, h, l, c, E, St = O, H, L, C, r.lvl, r.ibl
    else:
        o, h, l, c, E, St = -O, -L, -H, -C, -r.lvl, -r.ibh
    R = E - St
    for nm in NAMES:
        ords, base, be = variant(nm)
        ords = [(k, E - R + R * f if k == "lim" else E + R * (f - 1), None if tp is None else E - R + R * tp, rep)
                for k, f, tp, rep in ords]
        res[nm][t_i] = (sim(o, h, l, c, S, r.i, E, St, r.spread, ords, base, be) - 0) / R
    if t_i % 500 == 0:
        print("trade", t_i, len(T), flush=True)

# 檢查：原版損益與 final_A 模擬一致
P(f"第二十七部分：版本 A 訊號 + 網格（{len(T)} 筆訊號，{T.day.min():%Y-%m-%d} ~ {T.day.max():%Y-%m-%d}）")
P(f"檢查：本程式 A 的每筆 R 與 final_A 模擬的相關 {np.corrcoef(res['A'], T.R)[0, 1]:.4f}，平均 {res['A'].mean():+.3f} vs {T.R.mean():+.3f}")
P("單位：R（原版一筆停損 = −1R）；最大曝險 = 全部掛單成交後被停損的損失（R）")
P("期間：樣本外 = 2023/6~2025/12（版本 A 開發時沒看過），研究期 = 2026")

R_df = pd.DataFrame(res); R_df["day"] = T.day.values; R_df["year"] = T.day.dt.year.values
R_df.to_csv("part27_signals.csv", index=False, float_format="%.4f")
daily = R_df.groupby("day")[NAMES].sum()


def stats(x):
    eq = x.cumsum(); dd = (eq - eq.cummax()).min()
    sh = x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    return dict(每日=x.mean(), 日Sharpe=sh, t=x.mean() / x.std() * np.sqrt(len(x)), 最大回撤=dd, 總除回撤=x.sum() / -dd if dd < 0 else np.inf,
                最差日=x.min())


for lab, sel in (("樣本外 2023/6~2025/12", daily.index.year < 2026), ("研究期 2026", daily.index.year == 2026)):
    P(f"\n{lab}（{sel.sum()} 個交易日、{(R_df.year < 2026).sum() if '樣本外' in lab else (R_df.year == 2026).sum()} 筆訊號）")
    P(f"  {'變體':8s} {'最大曝險':>6s} {'每筆R':>7s} {'每日R':>7s} {'日Sharpe':>8s} {'t':>6s} {'最大回撤R':>9s} {'總/回撤':>7s} {'最差日R':>8s} | 逐年每筆 R")
    for nm in NAMES:
        s = stats(daily.loc[sel, nm])
        per = R_df.loc[(R_df.year < 2026) if "樣本外" in lab else (R_df.year == 2026), nm].mean()
        yr = "  ".join(f"{y}:{R_df.loc[R_df.year == y, nm].mean():+.3f}" for y in sorted(R_df.year.unique())
                       if (y < 2026) == ("樣本外" in lab))
        P(f"  {nm:8s} {EXPO[nm]:6d} {per:+7.3f} {s['每日']:+7.2f} {s['日Sharpe']:+8.2f} {s['t']:+6.2f} {s['最大回撤']:+9.1f} "
          f"{s['總除回撤']:7.2f} {s['最差日']:+8.1f} | {yr}")
out.close()
