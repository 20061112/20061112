"""第五十二部分 A：Range Bar（磚塊）上的 Entropy × Volatility —— 單獨的預測力。

問題：在磚塊序列上，熵特徵（H_trans、MI、p_cont、H_dir）與波動特徵（V_seas、V_roll、J60）、ER60
     能不能預測「下一塊是延續還是反轉」？控制波動之後熵還有沒有資訊？扣掉點差後能不能交易？

可交易的事件定義（避免一根 M1 內產生多塊造成的假延續）：
  事件 = 有磚塊完成的那根 M1；方向 s = 該根最後一塊的方向；特徵 = 該根收盤時可得的值。
  進場 = 該根 M1 收盤價；出場 = 之後「第一根有磚塊完成的 M1」收盤價（h=1），或第 5 個事件（h=5）。
  報酬（以磚塊大小為單位）= s × (出場 − 進場) / 磚塊；成本 = 進場 K 棒點差 / 磚塊。
  正值 = 延續有利，負值 = 反轉有利。
樣本：1/2 ~ 5/31 = 樣本內（IS，用來決定分界與模型係數），6/1 ~ 10/8 = 樣本外（OOS）。
另做 walk-forward：6~10 月每月月初用之前全部資料重估模型。
"""
import numpy as np
import pandas as pd
from tpo import load_m1, build_days
from entropy_vol import build_bricks, brick_features, time_features

out = None                                                      # 只有直接執行本檔才寫入（被 part52_rocket 匯入時不可清空結果檔）
def P(s=""):
    print(s)
    if out: out.write(s + "\n"); out.flush()
CUT = pd.Timestamp("2026-06-01")
N_WIN = 60


def build_events(d, D):
    B = build_bricks(d, D.atr10.to_dict())
    F = brick_features(B, N_WIN)
    raw_cont = (B.s.values[1:] == B.s.values[:-1]).mean()
    same_bar = (B.i.values[1:] == B.i.values[:-1])
    raw_cont_dist = (B.s.values[1:][~same_bar] == B.s.values[:-1][~same_bar]).mean()
    E = F.groupby("i").last().reset_index()                     # 每根 M1 只留最後一塊
    E["nb"] = F.groupby("i").size().values                      # 這根 M1 完成幾塊
    TF = time_features(d)
    for c in ["V_seas", "V_roll", "J60", "ER60", "sER60", "PE"]:
        E[c] = TF[c].values[E.i.values]
    C = d.close.values; S = d.spread.values * 0.01
    E["px_ref"] = E.pop("px")                                  # 這根 M1 最後一塊的邊界價
    E["px"] = C[E.i.values]; E["cost"] = S[E.i.values] / E["size"]
    E["day"] = d.day.values[E.i.values]
    E["sER60"] = E.s * E.sER60                                  # 最近 60 分鐘走勢是否和這塊同方向
    E["sdrift"] = E.s * E.drift                                 # 最近 N 塊淨方向是否和這塊同方向
    for h in (1, 5):
        nxt = np.r_[E.px.values[h:], np.full(h, np.nan)]
        E[f"y{h}"] = E.s * (nxt - E.px) / E["size"]
    # 下一事件是否延續（方向與這塊相同）
    E["cont"] = (np.r_[E.s.values[1:], np.nan] == E.s.values).astype(float)
    E.loc[E.index[-1], "cont"] = np.nan
    # 超出量：進場收盤超過磚塊邊界多少（磚塊單位，0~1）。隨機漫步下 P(延續) = (1 + ovs) / 2，報酬期望 = 0
    E["ovs"] = E.s * (E.px - E.px_ref) / E["size"]
    E["cont_rw"] = (1 + E.ovs) / 2
    # 修正後的延續傾向：最近 N 個「已揭曉」事件的 (實際延續 − 隨機漫步預期)；事件 j 的結果在 j+1 才知道 → shift(1)
    E["ex_cont"] = (E.cont - E.cont_rw).shift(1).rolling(N_WIN, min_periods=N_WIN).mean()
    E = E.dropna(subset=["H_trans", "MI", "p_cont", "V_seas", "y5", "J60", "ER60", "PE", "ex_cont"]).reset_index(drop=True)
    return B, E, raw_cont, raw_cont_dist, same_bar.mean()


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


def quint_table(E, f, y="y1", q=5):
    IS = E[E.t < CUT]
    e = np.unique(np.nanquantile(IS[f], np.linspace(0, 1, q + 1))); e[0], e[-1] = -np.inf, np.inf
    g = pd.cut(E[f], e)
    cells = []
    for c in g.cat.categories:
        a = E[(g == c) & (E.t < CUT)][y]; b = E[(g == c) & (E.t >= CUT)][y]
        cells.append(f"≤{c.right:+.3f}: {a.mean():+.3f}({tstat(a):+.1f})|{b.mean():+.3f}({tstat(b):+.1f})")
    return cells


# ---------------------------------------------------------------- 線性模型：直接預測「做延續」的報酬 y1（磚塊單位）
# 不用 logistic 預測『是否延續』：收盤建構的磚塊，延續率被超出量墊高（見上），會讓模型學到假的「延續優勢」。
def ridge_fit(X, y, lam=1.0):
    X = np.c_[np.ones(len(X)), X]
    R = lam * np.eye(X.shape[1]); R[0, 0] = 0
    return np.linalg.solve(X.T @ X + R, X.T @ y)


def ridge_pred(w, X):
    return np.c_[np.ones(len(X)), X] @ w


def ic(p, y):
    """資訊係數 = 預測與實際報酬的 Pearson 相關。
    不用 Spearman：延續的報酬被超出量弄成「常小賺、偶大賠」的偏態，排序相關會和期望值方向相反。"""
    return np.corrcoef(p, y)[0, 1]


VOL = ["V_seas", "V_roll", "J60"]
ENT = ["H_trans", "MI", "p_cont"]
MODELS = {
    "M0 只有常數": [],
    "M1 波動 V_seas": ["V_seas"],
    "M2 波動 V_seas + V_roll + J60": VOL,
    "M3 熵 H_trans + MI + p_cont": ENT,
    "M3b 修正後延續 ex_cont + ovs": ["ex_cont", "ovs"],
    "M4 波動 + 熵（相加）": VOL + ENT,
    "M5 波動 + 熵 + 交互 H×V": VOL + ENT + ["HxV", "PxV"],
    "M6 M4 + ex_cont + 路徑 ER60/sER60/sdrift": VOL + ENT + ["ex_cont", "ER60", "sER60", "sdrift"],
    "M7 M6 + PE + H_dir + nb + ovs": VOL + ENT + ["ex_cont", "ER60", "sER60", "sdrift", "PE", "H_dir", "nb", "ovs"],
}


def design(E, cols, mu, sd):
    Z = ((E[[c for c in cols if c not in ("HxV", "PxV")]] - mu) / sd).clip(-4, 4)
    if "HxV" in cols: Z["HxV"] = Z["H_trans"] * Z["V_seas"]
    if "PxV" in cols: Z["PxV"] = Z["p_cont"] * Z["V_seas"]
    return Z[cols].values


def fit_predict(tr, te, cols, y="y1"):
    feats = [c for c in cols if c not in ("HxV", "PxV")]
    if not cols:
        return np.full(len(te), tr[y].mean()), None
    mu, sd = tr[feats].mean(), tr[feats].std()
    w = ridge_fit(design(tr, cols, mu, sd), tr[y].values, lam=len(tr) * 0.01)
    return ridge_pred(w, design(te, cols, mu, sd)), w


def rule_stats(pred, Y, cost):
    """規則：預測 > 0 做延續、< 0 做反轉；回傳 毛、淨、t淨，以及最有把握 20%（|預測| 前 20%）的淨與 t。"""
    side = np.where(pred > 0, 1, -1)
    gross = side * Y; net = gross - cost
    top = np.abs(pred) >= np.quantile(np.abs(pred), 0.8)
    return gross.mean(), net.mean(), tstat(net), top.sum(), net[top].mean(), tstat(net[top]), (side == 1).mean()


def eval_models(E, P, y="y1"):
    IS, OS = E[E.t < CUT], E[E.t >= CUT]
    P(f"  {'模型':42s} IS IC   OOS IC  | 規則（預測>0 做延續，否則反轉）毛 / 淨 / t淨  做延續% | 最有把握 20%：筆數 淨 t")
    for name, cols in MODELS.items():
        pi, _ = fit_predict(IS, IS, cols, y); po, _ = fit_predict(IS, OS, cols, y)
        a = ic(pi, IS[y].values) if cols else 0; b = ic(po, OS[y].values) if cols else 0
        g, n, t, k, nt, tt, fc = rule_stats(po, OS[y].values, OS.cost.values)
        P(f"  {name:42s} {a:+.3f}  {b:+.3f}  | {g:+.4f} / {n:+.4f} / {t:+.1f}   {fc:4.0%} | {k:5d} {nt:+.4f} {tt:+.1f}")


def walk_forward(E, cols, y="y1"):
    """6~10 月每月用之前全部資料（從 1 月起）重估，回傳 OOS 預測。"""
    ym = pd.Series(E.t).dt.to_period("M").values
    pr = np.full(len(E), np.nan)
    for m in sorted(set(ym)):
        if m < pd.Period("2026-06"):
            continue
        tr, te = ym < m, ym == m
        pr[te], _ = fit_predict(E[tr], E[te], cols, y)
    return pr


if __name__ == "__main__":
    out = open("results_part52.txt", "w")
    d = load_m1(); D, _ = build_days(d)
    B, E, raw_cont, raw_cont_dist, same_bar = build_events(d, D)
    IS, OS = E[E.t < CUT], E[E.t >= CUT]

    P("第五十二部分 A：Range Bar（磚塊 = 0.05 × ATR10）上的 Entropy × Volatility")
    P(f"  磚塊數 {len(B):,}（平均大小 {B['size'].mean():.2f} 美元）；有磚塊完成的 M1（事件）{len(E):,}；IS {len(IS):,} / OOS {len(OS):,}；熵窗 N = {N_WIN} 塊")
    P(f"  逐塊延續率 {raw_cont:.3f}（含同一根 M1 內的多塊，{same_bar:.1%} 的相鄰磚塊在同一根 M1）→ 只看不同 M1 的相鄰塊：{raw_cont_dist:.3f}")
    P(f"  事件層級（可交易）：下一事件延續率 IS {IS.cont.mean():.3f} / OOS {OS.cont.mean():.3f}")
    P(f"  無條件『做延續』每筆（磚塊單位）h=1：IS 毛 {IS.y1.mean():+.4f}（t {tstat(IS.y1):+.1f}） OOS 毛 {OS.y1.mean():+.4f}（t {tstat(OS.y1):+.1f}）；"
      f"點差成本平均 {E.cost.mean():.3f} 塊（中位 {E.cost.median():.3f}）")
    P(f"  h=5：IS {IS.y5.mean():+.4f}（t {tstat(IS.y5):+.1f}） OOS {OS.y5.mean():+.4f}（t {tstat(OS.y5):+.1f}）")
    P("  注意：同一根 M1 內多塊的『延續』是取樣假象，無法交易；以下一律用事件層級。")

    P("\n  特徵分布（IS）與相關係數")
    cols = ["H_dir", "H_trans", "MI", "p_cont", "ex_cont", "ovs", "sdrift", "PE", "V_seas", "V_roll", "J60", "ER60", "sER60"]
    P("  " + "  ".join(f"{c}: 中位 {IS[c].median():+.3f}" for c in cols))
    Cm = IS[cols].corr()
    P("  " + " " * 8 + "".join(f"{c[:7]:>8s}" for c in cols))
    for c in cols:
        P(f"  {c[:7]:>7s} " + "".join(f"{Cm.loc[c, k]:+8.2f}" for k in cols))

    for y, lab in (("y1", "h=1 下一事件"), ("y5", "h=5 第 5 個事件")):
        P(f"\n  五分位（分界取 IS）→ 做延續的平均報酬（磚塊單位，t），格式 IS | OOS；{lab}")
        for f in cols:
            P(f"  {f:8s} " + "  ".join(quint_table(E, f, y)))

    P("\n  控制波動後熵還有沒有資訊？V_seas 三分位內，p_cont / H_trans 的五分位 Q5 − Q1（h=1，IS | OOS）")
    ev = np.nanquantile(IS.V_seas, [1 / 3, 2 / 3])
    for f in ("p_cont", "H_trans", "MI", "sdrift"):
        cells = []
        for k, (lo, hi) in enumerate([(-np.inf, ev[0]), (ev[0], ev[1]), (ev[1], np.inf)]):
            sub = E[(E.V_seas > lo) & (E.V_seas <= hi)]
            qs = np.nanquantile(sub[sub.t < CUT][f], [0.2, 0.8])
            r = []
            for part in (sub[sub.t < CUT], sub[sub.t >= CUT]):
                lo_q, hi_q = part[part[f] <= qs[0]].y1, part[part[f] > qs[1]].y1
                diff = hi_q.mean() - lo_q.mean()
                se = np.sqrt(hi_q.var() / len(hi_q) + lo_q.var() / len(lo_q))
                r.append(f"{diff:+.3f}({diff / se:+.1f})")
            cells.append(f"V{'低中高'[k]}: " + "|".join(r))
        P(f"  {f:8s} " + "   ".join(cells))

    P("\n  超出量偏誤：進場收盤平均超過磚塊邊界 %.3f 塊 → 隨機漫步下的延續率 %.3f；實際 %.3f（IS）/ %.3f（OOS）"
      % (E.ovs.mean(), E.cont_rw.mean(), IS.cont.mean(), OS.cont.mean()))
    P("  → 延續率 58% 幾乎全部來自建構方式（進場已經離下一塊比較近），不是市場有延續性；"
      "因此 p_cont / H_trans / MI 會和『每塊內的波動』(V_seas) 相關。修正後的 ex_cont 才是真的超額延續。")
    P(f"  ex_cont（實際延續 − 隨機漫步預期，最近 {N_WIN} 事件）：IS 平均 {IS.ex_cont.mean():+.4f}，OOS {OS.ex_cont.mean():+.4f}")
    P("  ex_cont 五分位（h=1）：" + "  ".join(quint_table(E, "ex_cont", "y1")))
    P("  ex_cont 五分位（h=5）：" + "  ".join(quint_table(E, "ex_cont", "y5")))

    for y in ("y1", "y5"):
        P(f"\n  巢狀線性模型（ridge）：預測『做延續』的報酬 {y}（係數只用 IS 估計，OOS = 6/1~10/8；IC = Pearson 相關）")
        eval_models(E, P, y)

    P("\n  Walk-forward（6~10 月每月月初用之前全部資料重估），y1")
    for name in MODELS:
        pr = walk_forward(E, MODELS[name]); m = np.isfinite(pr)
        g, n, t, k, nt, tt, fc = rule_stats(pr[m], E.y1.values[m], E.cost.values[m])
        P(f"  {name:42s} IC {ic(pr[m], E.y1.values[m]) if MODELS[name] else 0:+.3f}  規則淨 {n:+.4f}（t {t:+.1f}）  做延續 {fc:.0%}  最有把握 20% 淨 {nt:+.4f}（t {tt:+.1f}）")

    P("\n  成本敏感度：M6（固定，不依 OOS 挑）在不同成本倍數下的 OOS 每筆淨值（磚塊單位）")
    po, w = fit_predict(IS, OS, MODELS["M6 M4 + ex_cont + 路徑 ER60/sER60/sdrift"])
    side = np.where(po > 0, 1, -1); conf = np.abs(po)
    for qq in (0.0, 0.8, 0.95):
        sel = conf >= np.quantile(conf, qq)
        line = [f"×{k}: {(side[sel] * OS.y1.values[sel] - k * OS.cost.values[sel]).mean():+.4f}" for k in (0, 0.5, 1, 1.5, 2)]
        P(f"  最有把握的 {1 - qq:.0%}（{sel.sum()} 筆）  " + "  ".join(line))
    P("  M6 係數（標準化特徵，y1 單位）： " + "  ".join(f"{c} {v:+.4f}" for c, v in zip(["常數"] + MODELS["M6 M4 + ex_cont + 路徑 ER60/sER60/sdrift"], w)))
    out.close()
