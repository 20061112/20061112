"""第二十三部分 B：回復力的物理模型 —— 彈簧 F = −kx + 慣性 + 以 ER 定義的靜/動摩擦力。

把價格相對均值的偏離當成一個有質量的物體（m = 1，時間單位 = 1 根 M5）：
  位置 x = D = (收盤 − EMA20) / ATR14
  速度 v = (C[t] − C[t−3]) / (3·ATR)      ← 慣性：已經在跑的東西會繼續跑（= 動能）；均值在 t 之後固定（交易看的是價格）
  彈簧 F_spring = −k·x                     ← 回復力
  動摩擦 F_k = −μk·sign(v)，μk = ck × (1 − ER10)   ← 走勢越亂（ER 低）摩擦越大，速度越快被吃掉
  靜摩擦 μs = cs × (1 − ER20)                       ← 停下來之後，彈簧力要大於 μs 才會開始往回走
逐根積分 H = 12 步，得到「預測位移」ΔD_pred；和實際的前瞻 12 根價格報酬（ATR）比較。

比較的模型（參數全部只用 IS = 1~5 月擬合，挑 IS 的 Spearman IC 最大者）：
  M0 純虎克      ΔD_pred = −k·x（沒有慣性、沒有摩擦）→ 和「−D」等價
  M1 彈簧+慣性   沒有摩擦（無阻尼振盪）
  M2 固定摩擦    μk、μs 為常數（不用 ER）
  M3 ER 摩擦     μk = ck(1−ER10)、μs = cs(1−ER20)   ← 使用者的想法
  M4 反向 ER     μk = ck·ER10、μs = cs·ER20          ← 對照：如果把 ER 的方向弄反
  M5 ER 摩擦 + 驅動力：再加一個外力 F_drive = g × VWAP 同側距（第二十二部分最強的動能因子）
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from part22_lib import load_tf, means, atr, er, fwd_returns, restoring_features, context_features, CUT, spearman, events

out = open("results_part23.txt", "a")
def P(s=""): print(s, flush=True); out.write(s + "\n"); out.flush()
CUT1 = pd.Timestamp("2026-04-01")
H = 12


def simulate(x0, v0, k, mk, ms, drive=None, steps=H):
    """向量化：每個 K 棒各自積分。回傳 (ΔD_pred, 最終速度, 是否在第一步就被靜摩擦卡住)。"""
    x, v = x0.copy(), v0.copy()
    drive = np.zeros_like(x) if drive is None else drive
    stuck0 = None
    for i in range(steps):
        Fs = -k * x + drive
        moving = np.abs(v) > 1e-6
        # 移動中：彈簧 + 動摩擦（與速度反向）
        a_mov = Fs - mk * np.sign(v)
        v_mov = v + a_mov
        # 動摩擦只會讓物體停下，不會讓它反向；停下後若彈簧力 ≤ 靜摩擦 → 卡住
        crossed = moving & (np.sign(v_mov) != np.sign(v)) & (np.abs(Fs) <= ms)
        v_mov = np.where(crossed, 0.0, v_mov)
        # 靜止中：彈簧力 > 靜摩擦才開始動（之後受動摩擦）
        go = np.abs(Fs) > ms
        v_rest = np.where(go, Fs - mk * np.sign(Fs), 0.0)
        v_rest = np.where(go & (np.sign(v_rest) != np.sign(Fs)), 0.0, v_rest)
        v = np.where(moving, v_mov, v_rest)
        if i == 0:
            stuck0 = (~moving) & (~go)
        x = x + v
    return x - x0, v, stuck0


def build(d, A, mean):
    c = d.close
    D = (c - mean) / A
    F = pd.DataFrame(index=d.index)
    F["x"] = D
    F["v"] = (c - c.shift(3)) / (3 * A)                  # 價格速度（ATR/根）；錨 = 當下均值，之後固定不動
    F["er10"], F["er20"] = er(c, 10), er(c, 20)
    C = context_features(d, A, mean)
    F["vwap_side"] = C.vwap_side
    F["y"] = fwd_returns(d, A, H)                         # 實際前瞻 12 根價格報酬（ATR）
    F["dD"] = D.shift(-H) - D                             # 實際 D 的變化（含均線移動）
    F["dM"] = -(mean.shift(-H) - mean) / A                # 均線往價格移動造成的 D 變化
    F["u3"] = (c.shift(-3) - c) / (3 * A)                 # 未來 3 根的價格速度（運動方程式回歸用）
    F["x_abs"] = D.abs()
    F["per"] = np.where(F.index < CUT1, "IS1", np.where(F.index < CUT, "IS2", "OOS"))
    return F.iloc[600:].dropna(subset=["x", "v", "er10", "er20", "vwap_side"])


def model_pred(F, name, p):
    x, v = F.x.to_numpy(), F.v.to_numpy()
    e10, e20 = F.er10.to_numpy(), F.er20.to_numpy()
    if name == "M0":
        return -p["k"] * x, None
    if name == "M1":
        return simulate(x, v, p["k"], 0.0, 0.0)[0], None
    if name == "M2":
        return simulate(x, v, p["k"], p["ck"], p["cs"])[0], None
    if name == "M3":
        r = simulate(x, v, p["k"], p["ck"] * (1 - e10), p["cs"] * (1 - e20))
        return r[0], r[2]
    if name == "M4":
        return simulate(x, v, p["k"], p["ck"] * e10, p["cs"] * e20)[0], None
    if name == "M5":
        drive = p["g"] * np.sign(x) * np.clip(F.vwap_side.to_numpy(), 0, None)
        r = simulate(x, v, p["k"], p["ck"] * (1 - e10), p["cs"] * (1 - e20), drive=drive)
        return r[0], r[2]


GRIDS = {
    "M0": dict(k=[0.05]),
    "M1": dict(k=[0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15]),
    "M2": dict(k=[0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15], ck=[0.01, 0.02, 0.05, 0.1, 0.2, 0.4], cs=[0, 0.05, 0.1, 0.2, 0.4]),
    "M3": dict(k=[0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15], ck=[0.01, 0.02, 0.05, 0.1, 0.2, 0.4], cs=[0, 0.05, 0.1, 0.2, 0.4, 0.8]),
    "M4": dict(k=[0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15], ck=[0.01, 0.02, 0.05, 0.1, 0.2, 0.4], cs=[0, 0.05, 0.1, 0.2, 0.4, 0.8]),
}
NAMES = {"M0": "純虎克 F=−kx", "M1": "彈簧+慣性", "M2": "固定摩擦", "M3": "ER 摩擦（你的想法）", "M4": "反向 ER（對照）",
         "M5": "ER 摩擦+VWAP 驅動力"}


def ic_by(F, pred, col="y"):
    s = pd.Series(pred, F.index); m = F.x.abs() >= 1
    return {p: spearman(s[(F.per == p) & m].iloc[::3], F[col][(F.per == p) & m].iloc[::3]) for p in ("IS1", "IS2", "OOS")}


def fit(F, name, grid):
    IS = ((F.per != "OOS") & (F.x.abs() >= 1)).to_numpy()   # 只在有偏離的地方擬合（可交易區）
    Fi = F[IS].iloc[::3]
    best = None
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, vals))
        pr, _ = model_pred(Fi, name, p)
        ic = spearman(pr, Fi.y)
        if best is None or ic > best[0]:
            best = (ic, p)
    return best


if __name__ == "__main__":
    open("results_part23.txt", "w").close()
    d = load_tf("M5"); A = atr(d, 14); M = means(d)
    F = build(d, A, M.ema20)
    P("=" * 110)
    P("第二十三部分 B：彈簧 + 慣性 + ER 摩擦力模型（M5、EMA20、前瞻 12 根；參數只用 1~5 月擬合）")
    P(f"樣本：IS1 {sum(F.per == 'IS1')} 根、IS2 {sum(F.per == 'IS2')} 根、OOS {sum(F.per == 'OOS')} 根")
    P("\n0. F = kx 成立嗎？把 D 的回歸拆成『價格往均線走』和『均線往價格走』（|D| ≥ 1，前瞻 12 根，每 12 根取一點）")
    P("   斜率 = 對 −D 回歸的係數（每 1 ATR 偏離，12 根後拉回多少 ATR）")
    for pp in ("IS1", "IS2", "OOS"):
        Z = F[(F.per == pp) & (F.x.abs() >= 1)].iloc[::12].dropna(subset=["dD", "y", "dM"])
        sl = lambda y: np.polyfit(-Z.x, y, 1)[0]
        P(f"  {pp}: D 的變化 {sl(-Z.dD) * -1:+.3f}（= 價格部分 {sl(Z.y):+.3f} + 均線部分 {sl(Z.dM):+.3f}）  n={len(Z)}")
    P("   → 幾乎全部的『回復』是 EMA 追上價格，價格本身沒有被拉回；在 D 上看 F = kx 很漂亮，但不能交易。")

    P("\n1. 實證運動方程式：未來 3 根價格速度 u（ATR/根）對各『力』做回歸（|D| ≥ 1，每 3 根取一點；t 值在括號）")
    P("   u = a0 + b1·v（慣性） + b2·(−x)（彈簧） + b3·(−sign(v)·(1−ER10))（ER 動摩擦） + b4·(−v·(1−ER10))（ER 黏滯摩擦）"
      " + b5·(−x·(1−ER20))（靜摩擦 × 彈簧）")
    cols = ["慣性 v", "彈簧 −x", "動摩擦 −sgn(v)(1−ER10)", "黏滯 −v(1−ER10)", "−x(1−ER20)"]
    for pp in ("IS1", "IS2", "OOS"):
        Z = F[(F.per == pp) & (F.x.abs() >= 1)].iloc[::3].dropna(subset=["u3"])
        Xr = np.column_stack([np.ones(len(Z)), Z.v, -Z.x, -np.sign(Z.v) * (1 - Z.er10), -Z.v * (1 - Z.er10), -Z.x * (1 - Z.er20)])
        b, *_ = np.linalg.lstsq(Xr, Z.u3, rcond=None)
        res = Z.u3 - Xr @ b
        se = np.sqrt(np.diag(np.linalg.inv(Xr.T @ Xr)) * res.var())
        P(f"  {pp}: " + "  ".join(f"{c} {b[i + 1]:+.4f}({b[i + 1] / se[i + 1]:+.1f})" for i, c in enumerate(cols)))
    P("   讀法：係數 > 0 = 這個力真的存在（方向如物理）；彈簧 < 0 = 價格反而往外。")

    P("\n2. 預測能力（|D| ≥ 1 的 K 棒）：Spearman IC（預測位移 vs 實際前瞻 12 根價格報酬）；括號內 = 對實際 D 變化的 IC")
    fitted = {}
    for name, grid in GRIDS.items():
        ic_is, p = fit(F, name, grid)
        pr, _ = model_pred(F, name, p)
        a, b = ic_by(F, pr, "y"), ic_by(F, pr, "dD")
        fitted[name] = (p, pr)
        P(f"  {name} {NAMES[name]:16s} IS1 {a['IS1']:+.3f} ({b['IS1']:+.3f})  IS2 {a['IS2']:+.3f} ({b['IS2']:+.3f})  "
          f"OOS {a['OOS']:+.3f} ({b['OOS']:+.3f})   參數 {p}")
    # M5：在 M3 最佳參數上再加驅動力
    p3 = fitted["M3"][0]
    best = None
    Fi = F[F.per != "OOS"].iloc[::3]
    Fi = Fi[Fi.x.abs() >= 1]
    for g in (0.0, 0.002, 0.005, 0.01, 0.02, 0.04):
        p = dict(p3, g=g)
        ic = spearman(model_pred(Fi, "M5", p)[0], Fi.y)
        if best is None or ic > best[0]:
            best = (ic, p)
    p5 = best[1]; pr5, _ = model_pred(F, "M5", p5); fitted["M5"] = (p5, pr5)
    a, b = ic_by(F, pr5, "y"), ic_by(F, pr5, "dD")
    P(f"  M5 {NAMES['M5']:16s} IS1 {a['IS1']:+.3f} ({b['IS1']:+.3f})  IS2 {a['IS2']:+.3f} ({b['IS2']:+.3f})  "
      f"OOS {a['OOS']:+.3f} ({b['OOS']:+.3f})   參數 {p5}")

    # 2. 分解：物理狀態 → 實際結果
    P("\n3. M3（ER 摩擦）的物理狀態分類，|D| ≥ 1.5 的 K 棒；實際 = 前瞻 12 根「往均值方向」的價格報酬（ATR，未扣成本）")
    p = fitted["M3"][0]
    x, v = F.x.to_numpy(), F.v.to_numpy()
    mk = p["ck"] * (1 - F.er10.to_numpy()); ms = p["cs"] * (1 - F.er20.to_numpy())
    Fs = -p["k"] * x
    out_v = np.sign(v) == np.sign(x)                      # 速度往外
    state = np.where(out_v & (np.abs(v) > mk), "往外滑（慣性 > 動摩擦）",
             np.where(out_v, "往外但快停（動摩擦吃掉速度）",
             np.where(np.abs(Fs) > ms, "往回滑（彈簧 > 靜摩擦）", "卡住（彈簧 ≤ 靜摩擦）")))
    G = pd.DataFrame({"state": state, "rev": -np.sign(x) * F.y.to_numpy(), "per": F.per.to_numpy(), "far": np.abs(x) >= 1.5,
                      "er10": F.er10.to_numpy()})
    G = G[G.far]
    tab = G.groupby(["state", "per"]).rev.agg(["mean", "count"]).unstack()
    for st in tab.index:
        P(f"  {st:22s} " + "  ".join(f"{pp} {tab.loc[st, ('mean', pp)]:+.3f}（{int(tab.loc[st, ('count', pp)])}）" for pp in ("IS1", "IS2", "OOS")))

    # 3. 預測位移五分位 → 實際報酬
    P("\n4. 預測位移五分位（Q1 = 預測往外最多 = 動能，Q5 = 預測往回最多 = 回歸），|D| ≥ 1.5；實際往均值方向的報酬（ATR）")
    for name in ("M0", "M1", "M3", "M5"):
        pr = fitted[name][1]
        tow = -np.sign(F.x.to_numpy()) * pr                # 往均值方向的預測位移
        far = np.abs(F.x.to_numpy()) >= 1.5
        Z = pd.DataFrame({"p": tow, "r": -np.sign(F.x.to_numpy()) * F.y.to_numpy(), "per": F.per.to_numpy()})[far]
        e = np.unique(np.nanquantile(Z[Z.per != "OOS"].p, np.linspace(0, 1, 6))); e[0], e[-1] = -np.inf, np.inf
        Z["q"] = pd.cut(Z.p, e, labels=False)
        cells = []
        for pp in ("IS1", "IS2", "OOS"):
            m = Z[Z.per == pp].groupby("q").r.mean().reindex(range(len(e) - 1))
            cells.append(f"{pp} " + " ".join(f"{v:+.2f}" for v in m) + f" Δ{m.iloc[-1] - m.iloc[0]:+.2f}")
        P(f"  {name} {NAMES[name]:16s} " + " | ".join(cells))

    # 4. 交易：和第二十二部分相同的事件、出場、成本，方向改由物理模型決定
    from part23_trades import run, report
    P("\n5. 交易（同第二十二部分事件：|D| 第一次 ≥ 2，下一根開盤進場，持有 12 根或 2 ATR 停損，扣點差，單一部位）")
    P("   方向 = 預測位移的方向（往回 → 回歸單、往外 → 動能單）；|預測| 小於 IS 事件中位數的不做")
    ev = events(F.x.reindex(d.index), d.seg, d0=2.0, rearm=1.0)
    allres = {}
    for name in ("M0", "M1", "M2", "M3", "M4", "M5"):
        pr = pd.Series(fitted[name][1], F.index).reindex(d.index)
        Dfull = F.x.reindex(d.index)
        evv = [t for t in ev if not np.isnan(pr.iat[t])]
        thr = np.nanmedian(np.abs(pr.iloc[[t for t in evv if d.index[t] < CUT]]))
        def rule(t, pr=pr, thr=thr, D=Dfull):
            p_ = pr.iat[t]
            if np.isnan(p_) or abs(p_) < thr:
                return 0, ""
            s = int(np.sign(p_))
            return s, ("回歸" if s == -np.sign(D.iat[t]) else "動能")
        T = run(d, A, Dfull, pd.Series(0.0, d.index), 0, 0, rule=rule)
        allres[name] = T
        pf = lambda q: q[q > 0].sum() / -q[q <= 0].sum()
        seg = []
        for per in ("樣本內", "樣本外"):
            s = T[T["期間"] == per]
            seg.append(f"{per} {len(s):3d} 筆 每筆 {s['損益_美元'].mean():+5.2f} 美元 / {s['損益_ATR'].mean():+.3f} ATR PF {pf(s['損益_美元']):.2f} "
                       f"(回歸 {np.mean(s['類型'] == '回歸'):.0%})")
        P(f"  {name} {NAMES[name]:16s} " + " | ".join(seg))
    T = allres["M3"]
    T.to_csv("part23_trades_friction.csv", index=False, encoding="utf-8-sig")
    txt = report(T, "ER 摩擦模型（M3）交易")
    open("part23_stats_friction.txt", "w").write(txt + "\n")

    # 圖：同一個偏離，在不同 ER 下的預測軌跡
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    p = fitted["M3"][0]
    for e_, col in ((0.15, "tab:blue"), (0.4, "tab:green"), (0.7, "tab:orange"), (0.9, "tab:red")):
        xs, xx, vv = [2.0], np.array([2.0]), np.array([0.15])
        for i in range(36):
            dd_, vv, _ = simulate(xx, vv, p["k"], p["ck"] * (1 - np.array([e_])), p["cs"] * (1 - np.array([e_])), steps=1)
            xx = xx + dd_; xs.append(xx[0])
        ax[0].plot(xs, color=col, label=f"ER = {e_}")
    ax[0].axhline(0, color="k", lw=.6); ax[0].set_xlabel("bars ahead (M5)"); ax[0].set_ylabel("D (ATR from EMA20)")
    ax[0].set_title(f"Model path from D=2, v=+0.15/bar  (k={p['k']}, ck={p['ck']}, cs={p['cs']})"); ax[0].legend()
    lab = {"M0": "Hooke only", "M1": "spring+inertia", "M2": "const friction", "M3": "ER friction", "M4": "reversed ER", "M5": "ER friction+VWAP drive"}
    for name, T in allres.items():
        ax[1].plot(T["進場時間"], T["損益_美元"].cumsum(), label=lab[name], lw=2 if name == "M3" else 1)
    ax[1].axvline(CUT, color="k", lw=.8); ax[1].set_title("Trades by model direction: cumulative USD/oz"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("part23_friction.png", dpi=110)

    # 6. 意外發現：「往回滑」（離均值還遠，但價格已往均值走）之後反而續行 → 做成順勢回檔單
    P("\n6. 回檔續行單：|D| ≥ d0 且 3 根價格速度朝向均值（v·D < 0，回檔中）→ 順著偏離方向做（= 等回檔再追）")
    P("   下一根開盤進場、持有 12 根或 2 ATR 停損、扣點差、單一部位；同一段偏離只做第一次（|D| 回到 d0/2 內才重新武裝）")
    from part23_trades import SESS
    o_, h_, l_, c_ = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    spr = d.spread.to_numpy() / 100; seg = d.seg.to_numpy(); a_ = A.to_numpy()
    Dd = F.x.reindex(d.index).to_numpy(); Vd = F.v.reindex(d.index).to_numpy()
    E20 = F.er20.reindex(d.index).to_numpy()

    def pullback_trades(d0, vmin=0.0):
        rows, busy, armed = [], -1, True
        for t in range(600, len(c_) - 2):
            x, v = Dd[t], Vd[t]
            if np.isnan(x) or np.isnan(v):
                continue
            if abs(x) < d0 / 2:
                armed = True
            if t <= busy or not armed or abs(x) < d0 or not (v * x < 0 and abs(v) >= vmin):
                continue
            s = np.sign(x); e = t + 1; ep = o_[e]; st = ep - s * 2 * a_[t]; xp = xi = None; why = "時間"
            for j in range(e, min(e + H, len(c_))):
                if seg[j] != seg[t]:
                    xp, xi, why = c_[j - 1], j - 1, "斷線"; break
                if (l_[j] <= st) if s > 0 else (h_[j] >= st):
                    xp = (min(st, o_[j]) if s > 0 else max(st, o_[j])) if j > e else st; xi, why = j, "停損"; break
            if xp is None:
                xi = min(e + H - 1, len(c_) - 1); xp = c_[xi]
            pnl = s * (xp - ep) - spr[e]
            rows.append(dict(訊號時間=d.index[t], 進場時間=d.index[e], 方向="多" if s > 0 else "空", 進場價=round(ep, 2),
                             停損價=round(st, 2), 出場價=round(xp, 2), 出場原因=why, 損益_美元=round(pnl, 2),
                             損益_ATR=round(pnl / a_[t], 3), 持倉_分鐘=(xi - e + 1) * 5, ATR14=round(a_[t], 2), D=round(x, 2),
                             v=round(v, 3), ER20=round(E20[t], 3), 時段=SESS(d.index[t].hour), 期間="樣本內" if d.index[t] < CUT else "樣本外"))
            busy, armed = xi, False
        return pd.DataFrame(rows)

    pf = lambda q: q[q > 0].sum() / -q[q <= 0].sum()
    for d0 in (1.5, 2.0, 2.5):
        T = pullback_trades(d0)
        cells = []
        for per in ("樣本內", "樣本外"):
            s_ = T[T["期間"] == per]
            cells.append(f"{per} {len(s_):3d} 筆 勝率 {np.mean(s_['損益_美元'] > 0):.0%} 每筆 {s_['損益_美元'].mean():+5.2f} 美元 / {s_['損益_ATR'].mean():+.3f} ATR PF {pf(s_['損益_美元']):.2f}")
        P(f"  d0 {d0}: " + " | ".join(cells))
        if d0 == 2.0:
            T2 = T
    P("\n   d0 = 2.0 依 ER20（摩擦）分三組（門檻 = 樣本內三分位）：ER 高 = 摩擦小")
    e1, e2 = np.nanquantile(T2[T2["期間"] == "樣本內"].ER20, [1 / 3, 2 / 3])
    T2["摩擦"] = np.where(T2.ER20 <= e1, "ER 低（摩擦大）", np.where(T2.ER20 >= e2, "ER 高（摩擦小）", "中"))
    for g_, s_ in T2.groupby("摩擦"):
        P(f"   {g_:12s} " + " | ".join(f"{per} {len(q)} 筆 每筆 {q['損益_ATR'].mean():+.3f} ATR PF {pf(q['損益_美元']):.2f}"
                                       for per, q in s_.groupby("期間")))
    T2.drop(columns="摩擦").to_csv("part23_trades_pullback.csv", index=False, encoding="utf-8-sig")
    txt = report(T2.assign(類型="回檔續行", 損益_R=T2["損益_ATR"] / 2, 最大浮虧_美元=np.nan, 最大浮盈_美元=np.nan, RFI=np.nan,
                           點差=np.nan), "回檔續行單（d0 = 2.0）")
    open("part23_stats_pullback.txt", "w").write(txt + "\n")
