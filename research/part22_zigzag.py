"""第二十二部分：ZigZag 結構（HH / HL / LH / LL）碰觸後的順勢 / 逆勢操作。

資料：data/XAUUSD_M15_2023_2026.csv（2023/1/3 ~ 2026/10/9），H1 由 M15 重組。
依第十部分的教訓，一開始就用 2023~2026 全期間，逐年報告。

1. ZigZag（無未來資訊）：從最近極值反向走 k × ATR(50) 才確認轉折，轉折從「確認那根收盤後」才可使用。
2. 標記：高點和前一個高點比 → HH / LH；低點和前一個低點比 → HL / LL。
   趨勢：最近高 = HH 且最近低 = HL → 多頭；LH + LL → 空頭；其他 → 混合。
3. 滾動視窗：只看極值發生在最近 N 根內的轉折；價格第一次碰到該轉折價位 = 一個事件。
4. 事件分類（多頭為例，空頭鏡像）：
   A 順勢突破：碰 HH 且收盤站上 → 做多      B 逆勢做頂：碰 HH 但收回下方 → 做空
   C 順勢回檔：碰 HL 但收回上方 → 做多      D 逆勢破結構：碰 HL 且收破 → 做空
   另外「掛單在價位」版本（不看收盤）：碰到 HH/LL 就順勢追（A0），碰到 HL/LH 就順勢接（C0）；
   逆勢版本恰好是同一價位的反向，報酬約為相反數再扣成本。
5. 評估：前瞻報酬（ATR 倍數，扣點差）＋扣掉當年漂移（黃金 2023~2026 大漲，做多天生占便宜）；
   交易模擬：收盤進場、停損 1.5 ATR、停利 2R、最長 48 根，同根同時碰到停損停利算停損。
   t 值以「日」為群聚計算（同一天的事件彼此相關）。
"""
import numpy as np
import pandas as pd
from load import load

M15 = load("data/XAUUSD_M15_2023_2026.csv")
M15 = M15[~M15.index.duplicated()]
HORIZONS = (4, 16, 48)
STOP_ATR, TARGET_R, MAX_HOLD = 1.5, 2.0, 48


def resample(df, rule):
    o = df.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                               "tickvol": "sum", "spread": "median"})
    return o.dropna()


def prep(df):
    d = df.copy()
    pc = d.close.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(50).mean()
    d["sp"] = d.spread * 0.01
    return d.dropna()


def zigzag(d, k):
    """回傳轉折表：type H/L、price、ext（極值所在 bar）、conf（確認 bar）、label。"""
    h, l, a = d.high.values, d.low.values, d.atr.values
    piv = []
    dirn, ext, ext_i = 1, h[0], 0
    for i in range(1, len(d)):
        if dirn == 1:
            if h[i] > ext:
                ext, ext_i = h[i], i
            elif l[i] <= ext - k * a[i]:
                piv.append(["H", ext, ext_i, i])
                dirn, ext, ext_i = -1, l[i], i
        else:
            if l[i] < ext:
                ext, ext_i = l[i], i
            elif h[i] >= ext + k * a[i]:
                piv.append(["L", ext, ext_i, i])
                dirn, ext, ext_i = 1, h[i], i
    P = pd.DataFrame(piv, columns=["type", "price", "ext", "conf"])
    lab = []
    last = {"H": None, "L": None}
    for t, p in zip(P.type, P.price):
        prev = last[t]
        if prev is None:
            lab.append(None)
        elif t == "H":
            lab.append("HH" if p > prev else "LH")
        else:
            lab.append("HL" if p > prev else "LL")
        last[t] = p
    P["label"] = lab
    return P.dropna().reset_index(drop=True)


def events(d, P, N):
    """逐根滾動：視窗內（極值在最近 N 根）、已確認、尚未被碰過的轉折價位，第一次被碰到就記一筆事件。"""
    h, l, c = d.high.values, d.low.values, d.close.values
    conf = P.conf.values
    order = np.argsort(conf, kind="stable")
    j = 0
    active = []
    lastH = lastL = None  # 最近確認的高 / 低點標籤
    out = []
    for t in range(len(d)):
        # 趨勢與可用價位只用 t 之前確認的轉折
        while j < len(order) and conf[order[j]] < t:
            r = P.iloc[order[j]]
            active.append(r)
            if r.type == "H":
                lastH = r.label
            else:
                lastL = r.label
            j += 1
        if lastH is None or lastL is None:
            continue
        trend = 1 if (lastH == "HH" and lastL == "HL") else -1 if (lastH == "LH" and lastL == "LL") else 0
        keep = []
        for r in active:
            if t - r.ext > N:
                continue
            hit = h[t] >= r.price if r.type == "H" else l[t] <= r.price
            if not hit:
                keep.append(r)
                continue
            through = 1 if r.type == "H" else -1
            beyond = (c[t] - r.price) * through > 0
            out.append((t, r.price, r.type, r.label, trend, through, beyond, lastH, lastL))
        active = keep
    E = pd.DataFrame(out, columns=["i", "level", "ptype", "plabel", "trend", "through", "beyond", "lastH", "lastL"])
    return E


def classify(E):
    """A/B/C/D 及掛單版 A0/C0，dir = 交易方向。"""
    s = pd.Series("", index=E.index)
    d = pd.Series(0, index=E.index)
    up, dn = E.trend == 1, E.trend == -1
    ext_lvl = (up & (E.plabel == "HH")) | (dn & (E.plabel == "LL"))   # 趨勢的延伸極值
    pb_lvl = (up & (E.plabel == "HL")) | (dn & (E.plabel == "LH"))    # 趨勢中的回檔轉折
    tr = E.trend
    s[ext_lvl & E.beyond], d[ext_lvl & E.beyond] = "A", tr
    s[ext_lvl & ~E.beyond], d[ext_lvl & ~E.beyond] = "B", -tr
    s[pb_lvl & ~E.beyond], d[pb_lvl & ~E.beyond] = "C", tr
    s[pb_lvl & E.beyond], d[pb_lvl & E.beyond] = "D", -tr
    E = E.assign(setup=s, dir=d, ext_lvl=ext_lvl, pb_lvl=pb_lvl)
    return E


def simulate(d, E, at_level=False):
    """交易模擬：停損 STOP_ATR × ATR、停利 TARGET_R、最長 MAX_HOLD 根，扣點差，回傳 R。
    at_level=False：事件 K 棒收盤進場；True：在轉折價位掛單（成交價 E.fill），
    事件 K 棒本身只檢查停損（保守：不知道 K 棒內先後順序），停利從下一根開始算。"""
    c, h, l, a, sp = (d[x].values for x in ("close", "high", "low", "atr", "sp"))
    n = len(d)
    i = E.i.values
    res = np.full(len(E), np.nan)
    for k, (t, dr) in enumerate(zip(i, E.dir.values)):
        if dr == 0 or t + 1 >= n:
            continue
        e = E.fill.values[k] if at_level else c[t]
        st = e - dr * STOP_ATR * a[t]
        tg = e + dr * TARGET_R * STOP_ATR * a[t]
        r = None
        if at_level and (l[t] <= st if dr == 1 else h[t] >= st):
            res[k] = -1.0 - sp[t] / (STOP_ATR * a[t])
            continue
        for u in range(t + 1, min(t + 1 + MAX_HOLD, n)):
            hit_s = l[u] <= st if dr == 1 else h[u] >= st
            hit_t = h[u] >= tg if dr == 1 else l[u] <= tg
            if hit_s:
                r = -1.0
                break
            if hit_t:
                r = TARGET_R
                break
        if r is None:
            r = dr * (c[min(t + MAX_HOLD, n - 1)] - e) / (STOP_ATR * a[t])
        res[k] = r - sp[t] / (STOP_ATR * a[t])
    return res


def forward(d, E):
    c, h, l, a, sp, o = (d[x].values for x in ("close", "high", "low", "atr", "sp", "open"))
    n = len(d)
    i = E.i.values
    A = a[i]
    cost = sp[i] / A
    # 價位掛單成交價：跳空越過價位時以開盤成交（停損單較差、限價單較好）
    th = E.through.values
    fill = np.where(th == 1, np.maximum(E.level.values, o[i]), np.minimum(E.level.values, o[i]))
    E["fill"] = fill
    for H in HORIZONS:
        j = np.minimum(i + H, n - 1)
        E[f"cl{H}"] = (c[j] - c[i]) / A               # 收盤進場，未帶方向
        E[f"lv{H}"] = (c[j] - fill) / A               # 價位進場，未帶方向
    E["cost"] = cost
    E["R"] = simulate(d, E)
    E["time"] = d.index[i]
    E["year"] = E.time.dt.year
    return E


def drift(d):
    """每年所有 bar 的平均前瞻報酬（ATR），用來扣掉方向漂移。"""
    c, a = d.close.values, d.atr.values
    out = {}
    for H in HORIZONS:
        f = (np.roll(c, -H) - c) / a
        f[-H:] = np.nan
        out[H] = pd.Series(f, index=d.index).groupby(d.index.year).mean()
    return out


def ct(x, day):
    """依日群聚的 t 值。"""
    x = pd.Series(np.asarray(x, float))
    day = pd.Series(np.asarray(day))
    m = x.mean()
    g = (x - m).groupby(day.values).sum()
    se = np.sqrt((g ** 2).sum()) / len(x)
    return m / se if se > 0 else np.nan


def summarize(E, dr, col_pref="cl", H=16):
    """回傳帶方向、扣成本、扣漂移的前瞻報酬。"""
    y = E.year.values
    raw = E.dir.values * E[f"{col_pref}{H}"].values - E.cost.values
    adj = raw - E.dir.values * dr[H].reindex(y).values
    return raw, adj


SETUP_NAME = {"A": "A 順勢突破（碰 HH/LL 收盤突破，順勢）",
              "B": "B 逆勢做頂底（碰 HH/LL 收回，逆勢）",
              "C": "C 順勢回檔（碰 HL/LH 收回，順勢）",
              "D": "D 逆勢破結構（碰 HL/LH 收破，逆勢）"}


def report_setup(E, dr, name, mask, col="cl", H=16):
    S = E[mask]
    if len(S) < 20:
        print(f"  {name:38s} n={len(S)} 太少")
        return
    raw, adj = summarize(S, dr, col, H)
    day = S.time.dt.date.values
    R = S.R.values
    yrs = "  ".join(f"{yy}:{adj[S.year.values == yy].mean():+.2f}" for yy in sorted(S.year.unique()))
    line = (f"  {name:38s} n={len(S):5d} 多空 {(S.dir == 1).mean():.0%}多 | {H}根 {raw.mean():+.3f} "
            f"扣漂移 {adj.mean():+.3f} (t {ct(adj, day):+.2f})")
    r = R[~np.isnan(R)]
    if len(r):
        gp, gl = r[r > 0].sum(), -r[r < 0].sum()
        line += f" | 交易 {r.mean():+.3f}R PF {gp / gl:.2f}"
    print(line + f" | 逐年扣漂移 {yrs}")


def run(tf, k, N, verbose=True):
    d = prep(M15 if tf == "M15" else resample(M15, "1h"))
    P = zigzag(d, k)
    E = forward(d, classify(events(d, P, N)))
    dr = drift(d)
    if verbose:
        print(f"\n### {tf} ZigZag k={k} ATR, 視窗 N={N} 根：轉折 {len(P)} 個，事件 {len(E)} 筆"
              f"（多頭 {(E.trend == 1).mean():.0%} / 空頭 {(E.trend == -1).mean():.0%} / 混合 {(E.trend == 0).mean():.0%}）")
    return d, P, E, dr


def main_table(d, E, dr, H=16):
    for s in "ABCD":
        report_setup(E, dr, SETUP_NAME[s], E.setup == s, "cl", H)
    # 掛單版：碰到就進場（不看收盤），進場價 = 轉折價位
    E0 = E.copy()
    E0["dir"] = np.where(E0.ext_lvl, E0.trend, np.where(E0.pb_lvl, E0.trend, 0))
    E0["R"] = simulate(d, E0, at_level=True)
    report_setup(E0, dr, "A0 碰 HH/LL 價位即順勢追（掛停損單）", E0.ext_lvl, "lv", H)
    report_setup(E0, dr, "C0 碰 HL/LH 價位即順勢接（掛限價單）", E0.pb_lvl, "lv", H)


def plot_example(d, P, E, start="2025-10-06", end="2025-10-25"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    w = d.loc[start:end]
    fig, ax = plt.subplots(figsize=(14, 6), dpi=110)
    ax.plot(w.index, w.close, color="#9a9890", lw=1)
    Q = P[(P.ext >= d.index.get_loc(w.index[0])) & (P.ext <= d.index.get_loc(w.index[-1]))]
    xs = d.index[Q.ext.values]
    ax.plot(xs, Q.price, color="#2a78d6", lw=2)
    col = {"HH": "#008300", "HL": "#1baf7a", "LH": "#eb6834", "LL": "#e34948"}
    for x, p, lab, t in zip(xs, Q.price, Q.label, Q.type):
        ax.annotate(lab, (x, p), xytext=(0, 8 if t == "H" else -14), textcoords="offset points",
                    ha="center", fontsize=9, color="#0b0b0b",
                    bbox=dict(boxstyle="round,pad=0.15", fc=col[lab], alpha=.35, lw=0))
    Ew = E[(E.time >= w.index[0]) & (E.time <= w.index[-1]) & (E.setup != "")]
    for _, r in Ew.iterrows():
        ax.scatter(r.time, d.close.iloc[r.i], marker="^" if r.dir == 1 else "v", s=70,
                   color="#008300" if r.dir == 1 else "#e34948", zorder=5)
        ax.annotate(r.setup, (r.time, d.close.iloc[r.i]), xytext=(6, 0), textcoords="offset points", fontsize=9)
    ax.set_title(f"XAUUSD H1 ZigZag（k=3 ATR）結構標記與事件 {start} ~ {end}（▲多 ▼空，字母 = setup）")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig("part22_example.png")


def plot_grid(G):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    G = G.copy()
    G["cfg"] = G.tf + " k" + G.k.astype(str) + " N" + G.N.astype(str)
    M = G.pivot(index="cfg", columns="setup", values="adj16").reindex(G.cfg.unique())
    M = M[["A", "B", "C", "D", "A0", "C0"]]
    fig, ax = plt.subplots(figsize=(8, 9), dpi=110)
    v = 0.8
    im = ax.imshow(M.values, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(M.shape[1]), ["A 順勢突破", "B 逆勢頂底", "C 順勢回檔", "D 逆勢破結構", "A0 掛單追", "C0 掛單接"],
                  rotation=30, ha="right")
    ax.set_yticks(range(M.shape[0]), M.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                    color="#0b0b0b" if abs(M.values[i, j]) < .5 else "white")
    fig.colorbar(im, ax=ax, label="16 根後報酬（ATR，扣點差、扣當年漂移）")
    ax.set_title("各 setup × ZigZag 設定")
    fig.tight_layout()
    fig.savefig("part22_grid.png")


def main():
    print("=" * 100)
    print("一、主要設定：M15，k = 3 ATR，視窗 192 根（2 天）")
    d, P, E, dr = run("M15", 3, 192)
    print("  各年所有 bar 的 16 根平均漂移（ATR）：", "  ".join(f"{y}:{v:+.3f}" for y, v in dr[16].items()))
    for H in HORIZONS:
        print(f"\n  -- 前瞻 {H} 根 --")
        main_table(d, E, dr, H)

    print("\n  -- 標籤 × 趨勢 × 收盤是否突破：通過方向（突破方向）16 根報酬，扣漂移、未扣成本 --")
    E["thr16"] = E.through * E.cl16 - E.through * dr[16].reindex(E.year.values).values
    tb = E.groupby(["trend", "plabel", "beyond"]).thr16.agg(["count", "mean"]).round(3)
    print(tb.to_string())

    print("\n" + "=" * 100)
    print("一b、第二個主要設定：H1，k = 3 ATR，視窗 120 根（5 天）")
    dH, PH, EH, drH = run("H1", 3, 120)
    for H in HORIZONS:
        print(f"\n  -- 前瞻 {H} 根 --")
        main_table(dH, EH, drH, H)

    print("\n" + "=" * 100)
    print("一e、交易模擬逐年（停損 1.5 ATR、停利 2R、最長 48 根，扣點差）：平均 R（筆數）")
    for tag, d_, E_ in (("M15 k3", d, E), ("H1 k3", dH, EH)):
        E0 = E_.copy()
        E0["dir"] = np.where(E0.ext_lvl | E0.pb_lvl, E0.trend, 0)
        E0["R"] = simulate(d_, E0, at_level=True)
        for s_, S in [(x, E_[E_.setup == x]) for x in "ABCD"] + [("A0", E0[E0.ext_lvl]), ("C0", E0[E0.pb_lvl])]:
            g = S.groupby("year").R
            print(f"  [{tag}] {s_:2s} 全期 {S.R.mean():+.3f}R ({len(S)})  " +
                  "  ".join(f"{y}:{m:+.2f}({c})" for (y, m), c in zip(g.mean().items(), g.count())))

    print("\n" + "=" * 100)
    print("一c、不看趨勢標籤：收盤突破就跟（順突破方向）/ 收回就反向（做反彈），依是否與趨勢同向分組（16 根）")
    for tag, d_, E_, dr_ in (("M15 k3", d, E, dr), ("H1 k3", dH, EH, drH)):
        G_ = E_.copy()
        G_["dir"] = np.where(G_.beyond, G_.through, -G_.through)
        G_["R"] = simulate(d_, G_)
        G_["aln"] = np.where(G_.trend == 0, "混合", np.where(G_.dir == G_.trend, "順勢", "逆勢"))
        print(f"  [{tag}]")
        for b, bn in ((True, "收盤突破→跟突破"), (False, "收回→做反彈")):
            for al in ("順勢", "逆勢", "混合"):
                report_setup(G_, dr_, f"{bn}（{al}）", (G_.beyond == b) & (G_.aln == al), "cl", 16)

    print("\n" + "=" * 100)
    print("一d、混合結構拆開：擴張（HH+LL）/ 收斂（LH+HL），收盤突破就跟（16 根）")
    for tag, d_, E_, dr_ in (("M15 k3", d, E, dr), ("H1 k3", dH, EH, drH)):
        G_ = E_[E_.trend == 0].copy()
        G_["dir"] = np.where(G_.beyond, G_.through, -G_.through)
        G_["R"] = simulate(d_, G_)
        G_["shp"] = np.where((G_.lastH == "HH") & (G_.lastL == "LL"), "擴張", "收斂")
        print(f"  [{tag}]")
        for sh in ("擴張", "收斂"):
            for b, bn in ((True, "突破→跟"), (False, "收回→反彈")):
                report_setup(G_, dr_, f"{sh} {bn}", (G_.shp == sh) & (G_.beyond == b), "cl", 16)

    print("\n" + "=" * 100)
    print("二、穩健性：時框 × k × 視窗（16 根，扣漂移；交易 R）")
    rows = []
    for tf, Ns in (("M15", (96, 192, 480)), ("H1", (48, 120, 240))):
        for k in (2, 3, 5):
            for N in Ns:
                d_, P_, E_, dr_ = run(tf, k, N, verbose=False)
                for s in "ABCD":
                    S = E_[E_.setup == s]
                    raw, adj = summarize(S, dr_, "cl", 16)
                    yr = pd.Series(adj).groupby(S.year.values).mean()
                    rows.append(dict(tf=tf, k=k, N=N, setup=s, n=len(S), adj16=adj.mean(),
                                     t=ct(adj, S.time.dt.date.values), R=np.nanmean(S.R.values),
                                     pos_years=f"{(yr > 0).sum()}/{len(yr)}"))
                E0 = E_.copy()
                E0["dir"] = np.where(E0.ext_lvl | E0.pb_lvl, E0.trend, 0)
                for s, m in (("A0", E0.ext_lvl), ("C0", E0.pb_lvl)):
                    S = E0[m]
                    raw, adj = summarize(S, dr_, "lv", 16)
                    yr = pd.Series(adj).groupby(S.year.values).mean()
                    rows.append(dict(tf=tf, k=k, N=N, setup=s, n=len(S), adj16=adj.mean(),
                                     t=ct(adj, S.time.dt.date.values), R=np.nan,
                                     pos_years=f"{(yr > 0).sum()}/{len(yr)}"))
    G = pd.DataFrame(rows)
    G.to_csv("part22_grid.csv", index=False)
    pd.set_option("display.width", 200)
    print(G.round(3).to_string(index=False))
    print("\n  各 setup 在 18 組設定中的平均：")
    print(G.groupby("setup").agg(n=("n", "mean"), adj16=("adj16", "mean"), R=("R", "mean"),
                                 正的組數=("adj16", lambda x: f"{(x > 0).sum()}/{len(x)}")).round(3).to_string())

    E.to_csv("part22_events_M15_k3.csv", index=False)
    EH.to_csv("part22_events_H1_k3.csv", index=False)
    plot_example(dH, PH, EH)
    plot_grid(G)


if __name__ == "__main__":
    main()
