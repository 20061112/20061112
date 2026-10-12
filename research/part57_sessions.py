"""第五十七部分：時區（交易時段）—— 兩個策略各自在哪個時段賺 / 虧，以及兩者的時段配合。

時段（broker 時間 = 紐約 + 7，以進場時間分）：
  亞洲   01~09 點（紐約 18:00~02:59）
  倫敦   10~15 點（倫敦開盤 = 紐約 3:00 = broker 10:00）
  重疊   16~19 點（紐約開盤 9:30 = broker 16:30；倫敦 / 紐約重疊）
  紐約尾 20~23 點
策略：M15 最終版（A+B+S → 上限 8 → 權益濾網 → ATR比>2.2 → 熵 → −15R），IB 核心 + 2R 保本。
時段刪除規則事先固定：只看訓練期（2023/8~2024/12），某策略某時段 PF < 1.05 → 那個策略在那個時段不做；再看驗證期（2025~2026/10）。
（刪的是實際下單；M15 的影子帳本照舊用全部時段的影子單。）
成本：只扣點差（另列 +0.3 滑價）。數量：份 = 0.01 口。
"""
import numpy as np
import pandas as pd
import part56_noslip_layers_mtf as Q

TRAIN_END = Q.TRAIN_END
SESS = [("亞洲 01-09", 1, 9), ("倫敦 10-15", 10, 15), ("重疊 16-19", 16, 19), ("紐約尾 20-23", 20, 23)]
lines = []
Q.lines = lines


def p(s=""):
    print(s, flush=True)
    lines.append(s)


def pf(v):
    v = np.asarray(v)
    l = -v[v < 0].sum()
    return v[v > 0].sum() / l if l > 0 else np.nan


def sess_of(t):
    h = t.dt.hour.values
    out = np.full(len(h), "", dtype=object)
    for lab, a, b in SESS:
        out[(h >= a) & (h <= b)] = lab
    return out


DATA = {}
for slip in (0.0, 0.3):
    m = Q.pipeline(15, Q.ALL, slip=slip)
    m["t"] = m.進場時間
    ib = Q.ib_prep(Q.CI.simulate(Q.CI.events(), be=2), slip)
    ib["t"] = pd.to_datetime(ib.t_in)
    ib = ib[ib.t >= m.t.min().normalize()].reset_index(drop=True)
    for x in (m, ib):
        x["sess"] = sess_of(x.t)
        x["train"] = (x.t < TRAIN_END).values
    DATA[slip] = {"M15": m, "IB": ib}

p("第五十七部分：交易時段（只扣點差；括號 = 扣 0.3 滑價）")
p("時段（broker 時間 = 紐約 + 7）：亞洲 01-09｜倫敦 10-15｜重疊 16-19（紐約開盤 16:30）｜紐約尾 20-23")
DROP = {}
for name, q in (("M15", 2), ("IB", 1)):
    p(f"\n=== {name}（{q} 份）各時段 ===")
    p(f"  {'時段':10s} {'筆數':>6s} {'占比':>4s} | {'每筆$/oz':>14s} | {'PF 全期':>12s} | {'訓練 PF':>12s} {'驗證 PF':>12s} | 逐年 PF 2023 / 2024 / 2025 / 2026 | 損益（{q} 份）")
    x0, x3 = DATA[0.0][name], DATA[0.3][name]
    DROP[name] = []
    for lab, a, b in SESS:
        g0, g3 = x0[x0.sess == lab], x3[x3.sess == lab]
        yr = " / ".join(f"{pf(g0.net[g0.t.dt.year == y]):.2f}" for y in (2023, 2024, 2025, 2026))
        trpf = pf(g0.net[g0.train])
        p(f"  {lab:10s} {len(g0):6d} {len(g0) / len(x0):4.0%} | {g0.net.mean():+6.2f}（{g3.net.mean():+6.2f}） | {pf(g0.net):4.2f}（{pf(g3.net):4.2f}） | "
          f"{trpf:4.2f}（{pf(g3.net[g3.train]):4.2f}） {pf(g0.net[~g0.train]):4.2f}（{pf(g3.net[~g3.train]):4.2f}） | {yr} | {g0.net.sum() * q:+8,.0f}（{g3.net.sum() * q:+8,.0f}）")
        if trpf < 1.05:
            DROP[name].append(lab)
    p("  逐小時（只扣點差）每筆 $/oz / PF / 筆數：")
    hh = x0.groupby(x0.t.dt.hour)
    p("   " + "  ".join(f"{h:02d}點 {g.net.mean():+.2f}/{pf(g.net):.2f}/{len(g)}" for h, g in hh))
    p(f"  事先規則（訓練 PF < 1.05 就不做）→ 刪：{DROP[name] or '無'}")

# ---------------- 時段配合：兩策略在同一時段的曝險與損益 ----------------
p("\n=== 時段配合 ===")
x_m, x_i = DATA[0.0]["M15"], DATA[0.0]["IB"]
tab = pd.DataFrame({
    "M15 筆數": x_m.groupby("sess").size(), "IB 筆數": x_i.groupby("sess").size(),
    "M15 損益(2份)": x_m.groupby("sess").net.sum() * 2, "IB 損益(1份)": x_i.groupby("sess").net.sum()}).reindex([s[0] for s in SESS])
p(tab.round(0).to_string())
# 同時持倉：逐分鐘部位
n = len(Q.IDX)


def pos(x, q):
    D = np.zeros(n + 1)
    np.add.at(D, x.i.values, q); np.add.at(D, x.j.values, -q)
    return np.cumsum(D)[:n]


pm, pi_ = pos(x_m, 2), pos(x_i, 1)
hrs = Q.IDX.hour
p("平均在場口數（0.01 口為單位，含多空總數）依小時：")
p("  M15: " + " ".join(f"{h:02d}:{pm[hrs == h].mean():.1f}" for h in range(1, 24)))
p("  IB : " + " ".join(f"{h:02d}:{pi_[hrs == h].mean():.1f}" for h in range(1, 24)))
both = (pm > 0) & (pi_ > 0)
p(f"兩策略同時有單的分鐘比例 {both.mean():.0%}（M15 有單 {(pm > 0).mean():.0%}、IB 有單 {(pi_ > 0).mean():.0%}）；同時最多 {(pm + pi_).max():.0f} 份")
# 同一天同時段：兩策略的損益相關
dm = (x_m.net * 2).groupby([x_m.t.dt.normalize(), x_m.sess]).sum()
di = x_i.net.groupby([x_i.t.dt.normalize(), x_i.sess]).sum()
jj = pd.concat([dm.rename("M15"), di.rename("IB")], axis=1).dropna()
p("兩策略在同一天同一時段都有單時的損益相關：" + "；".join(
    f"{lab} {jj.xs(lab, level=1).corr().iloc[0, 1]:+.2f}（{len(jj.xs(lab, level=1))} 次）" for lab, a, b in SESS if lab in jj.index.get_level_values(1)))

# ---------------- 刪時段後的組合 ----------------
p("\n=== 組合：依訓練期刪掉弱時段（份 = 0.01 口）===")


def combo(lab, spec, slip, drops):
    parts, ds = [], []
    for k, q in spec.items():
        x = DATA[slip][k]
        x = x[~x.sess.isin(drops.get(k, []))]
        parts.append(Q.mpart(x, q)); ds.append(Q.daily_of(x, q, "t"))
    Q.show(lab, parts, pd.concat(ds, axis=1).fillna(0).sum(1))


for slip in (0.0, 0.3):
    p(f"-- {'只扣點差' if slip == 0 else '扣 0.3 滑價'} --")
    for lab, spec in (("IB 1 + M15 2", {"IB": 1, "M15": 2}), ("IB 2 + M15 2", {"IB": 2, "M15": 2})):
        combo(lab + "（全部時段）", spec, slip, {})
        combo(lab + "（刪訓練期弱時段）", spec, slip, DROP)
    for nm, spec in (("M15 2", {"M15": 2}), ("IB 1", {"IB": 1})):
        combo(nm + "（全部時段）", spec, slip, {})
        combo(nm + "（刪弱時段）", spec, slip, DROP)
        for lab, a, b in SESS:
            combo(f"{nm} 只拿掉 {lab}", spec, slip, {list(spec)[0]: [lab]})
open("results_part57.txt", "w").write("\n".join(lines) + "\n")
