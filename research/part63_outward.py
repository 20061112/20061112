"""第六十三部分：從 IB 突破核心往外延伸 —— 一次只改一個面向 → results_part63.txt、part63_scan.csv

基準 = core_ib.py 的核心（IB 60 分、02:00~10:30 每 30 分、3 小時內 M5 收盤 + 0.05 ATR 突破、IB 另一側停損、收盤出場、
       前一天效率 > 0.633 不做）。
通過標準（只看研究期 2024/11~2025/12）：研究期日 t 值 ≥ 基準，且三個子期間（24/11~12、25H1、25H2）至少 2 個 PF ≥ 基準。
驗證期 2026 只拿來看，不參與挑選。
面向：IB 長度、IB 起點範圍、起點間隔、突破有效時間、確認方式與緩衝、停損、出場、進場方式、疊單上限、方向濾網。
"""
import numpy as np
import pandas as pd
from core_ib import events, simulate, report, PEFF_MAX

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 30)


def cap_stack(X, n):
    """每天每個方向最多 n 筆（保留最早的）。"""
    X = X.sort_values("t_in")
    return X[X.groupby(["day", "side"]).cumcount() < n]


def main():
    rows = []

    def add(dim, lab, X):
        rows.append(dict(面向=dim, 設定=lab, **report(X)))

    base_E = events()
    base = simulate(base_E)
    add("基準", "核心", base)

    for v in (30, 45, 90, 120):
        add("IB 長度（分）", v, simulate(events(ib_min=v)))
    for hrs, lab in (((1, 10.5), "01:00~10:30"), ((2, 6), "02:00~06:00 亞洲"), ((6, 10.5), "06:00~10:30 倫敦前"),
                     ((2, 14), "02:00~14:00"), ((2, 18), "02:00~18:00"), ((12, 18), "12:00~18:00 紐約")):
        add("IB 起點範圍", lab, simulate(events(hours=hrs)))
    for v in (15, 60):
        add("起點間隔（分）", v, simulate(events(step=v)))
    for v in (1, 2, 5, None):
        add("突破有效（小時）", "到收盤" if v is None else v, simulate(events(win_h=v)))
    for c, b in (("touch", 0.0), ("touch", 0.05), ("close1", 0.05), ("close5", 0.0), ("close5", 0.1), ("close5", 0.2),
                 ("close15", 0.05)):
        add("確認 / 緩衝 ATR", f"{c} +{b}", simulate(events(confirm=c, buf=b)))
    for st, lab in (("mid", "IB 中點"), (("cap", 0.3), "IB 另一側，最多 0.3 ATR"), (("cap", 0.5), "IB 另一側，最多 0.5 ATR"),
                    (("atr", 0.25), "進場 ∓ 0.25 ATR"), (("atr", 0.5), "進場 ∓ 0.5 ATR")):
        add("停損", lab, simulate(base_E, stop=st))
    for em in (16, 18, 20, 22):
        add("出場", f"{em}:00 平倉", simulate(base_E, exit_min=em * 60))
    for hm in (180, 360):
        add("出場", f"最長持有 {hm // 60} 小時", simulate(base_E, hold_min=hm))
    for tp in (1, 2, 3, 5):
        add("出場", f"停利 {tp}R", simulate(base_E, tp=tp))
    add("出場", "到 1R 移保本", simulate(base_E, be=1))
    for rb in (15, 60):
        add("進場", f"回測 IB 邊限價（{rb} 分內）", simulate(base_E, entry="retest", retest_bars=rb))
    for n in (1, 2, 3, 5):
        add("疊單上限", f"每天每方向 ≤ {n}", cap_stack(base, n))
    add("方向", "只做順 20 天趨勢", base[np.sign(base.trend20) == base.side])
    add("方向", "只做逆 20 天趨勢", base[np.sign(base.trend20) == -base.side])
    add("方向", "|20 天趨勢| ≥ 0.5", base[base.trend20.abs() >= 0.5])
    add("方向", "只做順前一天方向", base[base.aligned])
    add("方向", "只做逆前一天方向", base[~base.aligned])
    add("星期", "不做週一", base[base.day.dt.dayofweek != 0])
    add("星期", "不做週五", base[base.day.dt.dayofweek != 4])

    R = pd.DataFrame(rows)
    b = R.iloc[0]
    sub = ["24/11~12 PF", "25H1 PF", "25H2 PF"]
    R["子期間勝基準"] = sum((R[c] >= b[c]).astype(int) for c in sub)
    R["通過"] = np.where((R["研究 日t"] >= b["研究 日t"]) & (R["子期間勝基準"] >= 2), "✓", "")
    R.iloc[0, R.columns.get_loc("通過")] = "基準"
    R.to_csv("part63_scan.csv", index=False)
    cols = ["面向", "設定", "通過", "24/11~12 PF", "25H1 PF", "25H2 PF", "研究 PF", "研究 R", "研究 日t", "研究 筆/天", "研究 總",
            "驗證26 PF", "驗證26 R", "驗證26 日t", "驗證26 總"]
    txt = "第六十三部分：從核心往外，一次改一個面向（濾網 peff ≤ %.3f 一律套用；金額 美元/盎司）\n" % PEFF_MAX
    txt += R[cols].to_string(index=False)
    open("results_part63.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
