"""第二十二部分：預先排定的美國經濟數據日曆（2023/6 ~ 2026/10）。

只收「事先公告日期與時間」的數據：非農 (NFP)、CPI、FOMC 利率決議，另加每週四初領失業金 (Claims) 當低重要度對照。
時間為美東 (ET)，broker 時間 = 美東 + 7 小時（固定；美歐夏令時間錯開的幾週也是，已用 tick volume 驗證）。
2025/10~11 美國政府關門：9 月非農延到 11/20、10 月非農與 11 月合併在 12/16 公布、10 月 CPI 取消、9 月 CPI 延到 10/24、
11 月 CPI 延到 12/18；2026/1 底短暫關門：1 月非農延到 2/11、1 月 CPI 延到 2/13。
每一筆都用 M1 tick volume 在發布那一分鐘是否爆量核對（verify()）。
"""
import pandas as pd

NFP = """2023-06-02 2023-07-07 2023-08-04 2023-09-01 2023-10-06 2023-11-03 2023-12-08
2024-01-05 2024-02-02 2024-03-08 2024-04-05 2024-05-03 2024-06-07 2024-07-05 2024-08-02 2024-09-06 2024-10-04 2024-11-01 2024-12-06
2025-01-10 2025-02-07 2025-03-07 2025-04-04 2025-05-02 2025-06-06 2025-07-03 2025-08-01 2025-09-05 2025-11-20 2025-12-16
2026-01-09 2026-02-11 2026-03-06 2026-04-03 2026-05-08 2026-06-05 2026-07-02 2026-08-07 2026-09-04 2026-10-02"""

CPI = """2023-06-13 2023-07-12 2023-08-10 2023-09-13 2023-10-12 2023-11-14 2023-12-12
2024-01-11 2024-02-13 2024-03-12 2024-04-10 2024-05-15 2024-06-12 2024-07-11 2024-08-14 2024-09-11 2024-10-10 2024-11-13 2024-12-11
2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11 2025-07-15 2025-08-12 2025-09-11 2025-10-24 2025-12-18
2026-01-13 2026-02-13 2026-03-11 2026-04-10 2026-05-12 2026-06-10 2026-07-14 2026-08-12 2026-09-11"""

FOMC = """2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16"""

ET_TIME = {"NFP": (8, 30), "CPI": (8, 30), "FOMC": (14, 0), "CLAIMS": (8, 30)}


def to_broker(day, hm):
    return pd.Timestamp(day).replace(hour=hm[0], minute=hm[1]) + pd.Timedelta(hours=7)


def calendar(start="2023-06-01", end="2026-10-08"):
    rows = []
    for name, txt in (("NFP", NFP), ("CPI", CPI), ("FOMC", FOMC)):
        for d in txt.split():
            rows.append((name, pd.Timestamp(d)))
    major = {d for _, d in rows}
    # 每週四初領失業金（遇節日改週三；同日已有 NFP/CPI 的不另列）
    hol = {pd.Timestamp(x) for x in ("2023-11-23", "2024-11-28", "2025-11-27", "2025-12-25", "2026-11-26",
                                       "2024-07-04", "2026-01-01", "2025-01-01")}
    for d in pd.date_range(start, end, freq="W-THU"):
        dd = d - pd.Timedelta(days=1) if d in hol else d
        if dd not in major:
            rows.append(("CLAIMS", dd))
    C = pd.DataFrame(rows, columns=["event", "date"])
    C["t"] = [to_broker(d, ET_TIME[e]) for e, d in zip(C.event, C.date)]
    C = C[(C.t >= start) & (C.t <= end)].sort_values("t").reset_index(drop=True)
    return C


def verify(C, m1):
    """發布分鐘 tick volume ÷ 前 30 分鐘中位數；同時找出 ±90 分鐘內最爆量的那分鐘，確認時間沒對錯。"""
    out = []
    for _, r in C.iterrows():
        w = m1.loc[r.t - pd.Timedelta(minutes=90): r.t + pd.Timedelta(minutes=90)]
        pre = m1.loc[r.t - pd.Timedelta(minutes=30): r.t - pd.Timedelta(minutes=1)]
        if len(pre) < 10 or r.t not in m1.index:
            out.append((r.event, r.date, r.t, float("nan"), pd.NaT)); continue
        ratio = m1.tickvol.loc[r.t] / pre.tickvol.median()
        out.append((r.event, r.date, r.t, ratio, w.tickvol.idxmax()))
    return pd.DataFrame(out, columns=["event", "date", "t", "vol_ratio", "peak"])


if __name__ == "__main__":
    from load import load
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    C = calendar()
    V = verify(C, m1)
    V["peak_off"] = (V.peak - V.t).dt.total_seconds() / 60
    pd.set_option("display.width", 200)
    for e in ("NFP", "CPI", "FOMC"):
        x = V[V.event == e]
        print(e, len(x), "量比中位", round(x.vol_ratio.median(), 1), "最爆量就在發布分鐘(±1)",
              (x.peak_off.abs() <= 1).mean().round(2))
        print(x[(x.peak_off.abs() > 1) | (x.vol_ratio < 2)].to_string())
    x = V[V.event == "CLAIMS"]
    print("CLAIMS", len(x), "量比中位", round(x.vol_ratio.median(), 1))
    C.to_csv("part22_calendar.csv", index=False)
