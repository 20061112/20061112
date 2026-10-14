"""建立 session 表並存檔（part22_sessions.pkl）。"""
import pickle, time
from part22_core import build
t = time.time()
d, D, S = build()
pickle.dump((D, S), open("part22_sessions.pkl", "wb"))
print(f"{len(D)} 天、{len(S)} 個 session，{time.time()-t:.0f}s")
print(S.groupby(["sess", "year"]).size().unstack())
print(S.groupby("sess")[["orw_atr", "va_w_atr"]].median())
print(S.groupby("sess").otype.value_counts(normalize=True).unstack().round(2))
print(S.groupby("sess").open_loc.value_counts(normalize=True).unstack().round(2))
