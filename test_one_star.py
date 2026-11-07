import time
import lightkurve as lk

t = time.time()
sr = lk.search_lightcurve("KIC 10593626", author="Kepler", exptime="long")  # Kepler-22
print(f"search found {len(sr)} products in {round(time.time()-t)}s", flush=True)

t = time.time()
lcc = sr.download_all()
print(f"download_all got {len(lcc)} quarters in {round(time.time()-t)}s", flush=True)

t = time.time()
lc = lcc.stitch()
print(f"stitched to {len(lc.flux)} points in {round(time.time()-t)}s", flush=True)