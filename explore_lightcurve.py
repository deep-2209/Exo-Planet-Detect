import lightkurve as lk
import matplotlib.pyplot as plt

# 1. Search MAST for one quarter of a known hot-Jupiter host
# ==> Kepler-8b is a CONFIRMED planet with deep (approx. 1%) transits (a +ve example)
search_result = lk.search_lightcurve(
    "Kepler-8", author="Kepler", exptime="long", quarter=4
)
print("Search result:")
print(search_result)

# 2. Download that single quarter. Default flux column is PDCSAP (systematics-corrected).
lc = search_result[0].download()
print("\nLightCurve object:")
print(lc)
print("\nFirst 3 times:", lc.time[:3])
print("First 3 fluxes:", lc.flux[:3])
print("Target:", lc.meta.get("OBJECT"), "| Quarter:", lc.meta.get("QUARTER"))

# 3. Plot the raw light curve and save it to a file you can open.
ax = lc.plot()
ax.set_title("Kepler-8 — Quarter 4 (PDCSAP flux)")
plt.savefig("first_lightcurve.png", dpi=150, bbox_inches="tight")
print("\nSaved plot to first_lightcurve.png")