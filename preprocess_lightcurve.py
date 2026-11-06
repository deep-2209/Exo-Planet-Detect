import lightkurve as lk
import matplotlib.pyplot as plt

# Reuse Kepler-8 (cached from step 8)
lc = lk.search_lightcurve("Kepler-8", author="Kepler", exptime="long", quarter=4)[0].download()

processed = (
    lc.remove_nans()                 # drop gaps
      .normalize()                   # baseline -> ~1.0
      .flatten(window_length=401)    # remove slow trends, keep the dip
      .fold(period=3.5225, epoch_time=121.12)           # stack all orbits at Kepler-8b's period
      .bin(time_bin_size=0.01)       # average into fixed-width phase bins
)

ax = processed.plot()
ax.set_title("Kepler-8b — phase-folded, flattened transit")
plt.savefig("folded_transit.png", dpi=150, bbox_inches="tight")
print("Saved folded_transit.png | binned points:", len(processed.flux))