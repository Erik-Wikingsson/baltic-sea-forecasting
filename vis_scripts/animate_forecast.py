"""
Animate Baltic Sea ensemble forecast vs. CMEMS reanalysis.

Layout per frame (2 rows x 3 cols + 1 cbar col):
  Row 0:  Ground Truth | Ens. Mean | Ens. Std. | [cbar std]
  Row 1:  Member 1     | Member 2  | Member 3  | [cbar main]

One animation per feature/depth level, saved as .gif and .mp4.
Requires ffmpeg for mp4 (install with: conda install -c conda-forge ffmpeg).

Run with the `baltic` conda environment.
"""

import os
import shutil
import zarr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import cmocean
from datetime import datetime, timedelta

FC_PATH  = "data/njord_baltic_example_fc.zarr"
RA_PATH  = "data/cmems_baltic_analysis_2024_plot.zarr"
OUT_DIR  = "plots/animations"
FIG_DPI  = 100
GIF_FPS  = 1
MP4_FPS  = 2
FONTSIZE = 17
TITLE_FS = 17
DISPLAY_MEMBERS = [0, 1, 2]  # zero-indexed ensemble members to show

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load metadata
# ---------------------------------------------------------------------------
fc = zarr.open(FC_PATH)
ra = zarr.open(RA_PATH)

features  = [str(f) for f in fc["state_feature"][:]]
lead_days = fc["lead_time"][:]   # int array, days
n_lt      = len(lead_days)
ra_depth  = ra["depth"][:]

DEPTH_IDX = {
    d: int(np.argmin(np.abs(ra_depth - d))) for d in [1, 9, 28, 47, 91]
}

init_s   = int(fc["init_time"][()])
init_dt  = datetime(1970, 1, 1) + timedelta(seconds=init_s)
base_t   = datetime(2024, 1, 3)
fc_dates = [base_t + timedelta(days=int(t)) for t in fc["time"][:]]

MISSING = -999.0

# ---------------------------------------------------------------------------
# Variable configuration
# ---------------------------------------------------------------------------
FEATURE_CFG = {
    "sla":     dict(base_var="sla",     depth=None, cmap=cmocean.cm.balance, units="m",   long_name="Sea Level Anomaly",     symmetric=True),
    "siconc":  dict(base_var="siconc",  depth=None, cmap=cmocean.cm.ice,     units="",    long_name="Sea Ice Concentration", symmetric=False, vmin=0.0, vmax=1.0),
    "sithick": dict(base_var="sithick", depth=None, cmap=cmocean.cm.ice,     units="m",   long_name="Sea Ice Thickness",     symmetric=False),
}
for base_var, cfg in [
    ("thetao", dict(cmap=cmocean.cm.thermal, units="°C",  long_name="Potential Temperature", symmetric=False)),
    ("so",     dict(cmap=cmocean.cm.haline,  units="PSU", long_name="Salinity",              symmetric=False)),
    ("uo",     dict(cmap=cmocean.cm.balance, units="m/s", long_name="Eastward Current",      symmetric=True)),
    ("vo",     dict(cmap=cmocean.cm.balance, units="m/s", long_name="Northward Current",     symmetric=True)),
]:
    for depth in [1, 9, 28, 47, 91]:
        FEATURE_CFG[f"{base_var}_{depth}m"] = dict(base_var=base_var, depth=depth, **cfg)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_forecast(feat_idx):
    """Return (n_ens, n_lt, n_lat, n_lon) float64."""
    # prediction dims: [ensemble_member, lead_time, state_feature, longitude, latitude]
    data = fc["prediction"][:, :, feat_idx, :, :]    # (n_ens, n_lt, n_lon, n_lat)
    return np.transpose(data.astype(float), (0, 1, 3, 2))  # -> (n_ens, n_lt, n_lat, n_lon)


def load_reanalysis(base_var, depth):
    """Return (n_lt, n_lat, n_lon) float64, land points = NaN."""
    if depth is None:
        data = ra[base_var][:n_lt, :, :]
    else:
        data = ra[base_var][:n_lt, DEPTH_IDX[depth], :, :]
    data = data.astype(float)
    data[np.abs(data - MISSING) < 1.0] = np.nan
    return data


def vrange(fc_data, ra_data, symmetric, cfg):
    if "vmin" in cfg:
        return cfg["vmin"], cfg["vmax"]
    merged = np.concatenate([
        fc_data[np.isfinite(fc_data)].ravel(),
        ra_data[np.isfinite(ra_data)].ravel(),
    ])
    if symmetric:
        v = np.percentile(np.abs(merged), 99)
        return -v, v
    return np.percentile(merged, 1), np.percentile(merged, 99)

# ---------------------------------------------------------------------------
# Animation builder
# ---------------------------------------------------------------------------

def make_animation(feature, feat_idx, cfg):
    base_var  = cfg["base_var"]
    depth     = cfg["depth"]
    cmap      = cfg["cmap"]
    units     = cfg["units"]
    long_name = cfg["long_name"]
    symmetric = cfg["symmetric"]
    unit_str  = f" ({units})" if units else ""

    print("  Loading data ...", flush=True)
    fc_data = load_forecast(feat_idx)           # (n_ens, n_lt, n_lat, n_lon)
    ra_data = load_reanalysis(base_var, depth)  # (n_lt, n_lat, n_lon)

    vmin, vmax = vrange(fc_data, ra_data, symmetric, cfg)

    std_all  = np.nanstd(fc_data, axis=0)       # (n_lt, n_lat, n_lon)
    std_vmax = float(np.nanpercentile(std_all[np.isfinite(std_all)], 99))

    # --- Figure / GridSpec ---
    # 4 columns: [map | map | map | narrow cbar]
    # Each row has its own colorbar in col 3.
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(
        2, 4, figure=fig,
        width_ratios=[1, 1, 1, 0.055],
        hspace=0.12, wspace=0.04,
    )

    ax_re    = fig.add_subplot(gs[0, 0])
    ax_em    = fig.add_subplot(gs[0, 1])
    ax_std   = fig.add_subplot(gs[0, 2])
    ax_cstd  = fig.add_subplot(gs[0, 3])   # std colorbar
    ax_m     = [fig.add_subplot(gs[1, c]) for c in range(3)]
    ax_cmain = fig.add_subplot(gs[1, 3])   # main colorbar

    for ax in [ax_re, ax_em, ax_std] + ax_m:
        ax.set_axis_off()

    kw = dict(origin="lower", aspect="auto", interpolation="nearest")

    # Initial frame (lead time 0)
    im_re  = ax_re.imshow(ra_data[0],                          cmap=cmap,            vmin=vmin, vmax=vmax,  **kw)
    im_em  = ax_em.imshow(np.nanmean(fc_data[:, 0], axis=0),   cmap=cmap,            vmin=vmin, vmax=vmax,  **kw)
    im_std = ax_std.imshow(np.nanstd(fc_data[:, 0], axis=0),   cmap=cmocean.cm.amp,  vmin=0,    vmax=std_vmax, **kw)
    im_ms  = [
        ax_m[i].imshow(fc_data[DISPLAY_MEMBERS[i], 0], cmap=cmap, vmin=vmin, vmax=vmax, **kw)
        for i in range(3)
    ]

    # Std colorbar (right of Ens. Std. panel)
    cb_std = fig.colorbar(im_std, cax=ax_cstd)
    cb_std.set_label(f"Ens. Std.{unit_str}", fontsize=FONTSIZE)
    cb_std.ax.tick_params(labelsize=FONTSIZE - 1)

    # Main colorbar (right of Member 3 panel)
    cb_main = fig.colorbar(im_re, cax=ax_cmain)
    cb_main.set_label(f"{feature}{unit_str}", fontsize=FONTSIZE)
    cb_main.ax.tick_params(labelsize=FONTSIZE - 1)

    # Panel titles
    ax_re.set_title("Analysis",   fontsize=TITLE_FS, pad=4)
    ax_em.set_title("Ens. Mean",  fontsize=TITLE_FS, pad=4)
    ax_std.set_title("Ens. Std.", fontsize=TITLE_FS, pad=4)
    for i, m in enumerate(DISPLAY_MEMBERS):
        ax_m[i].set_title(f"Member {m + 1}", fontsize=TITLE_FS, pad=4)

    # Single animated suptitle
    depth_str = f" at {depth} m" if depth is not None else ""
    init_str  = init_dt.strftime("%Y-%m-%dT%H")
    title_obj = fig.suptitle(
        f"{long_name}{depth_str}, {init_str} + {lead_days[0]} days",
        fontsize=TITLE_FS + 2, y=0.995,
    )

    def update(lt_idx):
        title_obj.set_text(
            f"{long_name}{depth_str}, {init_str} + {lead_days[lt_idx]} days"
        )
        im_re.set_data(ra_data[lt_idx])
        im_em.set_data(np.nanmean(fc_data[:, lt_idx], axis=0))
        im_std.set_data(np.nanstd(fc_data[:, lt_idx], axis=0))
        for i in range(3):
            im_ms[i].set_data(fc_data[DISPLAY_MEMBERS[i], lt_idx])

    anim = animation.FuncAnimation(fig, update, frames=n_lt, interval=800)

    gif_path = os.path.join(OUT_DIR, f"{feature}.gif")
    print(f"  Saving {gif_path} ...", flush=True)
    anim.save(gif_path, writer="pillow", fps=GIF_FPS, dpi=FIG_DPI)

    mp4_path = os.path.join(OUT_DIR, f"{feature}.mp4")
    print(f"  Saving {mp4_path} ...", flush=True)
    if shutil.which("ffmpeg"):
        anim.save(mp4_path, writer="ffmpeg", fps=MP4_FPS, dpi=FIG_DPI,
                  extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
        print(f"  Saved {mp4_path}")
    else:
        print("  MP4 skipped — ffmpeg not found.")
        print("  -> Install with: conda install -c conda-forge ffmpeg")

    plt.close(fig)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for feat_idx, feature in enumerate(features):
        if feature not in FEATURE_CFG:
            print(f"[{feat_idx+1}/{len(features)}] {feature} — no config, skipping")
            continue
        print(f"\n[{feat_idx+1}/{len(features)}] {feature}", flush=True)
        make_animation(feature, feat_idx, FEATURE_CFG[feature])

    print("\nDone! GIFs saved to", OUT_DIR)
