# Third-party
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.cluster import KMeans

G2M_REDUCTION = 25
MESH_REDUCTION = 9

LAM_PROJ = ccrs.LambertConformal(
    central_longitude=20.0,
    central_latitude=60.0,
    standard_parallels=(
        60.0,
        60.0,
    ),
)

static_ds = xr.open_zarr(
    "configs/baltic_model0/data/baltic/static/bathymetry.zarr"
)
surface_mask_da = static_ds.mask.isel(depth=0)

grid_coords = np.stack(
    np.meshgrid(
        surface_mask_da.latitude, surface_mask_da.longitude, indexing="ij"
    ),
    axis=-1,
).reshape(-1, 2)
flat_mask = surface_mask_da.to_numpy().flatten()

sea_coords = grid_coords[flat_mask.astype(bool)]  # (N, 2), lat, lon
sea_coords_proj = LAM_PROJ.transform_points(
    ccrs.PlateCarree(), x=sea_coords[:, 1], y=sea_coords[:, 0]
)[:, :2]

# As projection means we have fewer points at low latitudes,
# we can weigh these up in the k-means alg.
sea_coords_lat_weights = np.cos(np.deg2rad(sea_coords[:, 0]))  # Not normalized

num_coords = sea_coords_proj.shape[0]

print("Running kmeans for g2m...")
g2m_model = KMeans(
    n_clusters=np.round(num_coords / G2M_REDUCTION).astype(int),
    init="k-means++",
    n_init=1,
)
g2m_model.fit(sea_coords_proj, sample_weight=sea_coords_lat_weights)

mesh_pos = [g2m_model.cluster_centers_]
num_mesh_levels = np.floor(
    np.log(mesh_pos[0].shape[0]) / np.log(MESH_REDUCTION)
).astype(int)
for level_i in range(1, num_mesh_levels, 1):
    print("Running kmeans for level {level_i}...")
    prev_level_pos = mesh_pos[-1]
    mesh_ref_model = KMeans(
        n_clusters=np.round(prev_level_pos.shape[0] / MESH_REDUCTION).astype(
            int
        ),
        init="k-means++",
        n_init=1,
    )

    mesh_ref_model.fit(prev_level_pos)
    mesh_pos.append(mesh_ref_model.cluster_centers_)

print("Plotting...")
fig, ax = plt.subplots()
ax.scatter(
    sea_coords_proj[:, 0], sea_coords_proj[:, 1], s=1, label="Grid nodes"
)
for level_i, level_pos in enumerate(mesh_pos, start=1):
    ax.scatter(
        level_pos[:, 0],
        level_pos[:, 1],
        s=8,
        marker="X",
        label=f"Mesh level {level_i} nodes",
    )

ax.legend()
plt.show()
