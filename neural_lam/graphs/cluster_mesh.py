# Third-party
import networkx
import numpy as np
import scipy
import torch
import torch_geometric as pyg
import torch_geometric.transforms as pygt
from sklearn.cluster import KMeans
from torch_geometric.utils.convert import from_networkx

# Local
from . import utils as gutils

G2M_REDUCTION = 25
MESH_REDUCTION = 9


def build_graph_from_node_pos(node_pos):
    """
    Build graph using Delaunay triangulation, based on given mesh node pos.
    """
    pos_data = pyg.data.Data(pos=torch.tensor(node_pos, dtype=torch.float32))
    pyg_graph = pygt.Compose(
        [
            pygt.Delaunay(),
            pygt.FaceToEdge(),
        ]
    )(pos_data)
    return pyg_graph


def build_cluster_mesh_graph(
    xy,
    land_mask,
    max_mesh_levels,
    mesh_plot_function,
):
    # TODO Work on projected lat-lons here, rather than toy coordinates in xy
    sea_point_coords = xy.reshape(2, -1)[:, ~land_mask.flatten()].T  # (N, 2)
    sea_coords_proj = sea_point_coords[:10000]

    # As projection means we have fewer points at low latitudes,
    # we can weigh these up in the k-means alg.
    # sea_coords_lat_weights = np.cos(np.deg2rad(sea_coords[:, 0]))  # Not normalized
    sea_coords_lat_weights = None

    num_coords = sea_coords_proj.shape[0]

    possible_mesh_levels = np.floor(
        np.log(sea_coords_proj.shape[0] / G2M_REDUCTION)
        / np.log(MESH_REDUCTION)
    ).astype(int)
    num_mesh_levels = min(possible_mesh_levels, max_mesh_levels)

    # Construct mesh levels
    mesh_level_graphs = []
    mesh_up_graphs = []
    for level_i in range(0, num_mesh_levels):
        print(f"Running kmeans for level {level_i}...")
        if level_i == 0:
            prev_level_pos = sea_coords_proj
            num_clusters = np.round(
                prev_level_pos.shape[0] / G2M_REDUCTION
            ).astype(int)
        else:
            prev_level_pos = mesh_level_graphs[-1].pos.numpy()
            num_clusters = np.round(
                prev_level_pos.shape[0] / MESH_REDUCTION
            ).astype(int)

        mesh_ref_model = KMeans(
            n_clusters=num_clusters,
            init="k-means++",
            n_init=1,
        )

        closest_cluster_index = mesh_ref_model.fit_predict(
            prev_level_pos,
            sample_weight=sea_coords_lat_weights if level_i == 0 else None,
        )

        # m2m
        level_graph = build_graph_from_node_pos(mesh_ref_model.cluster_centers_)
        mesh_level_graphs.append(level_graph)

        if level_i > 0:
            # up
            up_graph = pyg.data.Data(
                edge_index=torch.stack(
                    (
                        torch.arange(prev_level_pos.shape[0], dtype=torch.long),
                        prev_level_pos.shape[0]
                        + torch.tensor(closest_cluster_index, dtype=torch.long),
                    ),
                    dim=0,
                ),
                pos=torch.cat(
                    (
                        mesh_level_graphs[level_i - 1].pos,
                        level_graph.pos,
                    ),
                    dim=0,
                ),
            )
            # TODO

            # down
            # TODO

        if mesh_plot_function is not None:
            mesh_plot_function(level_graph, f"Mesh graph, level {level_i}")

    # Add edge features and convert to networkx
    mesh_levels_nx = []
    for mesh_level in mesh_level_graphs:
        mesh_level["vdiff"] = (
            mesh_level.pos[mesh_level.edge_index[1]]
            - mesh_level.pos[mesh_level.edge_index[0]]
        )
        mesh_level["len"] = torch.norm(mesh_level["vdiff"], dim=-1)

        mesh_levels_nx.append(
            pyg.utils.to_networkx(
                mesh_level,
                node_attrs=["pos"],
                edge_attrs=["vdiff", "len"],
            )
        )

    # Connect mesh levels

    # TODO
    return None, None, None
