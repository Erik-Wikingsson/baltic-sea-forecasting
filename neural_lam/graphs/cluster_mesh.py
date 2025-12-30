# Third-party
import numpy as np
import torch
import torch_geometric as pyg
import torch_geometric.transforms as pygt
from sklearn.cluster import KMeans

# Local
from . import utils as gutils


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
    xy_land,
    mesh_refinement_factor=9,
    grid_to_first_mesh_refinement=25,
    limit_mesh_levels=None,
    mesh_plot_function=None,
):
    possible_mesh_levels = np.floor(
        np.log(xy.shape[0] / grid_to_first_mesh_refinement)
        / np.log(mesh_refinement_factor)
    ).astype(int)
    if limit_mesh_levels is None:
        num_mesh_levels = possible_mesh_levels
    else:
        num_mesh_levels = min(possible_mesh_levels, limit_mesh_levels)

    # Construct mesh levels
    mesh_level_graphs = []
    mesh_up_graphs = []
    mesh_down_graphs = []
    for level_i in range(0, num_mesh_levels):
        print(f"Running kmeans for level {level_i}...")
        if level_i == 0:
            prev_level_pos = xy
            num_clusters = np.round(
                prev_level_pos.shape[0] / grid_to_first_mesh_refinement
            ).astype(int)
        else:
            prev_level_pos = mesh_level_graphs[-1].pos.numpy()
            num_clusters = np.round(
                prev_level_pos.shape[0] / mesh_refinement_factor
            ).astype(int)

        mesh_ref_model = KMeans(
            n_clusters=num_clusters,
            init="k-means++",
            n_init=1,
        )

        closest_cluster_index = mesh_ref_model.fit_predict(
            prev_level_pos,
        )

        # m2m
        level_graph = build_graph_from_node_pos(mesh_ref_model.cluster_centers_)
        # Filter out edges crossing land
        gutils.filter_edges_land(level_graph, xy, xy_land)
        gutils.add_edge_features_pyg(level_graph)
        mesh_level_graphs.append(level_graph)

        if mesh_plot_function is not None:
            mesh_plot_function(level_graph, f"Mesh graph, level {level_i}")

        if level_i > 0:
            # up
            up_edge_index = torch.stack(
                (
                    torch.arange(prev_level_pos.shape[0], dtype=torch.long),
                    prev_level_pos.shape[0]
                    + torch.tensor(closest_cluster_index, dtype=torch.long),
                ),
                dim=0,
            )
            up_graph = pyg.data.Data(
                edge_index=up_edge_index,
                pos=torch.cat(
                    (
                        mesh_level_graphs[level_i - 1].pos,
                        level_graph.pos,
                    ),
                    dim=0,
                ),
            )
            gutils.add_edge_features_pyg(up_graph)
            mesh_up_graphs.append(up_graph)

            # down, reverse up edges
            reversed_up_edge_index = torch.stack(
                (
                    up_edge_index[1],
                    up_edge_index[0],
                ),
                dim=0,
            )
            down_graph = pyg.data.Data(
                edge_index=reversed_up_edge_index,
                pos=up_graph.pos,  # same node indices, keep pos as is
            )
            gutils.add_edge_features_pyg(down_graph)
            mesh_down_graphs.append(down_graph)

            if mesh_plot_function is not None:
                mesh_plot_function(
                    down_graph,
                    f"Down graph, {level_i} -> {level_i - 1}",
                )
                mesh_plot_function(
                    up_graph,
                    f"Up graph, {level_i - 1} -> {level_i}",
                )

    # Compile mesh positions
    mesh_pos = [mesh.pos for mesh in mesh_level_graphs]

    # Compile graphs to save
    save_graphs = {
        "mesh_up": mesh_up_graphs,
        "mesh_down": mesh_down_graphs,
        "m2m": mesh_level_graphs,
    }

    # Convert bottom mesh to networkx
    bottom_mesh_nx = pyg.utils.to_networkx(
        mesh_level_graphs[0],
        node_attrs=["pos"],
    )
    # Change pos attribute to be a numpy array instead of list
    for node_id in bottom_mesh_nx.nodes:
        pos_list = bottom_mesh_nx.nodes[node_id]["pos"]
        bottom_mesh_nx.nodes[node_id]["pos"] = np.array(pos_list)

    return mesh_pos, bottom_mesh_nx, save_graphs
