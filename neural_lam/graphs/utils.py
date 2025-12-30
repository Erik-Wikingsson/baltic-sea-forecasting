# Third-party
import networkx
import numpy as np
import scipy
import torch
import torch_geometric as pyg
from torch_geometric.utils.convert import from_networkx


def node_lat_lon_to_cart(node_lat_lon):
    """Convert node positions from lat-lon to cartesian.

    Parameters
    ----------
    node_pos_lat_lon : np.ndarray
        (N_nodes, 2) array, lat-lon coordinates.

    Returns
    -------
    np.ndarray
        (N_nodes, 3) array, cartesian coordinates.
    """
    phi_grid = np.deg2rad(node_lat_lon[:, 0])
    theta_grid = np.deg2rad(90 - node_lat_lon[:, 1])

    cart = np.stack(
        [
            np.cos(phi_grid) * np.sin(theta_grid),
            np.sin(phi_grid) * np.sin(theta_grid),
            np.cos(theta_grid),
        ],
        axis=-1,
    )
    return cart


def sort_nodes_internally(nx_graph):
    # For some reason the networkx .nodes() return list can not be sorted,
    # but this is the ordering used by pyg when converting.
    # This function fixes this.
    H = networkx.DiGraph()
    H.add_nodes_from(sorted(nx_graph.nodes(data=True)))
    H.add_edges_from(nx_graph.edges(data=True))
    return H


def prepend_node_index(graph, new_index):
    # Relabel node indices in graph, insert (graph_level, i, j)
    ijk = [tuple((new_index,) + x) for x in graph.nodes]
    to_mapping = dict(zip(graph.nodes, ijk))
    return networkx.relabel_nodes(graph, to_mapping, copy=True)


def from_networkx_with_start_index(nx_graph, start_index):
    pyg_graph = from_networkx(nx_graph)
    pyg_graph.edge_index += start_index
    return pyg_graph


def add_edge_features_pyg(graph):
    """
    Adds `len` and `vdiff` edge features to given pyg graph
    with a `pos` attribute.
    Modifies graph in-place.
    """
    graph["vdiff"] = (
        graph.pos[graph.edge_index[1]] - graph.pos[graph.edge_index[0]]
    )
    graph["len"] = torch.norm(graph["vdiff"], dim=-1)


def filter_edges_land(
    graph: pyg.data.Data,
    sea_xy: np.ndarray,
    land_xy: np.ndarray,
    max_edge_len: float = 50000,  # in m
):
    """
    Filter edge set to only keep edges not crossing land.
    `graph` is pyg Data object with `edge_index` and `pos` attributes
    """
    # Compute (in pytorch) midpoint of each edge
    send_pos = graph.pos[graph.edge_index[0]]
    rec_pos = graph.pos[graph.edge_index[1]]
    midpoint_pos = (send_pos + rec_pos) / 2
    edge_len = torch.norm(rec_pos - send_pos, dim=1)

    # First filter, absolute edge length
    # NOTE: This is directly in meters
    edge_len_filter = edge_len < max_edge_len

    # Second filter, middle of edge
    # Look up (using numpy and scipy) closest gridpoint
    midpoint_pos_np = midpoint_pos.numpy()
    grid_point_kdt = scipy.spatial.KDTree(
        np.concatenate((sea_xy, land_xy), axis=0)
    )
    closest_grid_index = grid_point_kdt.query(midpoint_pos_np)[1]
    # As sea points come first, can only check magnitude
    # of index of closest point
    midpoint_over_sea = closest_grid_index < sea_xy.shape[0]  # bool np array
    midpoint_filter = torch.tensor(midpoint_over_sea, dtype=bool)

    edge_filter = edge_len_filter & midpoint_filter
    new_edge_index = graph.edge_index[:, edge_filter]

    # Change graph in-place
    graph.edge_index = new_edge_index
