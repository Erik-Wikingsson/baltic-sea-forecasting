# Standard library
import os
from argparse import ArgumentParser

# Third-party
import matplotlib.pyplot as plt
import networkx
import numpy as np
import scipy.spatial
import torch
from torch_geometric.utils.convert import from_networkx

# Local
from .config import load_config_and_datastores
from .datastore.base import BaseRegularGridDatastore
from .graphs import regular_mesh, saving
from .graphs import utils as gutils
from .graphs import vis


def create_graph(
    graph_dir_path: str,
    xy: np.ndarray,
    land_mask: np.ndarray,
    n_max_levels: int,
    hierarchical: bool,
    create_plot: bool,
):
    """
    Create graph components from `xy` grid coordinates and store in
    `graph_dir_path`.

    Creates the following files for all graphs:
    - g2m_edge_index.pt  [2, N_g2m_edges]
    - g2m_features.pt    [N_g2m_edges, d_features]
    - m2g_edge_index.pt  [2, N_m2m_edges]
    - m2g_features.pt    [N_m2m_edges, d_features]
    - m2m_edge_index.pt  list of [2, N_m2m_edges_level], length==n_levels
    - m2m_features.pt    list of [N_m2m_edges_level, d_features],
                         length==n_levels
    - mesh_features.pt   list of [N_mesh_nodes_level, d_mesh_static],
                         length==n_levels

    where
      d_features:
            number of features per edge (currently d_features==3, for
            edge-length, x and y)
      N_g2m_edges:
            number of edges in the graph from grid-to-mesh
      N_m2g_edges:
            number of edges in the graph from mesh-to-grid
      N_m2m_edges_level:
            number of edges in the graph from mesh-to-mesh at a given level
            (list index corresponds to the level)
      d_mesh_static:
            number of static features per mesh node (currently
            d_mesh_static==2, for x and y)
      N_mesh_nodes_level:
            number of nodes in the mesh at a given level

    And in addition for hierarchical graphs:
    - mesh_up_edge_index.pt
        list of [2, N_mesh_updown_edges_level], length==n_levels-1
    - mesh_up_features.pt
        list of [N_mesh_updown_edges_level, d_features], length==n_levels-1
    - mesh_down_edge_index.pt
        list of [2, N_mesh_updown_edges_level], length==n_levels-1
    - mesh_down_features.pt
        list of [N_mesh_updown_edges_level, d_features], length==n_levels-1

    where N_mesh_updown_edges_level is the number of edges in the graph from
    mesh-to-mesh between two consecutive levels (list index corresponds index
    of lower level)


    Parameters
    ----------
    graph_dir_path : str
        Path to store the graph components.
    xy : np.ndarray
        Grid coordinates, expected to be of shape (Nx, Ny, 2).
    n_max_levels : int
        Limit multi-scale mesh to given number of levels, from bottom up
        (default: None (no limit)).
    hierarchical : bool
        Generate hierarchical mesh graph (default: False).
    create_plot : bool
        If graphs should be plotted during generation (default: False).

    Returns
    -------
    None

    """
    os.makedirs(graph_dir_path, exist_ok=True)

    print(f"Writing graph components to {graph_dir_path}")

    grid_xy = torch.tensor(xy)
    pos_max = torch.max(torch.abs(grid_xy))

    #
    # Mesh
    #

    if create_plot:

        def mesh_plot_func(graph, title):
            vis.plot_graph(
                graph,
                title,
                graph_dir_path,
            )
            plt.show()

    else:
        mesh_plot_func = None

    mesh_pos, G_bottom_mesh, save_graphs = (
        regular_mesh.build_regular_mesh_graph(
            xy,
            land_mask,
            max_mesh_levels=n_max_levels,
            hierarchical=hierarchical,
            mesh_plot_function=mesh_plot_func,
        )
    )

    # Save all graphs
    for graph_name, graph in save_graphs.items():
        saving.save_edges_list(graph, graph_name, graph_dir_path)

    # Divide mesh node pos by max coordinate of grid cell
    mesh_pos = [pos / pos_max for pos in mesh_pos]

    # Save mesh positions
    torch.save(
        mesh_pos, os.path.join(graph_dir_path, "mesh_features.pt")
    )  # mesh pos, in float32

    #
    # Grid2Mesh
    #

    # radius within which grid nodes are associated with a mesh node
    # (in terms of mesh distance)
    DM_SCALE = 0.67

    # mesh nodes on lowest level
    vm = G_bottom_mesh.nodes
    vm_xy = np.array([xy for _, xy in vm.data("pos")])

    # compute dm as mean edge length
    edge_lengths = [
        np.linalg.norm(
            G_bottom_mesh.nodes[u]["pos"] - G_bottom_mesh.nodes[v]["pos"]
        )
        for u, v in G_bottom_mesh.edges
    ]
    dm = np.mean(edge_lengths)

    # grid nodes
    Nx, Ny = xy.shape[:2]

    G_grid = networkx.grid_2d_graph(Nx, Ny)
    G_grid.clear_edges()

    # vg features (only pos introduced here)
    nodes_to_remove = []
    for node in G_grid.nodes:
        # Remove the node from the graph if it is a land node
        if land_mask[node[0], node[1]]:
            nodes_to_remove.append(node)
        else:
            # pos is in feature but here explicit for convenience
            G_grid.nodes[node]["pos"] = xy[node[0], node[1]]

    for node in nodes_to_remove:
        G_grid.remove_node(node)

    # add 1000 to node key to separate grid nodes (1000,i,j) from mesh nodes
    # (i,j) and impose sorting order such that vm are the first nodes
    G_grid = gutils.prepend_node_index(G_grid, 1000)

    # build kd tree for grid point pos
    # order in vg_list should be same as in vg_xy
    vg_list = list(G_grid.nodes)
    vg_xy = np.array([G_grid.nodes[node]["pos"] for node in vg_list])
    kdt_g = scipy.spatial.KDTree(vg_xy)

    # now add (all bottom) mesh nodes, include features (pos)
    G_grid.add_nodes_from(G_bottom_mesh.nodes(data=True))

    # Re-create graph with sorted node indices
    # Need to do sorting of nodes this way for indices to map correctly to pyg
    G_g2m = networkx.Graph()
    G_g2m.add_nodes_from(sorted(G_grid.nodes(data=True)))

    # turn into directed graph
    G_g2m = networkx.DiGraph(G_g2m)

    # add edges
    for v in vm:
        # find neighbours (index to vg_xy)
        neigh_idxs = kdt_g.query_ball_point(vm[v]["pos"], dm * DM_SCALE)
        for i in neigh_idxs:
            u = vg_list[i]
            # add edge from grid to mesh
            G_g2m.add_edge(u, v)
            d = np.sqrt(
                np.sum((G_g2m.nodes[u]["pos"] - G_g2m.nodes[v]["pos"]) ** 2)
            )
            G_g2m.edges[u, v]["len"] = d
            G_g2m.edges[u, v]["vdiff"] = (
                G_g2m.nodes[u]["pos"] - G_g2m.nodes[v]["pos"]
            )

    pyg_g2m = from_networkx(G_g2m)

    if create_plot:
        pyg_g2m_reversed = pyg_g2m.clone()
        pyg_g2m_reversed.edge_index = pyg_g2m.edge_index[[1, 0]]
        vis.plot_graph(pyg_g2m_reversed, "Grid-to-mesh", graph_dir_path)
        plt.show()

    #
    # Mesh2Grid
    #

    # start out from Grid2Mesh and then replace edges
    G_m2g = G_g2m.copy()
    G_m2g.clear_edges()

    # build kd tree for mesh point pos
    # order in vm should be same as in vm_xy
    vm_list = list(vm)
    kdt_m = scipy.spatial.KDTree(vm_xy)

    # add edges from mesh to grid
    for v in vg_list:
        # find 4 nearest neighbours (index to vm_xy)
        neigh_idxs = kdt_m.query(G_m2g.nodes[v]["pos"], 4)[1]
        for i in neigh_idxs:
            u = vm_list[i]
            # add edge from mesh to grid
            G_m2g.add_edge(u, v)
            d = np.sqrt(
                np.sum((G_m2g.nodes[u]["pos"] - G_m2g.nodes[v]["pos"]) ** 2)
            )
            G_m2g.edges[u, v]["len"] = d
            G_m2g.edges[u, v]["vdiff"] = (
                G_m2g.nodes[u]["pos"] - G_m2g.nodes[v]["pos"]
            )

    # relabel nodes to integers (sorted)
    G_m2g_int = networkx.convert_node_labels_to_integers(
        G_m2g, first_label=0, ordering="sorted"
    )
    pyg_m2g = from_networkx(G_m2g_int)

    if create_plot:
        vis.plot_graph(pyg_m2g, "Mesh-to-grid", graph_dir_path)
        plt.show()

    # Save g2m and m2g everything
    # g2m
    saving.save_edges(pyg_g2m, "g2m", graph_dir_path)
    # m2g
    saving.save_edges(pyg_m2g, "m2g", graph_dir_path)


def create_graph_from_datastore(
    datastore: BaseRegularGridDatastore,
    datastore_boundary: BaseRegularGridDatastore,
    datastore_atmosphere: BaseRegularGridDatastore,
    output_root_path: str,
    n_max_levels: int = None,
    hierarchical: bool = False,
    create_plot: bool = False,
):
    land_mask = datastore.get_mask(surface=True, stacked=False, invert=True)
    xy = datastore.get_projected_xy("state", stacked=False)

    boundary_mask = datastore_boundary.get_mask(
        surface=True, stacked=False, invert=True
    )
    xy_boundary = datastore_boundary.get_projected_xy("forcing", stacked=False)

    atmosphere_mask = datastore_atmosphere.get_atmosphere_mask(
        stacked=False, invert=True
    )
    xy_atmosphere = datastore_atmosphere.get_projected_xy(
        "forcing", stacked=False
    )

    create_graph(
        graph_dir_path=output_root_path,
        xy=xy,
        land_mask=land_mask,
        n_max_levels=n_max_levels,
        hierarchical=hierarchical,
        create_plot=create_plot,
    )


def cli(input_args=None):
    parser = ArgumentParser(description="Graph generation arguments")
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to neural-lam configuration file",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="multiscale",
        help="Name to save graph as (default: multiscale)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If graphs should be plotted during generation "
        "(default: False)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        help="Limit multi-scale mesh to given number of levels, "
        "from bottom up (default: None (no limit))",
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Generate hierarchical mesh graph (default: False)",
    )
    args = parser.parse_args(input_args)

    assert (
        args.config_path is not None
    ), "Specify your config with --config_path"

    # Load neural-lam configuration and datastore to use
    _, datastore, datastore_boundary, datastore_atmosphere = (
        load_config_and_datastores(config_path=args.config_path)
    )

    create_graph_from_datastore(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=datastore_atmosphere,
        output_root_path=os.path.join(datastore.root_path, "graph", args.name),
        n_max_levels=args.levels,
        hierarchical=args.hierarchical,
        create_plot=args.plot,
    )


if __name__ == "__main__":
    cli()
