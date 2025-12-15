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
from .graphs import cluster_mesh, regular_mesh, saving, vis


def create_graph(
    graph_dir_path: str,
    xy: np.ndarray,
    xy_boundary: np.ndarray,
    xy_atmosphere: np.ndarray,
    g2m_radius: float,
    g2m_radius_atm: float,
    mesh_node_distance: float,
    n_max_levels: int,
    graph_type: str,
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
        Grid (interior) coordinates, expected to be of shape (num_grid, 2).
    xy_boundary : np.ndarray
        Auxiliary grid coordinates, for boundary grid nodes to only use
        in encoding (g2m), expected to be of shape (num_boundary, 2).
    xy_atmosphere : np.ndarray
        Auxiliary grid coordinates, for atmospheric grid nodes to only use
        in encoding (g2m), expected to be of shape (num_atm, 2).
    g2m_radius: float
        Radius to connect interior and boundary nodes within for g2m
    g2m_radius_atm: float
        Radius to connect atmospheric nodes within for g2m
    n_max_levels : int
        Limit multi-scale mesh to given number of levels, from bottom up
        (default: None (no limit)).
    graph_type : str
        What type of graph to generate multiscale/hierarchical/cluster
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

    # Build mesh graph over both interior and boundary nodes
    xy_for_mesh = np.concatenate((xy, xy_boundary), axis=0)

    if graph_type == "cluster":
        mesh_pos, G_bottom_mesh, save_graphs = (
            cluster_mesh.build_cluster_mesh_graph(
                xy_for_mesh,
                limit_mesh_levels=n_max_levels,
                mesh_plot_function=mesh_plot_func,
            )
        )
    else:
        mesh_pos, G_bottom_mesh, save_graphs = (
            regular_mesh.build_regular_mesh_graph(
                xy_for_mesh,
                mesh_node_distance=mesh_node_distance,
                limit_mesh_levels=n_max_levels,
                hierarchical=(graph_type == "hierarchical"),
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

    # mesh nodes on lowest level
    vm = G_bottom_mesh.nodes
    vm_list = list(vm)
    vm_xy = np.array([xy for _, xy in vm.data("pos")])

    # build kd tree for mesh point pos
    kdt_m = scipy.spatial.KDTree(vm_xy)

    # compute dm as mean edge length of bottom mesh graph
    edge_lengths = [
        np.linalg.norm(
            G_bottom_mesh.nodes[u]["pos"] - G_bottom_mesh.nodes[v]["pos"]
        )
        for u, v in G_bottom_mesh.edges
    ]
    dm = np.mean(edge_lengths)
    print(f"dm = {dm}")

    # grid nodes
    interior_nodes = [
        ((1, node_i), {"pos": pos}) for node_i, pos in enumerate(xy)
    ]
    G_interior = networkx.Graph()
    G_interior.add_nodes_from(interior_nodes)

    # build kd tree for grid point pos
    vg_list = list(G_interior.nodes)

    # now add (all bottom) mesh nodes, include features (pos)
    G_g2m = networkx.Graph()
    G_g2m.add_nodes_from(G_bottom_mesh.nodes(data=True))

    # Re-create graph with sorted node indices
    # Need to do sorting of nodes this way for indices to map correctly to pyg
    G_g2m.add_nodes_from(sorted(G_interior.nodes(data=True)))

    # Add auxiliary encoding nodes
    boundary_nodes = [
        ((2, aux_node_i), {"pos": pos})
        for aux_node_i, pos in enumerate(xy_boundary)
    ]
    atmospheric_nodes = [
        ((3, aux_node_i), {"pos": pos})
        for aux_node_i, pos in enumerate(xy_atmosphere)
    ]
    G_g2m.add_nodes_from(boundary_nodes)
    G_g2m.add_nodes_from(atmospheric_nodes)

    # turn into directed graph
    G_g2m = networkx.DiGraph(G_g2m)

    # add edges from each grid node set to mesh
    for node_list, connect_radius in (
        (G_interior.nodes(data=True), g2m_radius),
        (boundary_nodes, g2m_radius),
        (atmospheric_nodes, g2m_radius),
    ):
        # Note: Below could likely be vectorized, if networkx can play along
        for grid_node, node_attrs in node_list:
            # find neighbours (index in mesh graph)
            grid_node_pos = node_attrs["pos"]
            neigh_idxs = kdt_m.query_ball_point(
                grid_node_pos, dm * connect_radius
            )

            #  assert len(neigh_idxs) > 0, (
            #  f"Grid node {grid_node} not connected to mesh in g2m, "
            #  f"increase radius {connect_radius}"
            #  )
            # Add edges for each mesh node within radius
            for mesh_node_i in neigh_idxs:
                mesh_node = vm_list[mesh_node_i]
                mesh_node_pos = vm_xy[mesh_node_i]
                # add edge from grid to mesh
                G_g2m.add_edge(grid_node, mesh_node)
                vdiff = mesh_node_pos - grid_node_pos
                d = np.linalg.norm(vdiff)
                G_g2m.edges[grid_node, mesh_node]["len"] = d
                G_g2m.edges[grid_node, mesh_node]["vdiff"] = vdiff

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

    # add edges from mesh to grid
    # order in vm should be same as in vm_xy
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
    g2m_radius: float,
    g2m_radius_atm: float,
    mesh_node_distance: float,
    n_max_levels: int = None,
    hierarchical: bool = False,
    graph_type: str = "hierarchical",
    create_plot: bool = False,
):
    interior_mask = datastore.get_mask(
        surface=True, stacked=False, invert=False
    )
    xy_interior = datastore.get_projected_xy("state", stacked=False)

    boundary_mask = datastore_boundary.get_mask(
        surface=True, stacked=True, invert=False
    )
    xy_boundary = datastore_boundary.get_projected_xy("forcing", stacked=True)

    atmosphere_mask = datastore_atmosphere.get_atmosphere_mask(
        stacked=True, invert=False
    )
    xy_atmosphere = datastore_atmosphere.get_projected_xy(
        "forcing", stacked=True
    )

    # Node ordering is: 1) interior, 2) boundary, 3) atmosphere
    #  xy_aux = np.concatenate(
    #  (
    #  xy_boundary[boundary_mask],
    #  xy_atmosphere[atmosphere_mask],
    #  ),
    #  axis=0,
    #  )  # (num_aux_grid_nodes, 2)

    create_graph(
        graph_dir_path=output_root_path,
        xy=xy_interior[interior_mask],
        xy_boundary=xy_boundary[
            boundary_mask
        ],  # Only encode from these additional grid nodes
        xy_atmosphere=xy_atmosphere[atmosphere_mask],
        g2m_radius=g2m_radius,
        g2m_radius_atm=g2m_radius_atm,
        mesh_node_distance=mesh_node_distance,
        n_max_levels=n_max_levels,
        graph_type=graph_type,
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
        "--g2m_radius",
        type=float,
        default=0.67,
        help="Radius within which to connect grid nodes (interior and boundary)"
        "to mesh, a multiple of mean edge length in mesh (default: 0.67)",
    )
    parser.add_argument(
        "--g2m_radius_atm",
        type=float,
        default=0.67,
        help="Radius within which to connect grid nodes (atmospheric)"
        "to mesh, a multiple of mean edge length in mesh (default: 0.67)",
    )
    parser.add_argument(
        "--mesh_node_distance",
        type=float,
        default=20000,
        help="Distanc between mesh nodes, in m (default: 20000)",
    )
    parser.add_argument(
        "--type",
        type=str,
        help="Which type of graph structure to generate",
        choices=["multiscale", "hierarchical", "cluster"],
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
        g2m_radius=args.g2m_radius,
        g2m_radius_atm=args.g2m_radius_atm,
        mesh_node_distance=args.mesh_node_distance,
        n_max_levels=args.levels,
        graph_type=args.type,
        create_plot=args.plot,
    )


if __name__ == "__main__":
    cli()
