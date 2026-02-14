# Standard library
import os
from argparse import ArgumentParser
from types import SimpleNamespace
from typing import Optional, Tuple

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import torch
import torch_geometric as pyg
import zarr
from torch_geometric.utils import degree

# Local
from .config import load_config_and_datastores
from .datastore.base import BaseRegularGridDatastore
from .graphs import global_cluster_mesh, global_icosahedral_mesh, saving
from .graphs import utils as gutils
from .graphs import vis


def load_grid_from_zarr(dataset_path: str) -> np.ndarray:
    """Load grid from zarr. Expects 'latitude' and 'longitude' arrays.

    Handles 1D (n_points,) or 2D (n_lat, n_lon) arrays. Returns (n_points, 2)
    with columns [longitude, latitude] in degrees (x=lon, y=lat convention).
    """
    root = zarr.open(dataset_path, mode="r")
    lat = np.asarray(root["latitude"]).astype(np.float32)
    lon = np.asarray(root["longitude"]).astype(np.float32)
    if lat.ndim == 1 and lon.ndim == 1:
        if lat.shape[0] == lon.shape[0]:
            return np.stack([lon, lat], axis=1)
        # Assume 2D grid: lat (n_lat,), lon (n_lon,)
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        return np.stack([lon_2d.ravel(), lat_2d.ravel()], axis=1)
    if lat.ndim == 2 and lon.ndim == 2:
        return np.stack([lon.ravel(), lat.ravel()], axis=1)
    raise ValueError(
        f"Unsupported latitude/longitude shapes: {lat.shape}, {lon.shape}"
    )


def load_grid_from_datastore(
    datastore: BaseRegularGridDatastore,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load interior (sea) grid and land grid for global graph.

    Returns
    -------
    grid_xy : (n_interior, 2) interior (sea) points, [longitude, latitude]
    sea_xy : (n_interior, 2) same as grid_xy (for filter_edges_land)
    land_xy : (n_land, 2) land points, [longitude, latitude]
    """
    interior_mask = datastore.get_mask(surface=True, stacked=True, invert=False)
    xy = datastore.get_xy("state", stacked=True)  # (lon, lat) = (x, y)
    sea_xy = xy[interior_mask].astype(np.float32)
    land_xy = xy[~interior_mask].astype(np.float32)
    return sea_xy, sea_xy, land_xy


def _filter_icosahedral_mesh_to_sea(
    mesh_cart: np.ndarray,
    mesh_xy: np.ndarray,
    mesh_edge_index: np.ndarray,
    sea_xy: np.ndarray,
    land_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only icosahedral mesh nodes over sea (nearest sea <= nearest land).

    sea_xy, land_xy are (lon, lat). Returns reduced mesh_cart, mesh_xy,
    mesh_edge_index (reindexed).
    """
    sea_cart = gutils.node_lat_lon_to_cart(sea_xy)
    land_cart = gutils.node_lat_lon_to_cart(land_xy)
    kdt_sea = scipy.spatial.KDTree(sea_cart)
    kdt_land = scipy.spatial.KDTree(land_cart)
    num_mesh = mesh_cart.shape[0]
    d_sea, _ = kdt_sea.query(mesh_cart, k=1)
    d_land, _ = kdt_land.query(mesh_cart, k=1)
    keep_mask = (d_sea <= d_land).ravel()
    old_to_new = np.full(num_mesh, -1, dtype=np.int64)
    new_idx = 0
    for old_idx in range(num_mesh):
        if keep_mask[old_idx]:
            old_to_new[old_idx] = new_idx
            new_idx += 1
    new_mesh_cart = mesh_cart[keep_mask]
    new_mesh_xy = mesh_xy[keep_mask]
    # Keep only edges whose endpoints are both kept; reindex
    kept_src = keep_mask[mesh_edge_index[0]] & keep_mask[mesh_edge_index[1]]
    new_src = old_to_new[mesh_edge_index[0, kept_src]]
    new_dst = old_to_new[mesh_edge_index[1, kept_src]]
    new_mesh_edge_index = np.stack([new_src, new_dst], axis=0)
    return new_mesh_cart, new_mesh_xy, new_mesh_edge_index


def create_global_graph(
    graph_dir_path: str,
    grid_xy: np.ndarray,
    splits: int = 3,
    levels: Optional[int] = None,
    graph_type: str = "multiscale",
    g2m_radius: float = 0.6,
    m2g_k: int = 3,
    create_plot: bool = False,
    connect_disconnected: bool = False,
    sea_xy: Optional[np.ndarray] = None,
    land_xy: Optional[np.ndarray] = None,
    max_edge_len_deg: float = 2.0,
    mesh_refinement_factor: float = 9,
    grid_to_first_mesh_refinement: float = 25,
):
    """Create global graph: icosahedral mesh + g2m + m2g.

    grid_xy, sea_xy, land_xy: (N, 2) [longitude, latitude] in degrees (x=lon, y=lat).

    Node convention for loader: mesh 0..N_mesh-1, grid N_mesh..N_mesh+N_grid-1.
    When sea_xy and land_xy are provided, m2g edges over land are filtered.
    """
    os.makedirs(graph_dir_path, exist_ok=True)
    print(f"Writing global graph to {graph_dir_path}")

    # Build mesh: icosahedral or cluster
    if graph_type == "cluster":
        if sea_xy is None or land_xy is None:
            raise ValueError("graph_type=cluster requires sea_xy and land_xy")
        mesh_pos_list, bottom_mesh, save_graphs_cluster = (
            global_cluster_mesh.build_cluster_mesh_graph_global(
                sea_xy,
                land_xy,
                mesh_refinement_factor=mesh_refinement_factor,
                grid_to_first_mesh_refinement=grid_to_first_mesh_refinement,
                limit_mesh_levels=levels,
                base_max_edge_len_deg=max_edge_len_deg,
            )
        )
        mesh_xy = bottom_mesh.pos.numpy()
        mesh_cart = gutils.node_lat_lon_to_cart(mesh_xy)
        mesh_edge_index = bottom_mesh.edge_index.numpy()
        num_mesh = mesh_xy.shape[0]
        num_grid = grid_xy.shape[0]
        hierarchical_cluster = len(mesh_pos_list) > 1
    else:
        # Icosahedral (multiscale or hierarchical)
        if levels is None:
            levels = 1 if graph_type == "multiscale" else (splits + 1)
        mesh_levels = (
            global_icosahedral_mesh.get_hierarchy_of_triangular_meshes(
                splits=splits, levels=levels
            )
        )
        mesh_cart, mesh_xy, mesh_edge_index = mesh_levels[-1]
        num_mesh = mesh_cart.shape[0]
        num_grid = grid_xy.shape[0]
        hierarchical_cluster = False
        mesh_pos_list = None
        save_graphs_cluster = None

        # Restrict mesh to sea-only when ocean/land mask is provided
        if sea_xy is not None and land_xy is not None and land_xy.shape[0] > 0:
            mesh_cart, mesh_xy, mesh_edge_index = (
                _filter_icosahedral_mesh_to_sea(
                    mesh_cart, mesh_xy, mesh_edge_index, sea_xy, land_xy
                )
            )
            num_mesh = mesh_xy.shape[0]
            print(f"Filtered icosahedral mesh to sea-only: {num_mesh} nodes")

    num_m2m_edges = mesh_edge_index.shape[1]
    if num_m2m_edges == 0:
        raise ValueError("Mesh has 0 m2m edges after filter_edges_land.")
    dm = float(
        np.mean(
            global_icosahedral_mesh.mesh_edge_lengths_cart(
                mesh_cart, mesh_edge_index
            )
        )
    )
    print(f"Mesh: {num_mesh} nodes, mean edge length (chord) = {dm:.6f}")

    # G2M: grid -> mesh (radius query in Cartesian for chord distance on sphere;
    #      icosahedral mesh is native Cartesian; cluster is lon-lat but we pass
    #      mesh_cart so radius is correct globally)
    g2m_ei, g2m_len, g2m_vdiff = global_icosahedral_mesh.g2m_radius_query(
        grid_xy,
        mesh_cart,
        mesh_edge_index,
        radius_factor=g2m_radius,
    )
    g2m_edge_index_saved = np.stack(
        [g2m_ei[0] + num_mesh, g2m_ei[1]], axis=0
    ).astype(np.int64)
    g2m_edge_index_t = torch.from_numpy(g2m_edge_index_saved)
    g2m_len_t = torch.from_numpy(g2m_len)
    g2m_vdiff_t = torch.from_numpy(g2m_vdiff)

    # Connect disconnected g2m nodes
    if connect_disconnected:
        pos_g2m = np.concatenate([mesh_xy, grid_xy], axis=0)
        pyg_g2m = pyg.data.Data(
            pos=torch.from_numpy(pos_g2m).float(),
            edge_index=g2m_edge_index_t.clone(),
        )
        gutils.add_edge_features_pyg(pyg_g2m)
        is_mesh = np.array([True] * num_mesh + [False] * num_grid)
        is_grid_interior = np.array([False] * num_mesh + [True] * num_grid)
        is_grid_boundary = np.zeros(num_mesh + num_grid, dtype=bool)
        is_grid_atm = np.zeros(num_mesh + num_grid, dtype=bool)
        kdt_m = scipy.spatial.KDTree(mesh_xy)
        gutils.connect_disconnected_g2m(
            pyg_g2m,
            is_mesh,
            is_grid_interior,
            is_grid_boundary,
            is_grid_atm,
            list(range(num_mesh)),
            mesh_xy,
            kdt_m,
            dm,
            g2m_radius,
            0.0,
            0.0,
        )
        g2m_edge_index_t = pyg_g2m.edge_index
        g2m_len_t = pyg_g2m.len
        g2m_vdiff_t = pyg_g2m.vdiff

    g2m_graph = SimpleNamespace(
        edge_index=g2m_edge_index_t,
        len=g2m_len_t,
        vdiff=g2m_vdiff_t,
    )
    saving.save_edges(g2m_graph, "g2m", graph_dir_path)

    # M2G: mesh -> grid (knn)
    m2g_ei, m2g_len, m2g_vdiff = global_icosahedral_mesh.m2g_knn(
        grid_xy, mesh_cart, k=m2g_k
    )
    m2g_edge_index_saved = np.stack(
        [m2g_ei[0], m2g_ei[1] + num_mesh], axis=0
    ).astype(np.int64)
    m2g_edge_index_t = torch.from_numpy(m2g_edge_index_saved)
    m2g_len_t = torch.from_numpy(m2g_len)
    m2g_vdiff_t = torch.from_numpy(m2g_vdiff)

    # Filter m2g edges over land when sea_xy and land_xy are provided
    if sea_xy is not None and land_xy is not None and land_xy.shape[0] > 0:
        pos_m2g = np.concatenate([mesh_xy, grid_xy], axis=0)
        pyg_m2g = pyg.data.Data(
            pos=torch.from_numpy(pos_m2g).float(),
            edge_index=m2g_edge_index_t.clone(),
        )
        gutils.add_edge_features_pyg(pyg_m2g)
        gutils.filter_edges_land(
            pyg_m2g,
            sea_xy,
            land_xy,
            max_edge_len=max_edge_len_deg,
        )
        m2g_edge_index_t = pyg_m2g.edge_index
        m2g_len_t = pyg_m2g.len
        m2g_vdiff_t = pyg_m2g.vdiff

    # Connect disconnected m2g grid nodes
    if connect_disconnected:
        pos_m2g = np.concatenate([mesh_xy, grid_xy], axis=0)
        pyg_m2g = pyg.data.Data(
            pos=torch.from_numpy(pos_m2g).float(),
            edge_index=m2g_edge_index_t.clone(),
        )
        gutils.add_edge_features_pyg(pyg_m2g)
        is_mesh = np.array([True] * num_mesh + [False] * num_grid)
        is_grid = np.array([False] * num_mesh + [True] * num_grid)
        kdt_m = scipy.spatial.KDTree(mesh_xy)
        xy_land = land_xy if land_xy is not None else np.empty((0, 2))
        gutils.connect_disconnected_m2g(
            pyg_m2g,
            is_mesh,
            is_grid,
            grid_xy,
            list(range(num_mesh)),
            mesh_xy,
            kdt_m,
            xy_land,
        )
        m2g_edge_index_t = pyg_m2g.edge_index
        m2g_len_t = pyg_m2g.len
        m2g_vdiff_t = pyg_m2g.vdiff
        m2g_edge_index_saved = pyg_m2g.edge_index.numpy()

    m2g_graph = SimpleNamespace(
        edge_index=m2g_edge_index_t,
        len=m2g_len_t,
        vdiff=m2g_vdiff_t,
    )
    saving.save_edges(m2g_graph, "m2g", graph_dir_path)

    # M2M and mesh features: cluster (single or hierarchical) or icosahedral
    if graph_type == "cluster" and save_graphs_cluster is not None:
        m2m_graphs = save_graphs_cluster["m2m"]
        saving.save_edges_list(m2m_graphs, "m2m", graph_dir_path)
        mesh_features_list = [g.pos for g in m2m_graphs]  # (lon, lat) already
        if hierarchical_cluster:
            mesh_up = save_graphs_cluster["mesh_up"]
            mesh_down = save_graphs_cluster["mesh_down"]
            saving.save_edges_list(mesh_up, "mesh_up", graph_dir_path)
            saving.save_edges_list(mesh_down, "mesh_down", graph_dir_path)
    else:
        mesh_edge_index_t = torch.from_numpy(mesh_edge_index.astype(np.int64))
        m2m_len = global_icosahedral_mesh.mesh_edge_lengths_cart(
            mesh_cart, mesh_edge_index
        )
        m2m_src, m2m_dst = mesh_edge_index[0], mesh_edge_index[1]
        m2m_lon_lat_src = global_icosahedral_mesh._cartesian_to_lat_lon(
            mesh_cart[m2m_src]
        )
        m2m_lon_lat_dst = global_icosahedral_mesh._cartesian_to_lat_lon(
            mesh_cart[m2m_dst]
        )
        # Edge feature (dlat, dlon) to match g2m/m2g
        dlon = (m2m_lon_lat_dst[:, 0] - m2m_lon_lat_src[:, 0]).astype(
            np.float32
        )
        dlat = (m2m_lon_lat_dst[:, 1] - m2m_lon_lat_src[:, 1]).astype(
            np.float32
        )
        m2m_vdiff = np.stack([dlat, dlon], axis=1)
        m2m_graph = SimpleNamespace(
            edge_index=mesh_edge_index_t,
            len=torch.from_numpy(m2m_len),
            vdiff=torch.from_numpy(m2m_vdiff),
        )
        saving.save_edges_list([m2m_graph], "m2m", graph_dir_path)
        mesh_features_list = [torch.from_numpy(mesh_xy)]  # (lon, lat) already
    torch.save(
        mesh_features_list,
        os.path.join(graph_dir_path, "mesh_features.pt"),
    )

    if create_plot:
        # grid_xy and mesh_xy are (lon, lat) throughout; plot x=lon, y=lat
        mesh_plot_xy = mesh_xy
        grid_plot_xy = grid_xy

        # Overview: x=lon, y=lat
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(
            mesh_plot_xy[:, 0],
            mesh_plot_xy[:, 1],
            s=1,
            c="orange",
            label="Mesh",
        )
        ax.scatter(
            grid_plot_xy[:, 0],
            grid_plot_xy[:, 1],
            s=0.5,
            c="blue",
            alpha=0.3,
            label="Grid",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="upper right")
        ax.set_title("Global graph: mesh and grid")
        fig.savefig(
            os.path.join(graph_dir_path, "global_graph_overview.png"),
            dpi=150,
        )
        plt.close(fig)

        # Combined pos (lon, lat) for g2m/m2g plots
        pos_combined = np.concatenate([mesh_plot_xy, grid_plot_xy], axis=0)
        is_mesh = np.array([True] * num_mesh + [False] * num_grid)
        is_any_grid = np.array([False] * num_mesh + [True] * num_grid)

        # Mesh level(s)
        if graph_type == "cluster" and save_graphs_cluster is not None:
            m2m_graphs = save_graphs_cluster["m2m"]
            for level_i, g in enumerate(m2m_graphs):
                # Cluster mesh pos is (lon, lat); use (x=lon, y=lat) for plot
                level_graph = pyg.data.Data(
                    pos=g.pos.float(),
                    edge_index=g.edge_index,
                )
                vis.plot_graph(
                    level_graph,
                    f"Mesh graph, level {level_i}",
                    graph_dir_path,
                )
        else:
            mesh_level_graph = pyg.data.Data(
                pos=torch.from_numpy(mesh_plot_xy).float(),
                edge_index=mesh_edge_index_t,
            )
            vis.plot_graph(
                mesh_level_graph,
                "Mesh graph, level 0",
                graph_dir_path,
            )

        # G2M: build pyg with pos (lon, lat), compute disconnected
        pyg_g2m = pyg.data.Data(
            pos=torch.from_numpy(pos_combined).float(),
            edge_index=g2m_graph.edge_index.clone(),
        )
        g2m_src = pyg_g2m.edge_index[0]
        g2m_dst = pyg_g2m.edge_index[1]
        g2m_outdeg = degree(g2m_src, num_nodes=pyg_g2m.num_nodes)
        g2m_indeg = degree(g2m_dst, num_nodes=pyg_g2m.num_nodes)
        grid_mask_t = torch.as_tensor(is_any_grid, device=pyg_g2m.pos.device)
        mesh_mask_t = torch.as_tensor(is_mesh, device=pyg_g2m.pos.device)
        disc_grid_np = (
            torch.where((g2m_outdeg == 0) & grid_mask_t)[0].cpu().numpy()
        )
        disc_mesh_np = (
            torch.where((g2m_indeg == 0) & mesh_mask_t)[0].cpu().numpy()
        )
        vis.plot_disconnected_nodes(
            pos_combined,
            is_mesh,
            is_any_grid,
            disc_grid_np,
            disc_mesh_np,
            graph_dir_path,
            title="g2m_disconnected",
        )
        vis.plot_graph(
            pyg_g2m, "Grid-to-mesh", graph_dir_path, order_by_degree=True
        )
        pyg_g2m_r = pyg_g2m.clone()
        pyg_g2m_r.edge_index = pyg_g2m.edge_index[[1, 0]]
        vis.plot_graph(pyg_g2m_r, "Grid-to-mesh-r", graph_dir_path)

        # M2G
        pyg_m2g = pyg.data.Data(
            pos=torch.from_numpy(pos_combined).float(),
            edge_index=m2g_graph.edge_index.clone(),
        )
        m2g_dst = pyg_m2g.edge_index[1]
        m2g_indeg = degree(m2g_dst, num_nodes=pyg_m2g.num_nodes)
        m2g_disc_grid_np = (
            torch.where((m2g_indeg == 0) & grid_mask_t)[0].cpu().numpy()
        )
        vis.plot_disconnected_nodes(
            pos_combined,
            is_mesh,
            is_any_grid,
            m2g_disc_grid_np,
            np.array([], dtype=int),
            graph_dir_path,
            title="m2g_disconnected",
        )
        vis.plot_graph(
            pyg_m2g, "Mesh-to-grid", graph_dir_path, reindex_edges=False
        )
        pyg_m2g_r = pyg_m2g.clone()
        pyg_m2g_r.edge_index = pyg_m2g.edge_index[[1, 0]]
        vis.plot_graph(
            pyg_m2g_r,
            "Mesh-to-grid-r",
            graph_dir_path,
            reindex_edges=False,
            order_by_degree=True,
        )

    print(f"Created global graph: {num_grid} grid nodes, {num_mesh} mesh nodes")
    print(f"  g2m edges: {g2m_graph.edge_index.shape[1]}")
    print(f"  m2g edges: {m2g_graph.edge_index.shape[1]}")
    print(f"  m2m edges: {mesh_edge_index.shape[1]}")


def cli(input_args=None):
    parser = ArgumentParser(description="Create global graph.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to neural-lam config; grid is loaded from datastore. "
        "Mutually exclusive with --dataset.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="global_multiscale",
        help="Graph name (saved under datastore root or cwd).",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=3,
        help="Icosahedral mesh subdivisions (0=base, 3 ~ 2562 nodes).",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="multiscale",
        choices=["multiscale", "hierarchical", "cluster"],
        help="Graph type: multiscale (single icosahedral), hierarchical "
        "(multi-level icosahedral), or cluster (KMeans over sea, requires "
        "--config_path).",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=None,
        help="Number of mesh levels (default: from --type).",
    )
    parser.add_argument(
        "--g2m_radius",
        type=float,
        default=0.67,
        help="G2M connection radius as multiple of mean mesh edge length.",
    )
    parser.add_argument(
        "--m2g_k",
        type=int,
        default=3,
        help="Number of nearest mesh nodes per grid point for M2G.",
    )
    parser.add_argument(
        "--connect_disconnected",
        action="store_true",
        help="Connect disconnected grid/mesh nodes via nearest neighbor.",
    )
    parser.add_argument(
        "--max_edge_len_deg",
        type=float,
        default=2.0,
        help="Max m2g edge length in degrees for land filtering.",
    )
    parser.add_argument(
        "--mesh_refinement_factor",
        type=float,
        default=9,
        help="(Cluster) factor between mesh nodes at consecutive levels.",
    )
    parser.add_argument(
        "--grid_to_first_mesh_refinement",
        type=float,
        default=25,
        help="(Cluster) ratio grid nodes / first-level mesh nodes.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save overview plot of mesh and grid.",
    )
    args = parser.parse_args(input_args)

    _, datastore, _, _ = load_config_and_datastores(
        config_path=args.config_path
    )
    grid_xy, sea_xy, land_xy = load_grid_from_datastore(datastore)
    graph_dir_path = os.path.join(datastore.root_path, "graphs", args.name)

    create_global_graph(
        graph_dir_path=graph_dir_path,
        grid_xy=grid_xy,
        splits=args.splits,
        levels=args.levels,
        graph_type=args.type,
        g2m_radius=args.g2m_radius,
        m2g_k=args.m2g_k,
        create_plot=args.plot,
        connect_disconnected=args.connect_disconnected,
        sea_xy=sea_xy,
        land_xy=land_xy,
        max_edge_len_deg=args.max_edge_len_deg,
        mesh_refinement_factor=args.mesh_refinement_factor,
        grid_to_first_mesh_refinement=args.grid_to_first_mesh_refinement,
    )


if __name__ == "__main__":
    cli()
