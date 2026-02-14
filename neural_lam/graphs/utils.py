# Third-party
import networkx
import numpy as np
import scipy
import torch
import torch_geometric as pyg
from torch_geometric.utils.convert import from_networkx


def node_lat_lon_to_cart(node_xy):
    """Convert node positions from longitude-latitude to Cartesian on unit sphere.

    Convention: x = longitude (column 0), y = latitude (column 1), matching
    datastore get_xy and plotting (x-axis = lon, y-axis = lat).

    Parameters
    ----------
    node_xy : np.ndarray
        (N_nodes, 2) array, columns [longitude, latitude] in degrees.

    Returns
    -------
    np.ndarray
        (N_nodes, 3) array, Cartesian coordinates on unit sphere.
    """
    lon_rad = np.deg2rad(node_xy[:, 0])
    lat_rad = np.deg2rad(node_xy[:, 1])
    # theta = 90 - lat (so z = cos(theta) = sin(lat)), phi = lon
    theta_grid = np.deg2rad(90.0) - lat_rad
    phi_grid = lon_rad

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
    max_edge_len: float = 20000,  # in m
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


def _check_g2m_disconnected(
    xy,
    xy_boundary,
    xy_atmosphere,
    vm_xy,
    kdt_m,
    dm,
    g2m_radius,
    g2m_radius_boundary,
    g2m_radius_atm,
    check_interior=True,
    check_boundary=True,
    check_atm=True,
):
    """
    Check for disconnected nodes in g2m without building full graph.
    Returns (num_disconnected_grid, num_disconnected_mesh).
    """
    num_grid_interior = len(xy) if check_interior else 0
    num_grid_boundary = len(xy_boundary) if check_boundary else 0
    num_grid_atm = len(xy_atmosphere) if check_atm else 0
    num_mesh = len(vm_xy)

    # Track which grid nodes have outgoing edges
    grid_has_outgoing = np.zeros(
        num_grid_interior + num_grid_boundary + num_grid_atm, dtype=bool
    )
    # Track which mesh nodes have incoming edges
    mesh_has_incoming = np.zeros(num_mesh, dtype=bool)

    # Check interior grid nodes
    if check_interior:
        for i, grid_pos in enumerate(xy):
            neigh_idxs = kdt_m.query_ball_point(grid_pos, dm * g2m_radius)
            if len(neigh_idxs) > 0:
                grid_has_outgoing[i] = True
                mesh_has_incoming[neigh_idxs] = True

    # Check boundary grid nodes
    if check_boundary:
        for i, grid_pos in enumerate(xy_boundary):
            idx = num_grid_interior + i
            neigh_idxs = kdt_m.query_ball_point(
                grid_pos, dm * g2m_radius_boundary
            )
            if len(neigh_idxs) > 0:
                grid_has_outgoing[idx] = True
                mesh_has_incoming[neigh_idxs] = True

    # Check atmospheric grid nodes
    if check_atm:
        for i, grid_pos in enumerate(xy_atmosphere):
            idx = num_grid_interior + num_grid_boundary + i
            neigh_idxs = kdt_m.query_ball_point(grid_pos, dm * g2m_radius_atm)
            if len(neigh_idxs) > 0:
                grid_has_outgoing[idx] = True
                mesh_has_incoming[neigh_idxs] = True

    # Only count disconnected grid nodes of the types we're checking
    if check_interior and check_boundary and check_atm:
        # Checking all types - count all disconnected grid nodes
        num_disc_grid = np.sum(~grid_has_outgoing)
    elif check_interior:
        # Only checking interior - only count interior disconnected nodes
        num_disc_grid = np.sum(~grid_has_outgoing[:num_grid_interior])
    elif check_boundary:
        # Only checking boundary - only count boundary disconnected nodes
        start_idx = num_grid_interior
        end_idx = num_grid_interior + num_grid_boundary
        num_disc_grid = np.sum(~grid_has_outgoing[start_idx:end_idx])
    else:  # check_atm
        # Only checking atmosphere - only count atmosphere disconnected nodes
        start_idx = num_grid_interior + num_grid_boundary
        num_disc_grid = np.sum(~grid_has_outgoing[start_idx:])

    # Don't count mesh nodes when searching for a specific radius
    # (they'll be connected by the combination of all radii)
    num_disc_mesh = 0

    return num_disc_grid, num_disc_mesh


def _g2m_mean_out_degree(xy_subset, kdt_m, dm, radius):
    """
    mean number of mesh nodes within dm*radius of each
    grid point in xy_subset.
    """
    if len(xy_subset) == 0:
        return 0.0
    counts = np.array(
        [
            len(kdt_m.query_ball_point(grid_pos, dm * radius))
            for grid_pos in xy_subset
        ],
        dtype=np.float64,
    )
    return float(np.mean(counts))


def _search_single_g2m_radius_by_mean_degree(
    radius_idx,
    radius_name,
    xy,
    xy_boundary,
    xy_atmosphere,
    vm_xy,
    kdt_m,
    dm,
    mean_degree,
    precision=0.01,
    base_radii=None,
    low=0.01,
    high=5.0,
):
    """
    Binary search for smallest radius such that mean G2M out-degree
    (for this grid type) >= mean_degree.
    """
    if radius_idx == 0:
        xy_subset = xy
    elif radius_idx == 1:
        xy_subset = xy_boundary
    else:
        xy_subset = xy_atmosphere

    if len(xy_subset) == 0:
        return low

    current_radii = (
        [None, None, None] if base_radii is None else base_radii.copy()
    )
    best_radius = None

    # Ensure high gives at least mean_degree
    current_radii[radius_idx] = high
    med_deg = _g2m_mean_out_degree(
        xy_subset, kdt_m, dm, current_radii[radius_idx]
    )
    while med_deg < mean_degree and high < 20.0:
        high *= 2
        current_radii[radius_idx] = high
        med_deg = _g2m_mean_out_degree(
            xy_subset, kdt_m, dm, current_radii[radius_idx]
        )

    if med_deg < mean_degree:
        return high

    # Binary search for smallest radius with mean_degree >= target
    while high - low > precision:
        mid = (low + high) / 2
        current_radii[radius_idx] = mid
        med_deg = _g2m_mean_out_degree(xy_subset, kdt_m, dm, mid)
        print(
            f"{radius_name}: radius={mid:.2f} -> mean outdegree={med_deg:.2f}"
        )
        if med_deg >= mean_degree:
            best_radius = mid
            high = mid
        else:
            low = mid

    if best_radius is None:
        best_radius = high
    else:
        # Refine to 2-decimal precision
        test_start = round(best_radius - 0.01, 2)
        test_end = round(best_radius + 0.02, 2)
        for test_val in np.arange(test_start, test_end + 0.01, 0.01):
            test_val = round(test_val, 2)
            med_deg = _g2m_mean_out_degree(xy_subset, kdt_m, dm, test_val)
            if med_deg >= mean_degree:
                best_radius = test_val
                break
        else:
            best_radius = round(best_radius + 0.01, 2)

    return best_radius


def search_g2m_radii_by_mean_degree(
    xy,
    xy_boundary,
    xy_atmosphere,
    vm_xy,
    kdt_m,
    dm,
    mean_degree,
    precision=0.01,
    check_atm=True,
):
    """
    Search for G2M radii (interior, boundary, atmosphere) that achieve
    the given mean out-degree per grid node for each type.
    Returns (g2m_radius, g2m_radius_boundary, g2m_radius_atm).
    When check_atm is False, g2m_radius_atm is returned as g2m_radius_boundary.
    """
    print(f"Searching for G2M radii with mean connectivity {mean_degree}...")

    g2m_radius = _search_single_g2m_radius_by_mean_degree(
        0,
        "g2m_radius",
        xy,
        xy_boundary,
        xy_atmosphere,
        vm_xy,
        kdt_m,
        dm,
        mean_degree,
        precision,
    )

    base_radii = [g2m_radius, None, None]
    g2m_radius_boundary = _search_single_g2m_radius_by_mean_degree(
        1,
        "g2m_radius_boundary",
        xy,
        xy_boundary,
        xy_atmosphere,
        vm_xy,
        kdt_m,
        dm,
        mean_degree,
        precision,
        base_radii,
    )

    if check_atm and len(xy_atmosphere) > 0:
        base_radii = [g2m_radius, g2m_radius_boundary, None]
        g2m_radius_atm = _search_single_g2m_radius_by_mean_degree(
            2,
            "g2m_radius_atm",
            xy,
            xy_boundary,
            xy_atmosphere,
            vm_xy,
            kdt_m,
            dm,
            mean_degree,
            precision,
            base_radii,
        )
        print(
            f"Found radii: interior={g2m_radius:.2f}, "
            f"boundary={g2m_radius_boundary:.2f}, "
            f"atmosphere={g2m_radius_atm:.2f}"
        )
    else:
        g2m_radius_atm = g2m_radius_boundary
        print(
            f"Found radii: interior={g2m_radius:.2f}, "
            f"boundary={g2m_radius_boundary:.2f}"
        )

    return g2m_radius, g2m_radius_boundary, g2m_radius_atm


def connect_disconnected_g2m(
    pyg_g2m,
    is_mesh,
    is_grid_interior,
    is_grid_boundary,
    is_grid_atm,
    vm_list,
    vm_xy,
    kdt_m,
    dm,
    g2m_radius,
    g2m_radius_boundary,
    g2m_radius_atm,
):
    """
    Connect disconnected nodes in g2m graph using nearest neighbor.
    Modifies graph in-place.
    """
    pos = pyg_g2m.pos.cpu().numpy()
    src = pyg_g2m.edge_index[0]
    dst = pyg_g2m.edge_index[1]
    num_nodes = pyg_g2m.num_nodes

    outdeg = pyg.utils.degree(src, num_nodes=num_nodes)
    indeg = pyg.utils.degree(dst, num_nodes=num_nodes)

    # Find disconnected nodes
    is_any_grid = is_grid_interior | is_grid_boundary | is_grid_atm
    grid_mask_t = torch.as_tensor(is_any_grid, device=outdeg.device)
    mesh_mask_t = torch.as_tensor(is_mesh, device=outdeg.device)

    disc_grid = torch.where((outdeg == 0) & grid_mask_t)[0].cpu().numpy()
    disc_mesh = torch.where((indeg == 0) & mesh_mask_t)[0].cpu().numpy()

    new_edges = []

    # Connect disconnected grid nodes to nearest mesh node
    for grid_idx in disc_grid:
        grid_pos = pos[grid_idx]
        # Find nearest mesh node (index in vm_xy/vm_list)
        # Mesh nodes come first in graph, so index matches vm_list index
        _, mesh_idx_vm = kdt_m.query(grid_pos, k=1)
        mesh_graph_idx = int(np.asarray(mesh_idx_vm).flat[0])
        new_edges.append([grid_idx, mesh_graph_idx])

    # Connect disconnected mesh nodes to nearest grid node
    for mesh_graph_idx in disc_mesh:
        mesh_pos = pos[mesh_graph_idx]
        # Find nearest grid node (any type)
        grid_kdt = scipy.spatial.KDTree(pos[is_any_grid])
        dist, grid_idx_in_subset = grid_kdt.query(mesh_pos, k=1)
        grid_idx = np.where(is_any_grid)[0][grid_idx_in_subset]
        new_edges.append([grid_idx, mesh_graph_idx])

    if len(new_edges) > 0:
        new_edge_index = torch.tensor(
            new_edges, dtype=torch.long, device=pyg_g2m.edge_index.device
        ).T
        pyg_g2m.edge_index = torch.cat(
            [pyg_g2m.edge_index, new_edge_index], dim=1
        )
        # Update edge features
        add_edge_features_pyg(pyg_g2m)
        print(f"Connected {len(new_edges)} disconnected nodes in g2m")


def connect_disconnected_m2g(
    pyg_m2g,
    is_mesh,
    is_grid,
    xy,
    vm_list,
    vm_xy,
    kdt_m,
    xy_land,
):
    """
    Connect disconnected grid nodes in m2g graph using nearest neighbor.
    Modifies graph in-place.
    """
    pos = pyg_m2g.pos.cpu().numpy()
    dst = pyg_m2g.edge_index[1]
    num_nodes = pyg_m2g.num_nodes

    indeg = pyg.utils.degree(dst, num_nodes=num_nodes)
    grid_mask_t = torch.as_tensor(is_grid, device=indeg.device)

    disc_grid = torch.where((indeg == 0) & grid_mask_t)[0].cpu().numpy()

    new_edges = []

    # Connect disconnected grid nodes to nearest mesh node
    for grid_idx in disc_grid:
        grid_pos = pos[grid_idx]
        # Find nearest mesh node (index in vm_xy/vm_list)
        # Mesh nodes come first in graph, so index matches vm_list index
        dist, mesh_idx_vm = kdt_m.query(grid_pos, k=1)
        # Mesh nodes are at indices 0 to len(vm_list)-1 in the graph
        mesh_graph_idx = mesh_idx_vm
        new_edges.append([mesh_graph_idx, grid_idx])

    if len(new_edges) > 0:
        new_edge_index = torch.tensor(
            new_edges, dtype=torch.long, device=pyg_m2g.edge_index.device
        ).T
        pyg_m2g.edge_index = torch.cat(
            [pyg_m2g.edge_index, new_edge_index], dim=1
        )
        # Update edge features
        add_edge_features_pyg(pyg_m2g)
        print(f"Connected {len(new_edges)} disconnected nodes in m2g")


def print_graph_stats(save_graphs, pyg_g2m, pyg_m2g):
    """Print node and edge counts for all graph components."""
    print("\n" + "=" * 50)
    print("Graph statistics")
    print("=" * 50)
    m2m_graphs = save_graphs["m2m"]
    for lev, g in enumerate(m2m_graphs):
        n_nodes = g.num_nodes
        n_edges = g.edge_index.shape[1]
        print(f"  m2m (level {lev}): {n_nodes} nodes, {n_edges} edges")
    total_mesh_nodes = sum(g.num_nodes for g in m2m_graphs)
    total_m2m_edges = sum(g.edge_index.shape[1] for g in m2m_graphs)
    print(f"  m2m (total): {total_mesh_nodes} nodes, {total_m2m_edges} edges")
    if "mesh_up" in save_graphs:
        for lev, g in enumerate(save_graphs["mesh_up"]):
            n_edges = g.edge_index.shape[1]
            print(f"  mesh_up (level {lev}->{lev+1}): {n_edges} edges")
        for lev, g in enumerate(save_graphs["mesh_down"]):
            n_edges = g.edge_index.shape[1]
            print(f"  mesh_down (level {lev+1}->{lev}): {n_edges} edges")
    n_g2m_nodes = pyg_g2m.num_nodes
    n_g2m_edges = pyg_g2m.edge_index.shape[1]
    print(f"  g2m: {n_g2m_nodes} nodes, {n_g2m_edges} edges")
    n_m2g_nodes = pyg_m2g.num_nodes
    n_m2g_edges = pyg_m2g.edge_index.shape[1]
    print(f"  m2g: {n_m2g_nodes} nodes, {n_m2g_edges} edges")
    print("=" * 50)
