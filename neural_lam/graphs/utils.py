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


def _check_m2g_disconnected(
    xy,
    vm_xy,
    kdt_m,
    m2g_k,
    xy_land,
):
    """
    Check for disconnected nodes in m2g without building full graph.
    Returns num_disconnected_grid.
    """
    num_grid = len(xy)
    grid_has_incoming = np.zeros(num_grid, dtype=bool)

    # Build KDTree for land/sea check (only build once)
    all_xy = np.concatenate((xy, xy_land), axis=0)
    grid_kdt = scipy.spatial.KDTree(all_xy)
    sea_xy_len = len(xy)

    # Build m2g edges with k-NN
    for i, grid_pos in enumerate(xy):
        # Find k nearest mesh nodes
        k_actual = min(m2g_k, len(vm_xy))
        if k_actual == 1:
            dist, mesh_idx = kdt_m.query(grid_pos, k=1)
            neigh_idxs = (
                [mesh_idx]
                if not isinstance(mesh_idx, np.ndarray)
                else mesh_idx.flatten()
            )
        else:
            _, neigh_idxs = kdt_m.query(grid_pos, k=k_actual)
            if neigh_idxs.ndim > 1:
                neigh_idxs = neigh_idxs.flatten()

        # Check if any edge would survive land filtering
        for mesh_idx in neigh_idxs:
            mesh_pos = vm_xy[mesh_idx]
            midpoint = (grid_pos + mesh_pos) / 2
            edge_len = np.linalg.norm(grid_pos - mesh_pos)

            # Check if edge would be filtered
            closest_idx = grid_kdt.query(midpoint)[1]
            is_over_sea = closest_idx < sea_xy_len
            is_within_max_len = edge_len < 50000  # BASE_MAX_EDGE_LEN

            if is_over_sea and is_within_max_len:
                grid_has_incoming[i] = True
                break

    num_disc_grid = np.sum(~grid_has_incoming)
    return num_disc_grid


def _search_single_g2m_radius(
    radius_idx,
    radius_name,
    xy,
    xy_boundary,
    xy_atmosphere,
    vm_xy,
    kdt_m,
    dm,
    precision,
    fraction,
    base_radii=None,
    low=0.01,
    high=5.0,
):
    """
    Binary search for a single g2m radius.

    Parameters
    ----------
    base_radii : list or None
        Base radii to use for other radius types. If None, uses [None, None, None].
    """
    # Determine which grid type we're searching for
    check_interior = radius_idx == 0
    check_boundary = radius_idx == 1
    check_atm = radius_idx == 2

    # Calculate total nodes for this radius type
    if check_interior:
        total_nodes = len(xy)
    elif check_boundary:
        total_nodes = len(xy_boundary)
    else:  # check_atm
        total_nodes = len(xy_atmosphere)

    max_disconnected = int((1 - fraction) * total_nodes)

    best_radius = None
    if base_radii is None:
        current_radii = [None, None, None]
    else:
        current_radii = base_radii.copy()

    # First, find a high value that works
    current_radii[radius_idx] = high
    num_disc_grid, num_disc_mesh = _check_g2m_disconnected(
        xy,
        xy_boundary,
        xy_atmosphere,
        vm_xy,
        kdt_m,
        dm,
        current_radii[0],
        current_radii[1],
        current_radii[2],
        check_interior=check_interior,
        check_boundary=check_boundary,
        check_atm=check_atm,
    )
    while num_disc_grid > max_disconnected and high < 20.0:
        high *= 2
        current_radii[radius_idx] = high
        num_disc_grid, num_disc_mesh = _check_g2m_disconnected(
            xy,
            xy_boundary,
            xy_atmosphere,
            vm_xy,
            kdt_m,
            dm,
            current_radii[0],
            current_radii[1],
            current_radii[2],
            check_interior=check_interior,
            check_boundary=check_boundary,
            check_atm=check_atm,
        )

    # Binary search
    while high - low > precision:
        mid = (low + high) / 2
        current_radii[radius_idx] = mid
        num_disc_grid, num_disc_mesh = _check_g2m_disconnected(
            xy,
            xy_boundary,
            xy_atmosphere,
            vm_xy,
            kdt_m,
            dm,
            current_radii[0],
            current_radii[1],
            current_radii[2],
            check_interior=check_interior,
            check_boundary=check_boundary,
            check_atm=check_atm,
        )
        total_disc = num_disc_grid + num_disc_mesh
        connected_frac = (
            1.0 - (num_disc_grid / total_nodes) if total_nodes > 0 else 1.0
        )
        print(
            f"  {radius_name}: {mid:.2f} -> {num_disc_grid} disconnected ({connected_frac:.1%} connected)"
        )

        if num_disc_grid <= max_disconnected:
            best_radius = mid
            high = mid  # Try smaller
        else:
            low = mid  # Need larger

    # After binary search, find the minimum value with fraction connected at 2-decimal precision
    if best_radius is None:
        best_radius = high
    else:
        # Test values at 2-decimal precision around best_radius to find minimum
        # Start from best_radius rounded down, test up to best_radius + 0.02
        test_start = round(best_radius - 0.01, 2)
        test_end = round(best_radius + 0.02, 2)
        best_radius_rounded = None

        for test_val in np.arange(test_start, test_end + 0.01, 0.01):
            test_val = round(test_val, 2)
            current_radii[radius_idx] = test_val
            num_disc_grid, num_disc_mesh = _check_g2m_disconnected(
                xy,
                xy_boundary,
                xy_atmosphere,
                vm_xy,
                kdt_m,
                dm,
                current_radii[0],
                current_radii[1],
                current_radii[2],
                check_interior=check_interior,
                check_boundary=check_boundary,
                check_atm=check_atm,
            )
            if num_disc_grid <= max_disconnected:
                best_radius_rounded = test_val
                break  # Found minimum value with fraction connected

        if best_radius_rounded is None:
            # Fallback: use best_radius rounded up
            best_radius_rounded = round(best_radius + 0.01, 2)

        best_radius = best_radius_rounded

    return best_radius


def search_g2m_radii(
    xy,
    xy_boundary,
    xy_atmosphere,
    vm_xy,
    kdt_m,
    dm,
    precision=0.01,
    fraction=0.95,
):
    """
    Search for g2m radii that result in fraction of nodes connected.
    Uses binary search for each radius independently.
    Returns (g2m_radius, g2m_radius_boundary, g2m_radius_atm).

    Parameters
    ----------
    fraction : float
        Fraction of nodes that must be connected (default: 0.95).
    """
    print("Searching for optimal g2m radii...")

    # Search g2m_radius
    g2m_radius = _search_single_g2m_radius(
        0,
        "g2m_radius",
        xy,
        xy_boundary,
        xy_atmosphere,
        vm_xy,
        kdt_m,
        dm,
        precision,
        fraction,
    )

    # Search g2m_radius_boundary
    base_radii = [g2m_radius, None, None]
    g2m_radius_boundary = _search_single_g2m_radius(
        1,
        "g2m_radius_boundary",
        xy,
        xy_boundary,
        xy_atmosphere,
        vm_xy,
        kdt_m,
        dm,
        precision,
        fraction,
        base_radii,
    )

    # g2m_radius_atm
    base_radii = [g2m_radius, g2m_radius_boundary, None]
    g2m_radius_atm = _search_single_g2m_radius(
        2,
        "g2m_radius_atm",
        xy,
        xy_boundary,
        xy_atmosphere,
        vm_xy,
        kdt_m,
        dm,
        precision,
        fraction,
        base_radii,
    )

    print(
        f"Found optimal radii: g2m={g2m_radius:.2f}, boundary={g2m_radius_boundary:.2f}, atm={g2m_radius_atm:.2f}"
    )
    return g2m_radius, g2m_radius_boundary, g2m_radius_atm


def search_m2g_k(
    xy,
    vm_xy,
    kdt_m,
    xy_land,
    start_k=1,
    fraction=0.95,
    limit=10,
):
    """
    Search for smallest integer k that results in fraction of nodes connected in m2g.
    Returns k.

    Parameters
    ----------
    fraction : float
        Fraction of nodes that must be connected (default: 0.95).
    """
    print("Searching for optimal m2g k...")
    total_nodes = len(xy)
    max_disconnected = int((1 - fraction) * total_nodes)
    k = start_k

    while True:
        num_disc = _check_m2g_disconnected(xy, vm_xy, kdt_m, k, xy_land)
        connected_frac = (
            1.0 - (num_disc / total_nodes) if total_nodes > 0 else 1.0
        )
        print(
            f"  k={k} -> {num_disc} disconnected grid nodes ({connected_frac:.1%} connected)"
        )

        if num_disc <= max_disconnected:
            print(f"Found optimal k: {k}")
            return k

        k += 1
        if k > limit:
            print(f"Warning: k exceeded limit {limit}, using k={limit}")
            return limit


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
    # Third-party
    from torch_geometric.utils import degree

    pos = pyg_g2m.pos.cpu().numpy()
    src = pyg_g2m.edge_index[0]
    dst = pyg_g2m.edge_index[1]
    num_nodes = pyg_g2m.num_nodes

    outdeg = degree(src, num_nodes=num_nodes)
    indeg = degree(dst, num_nodes=num_nodes)

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
        # Determine which radius to use
        if is_grid_interior[grid_idx]:
            radius = g2m_radius
        elif is_grid_boundary[grid_idx]:
            radius = g2m_radius_boundary
        else:  # atmosphere
            radius = g2m_radius_atm

        # Find nearest mesh node (index in vm_xy/vm_list)
        # Mesh nodes come first in graph, so index matches vm_list index
        dist, mesh_idx_vm = kdt_m.query(grid_pos, k=1)
        if (
            dist < dm * radius * 10
        ):  # Allow up to 10x radius for disconnected nodes
            # Mesh nodes are at indices 0 to len(vm_list)-1 in the graph
            mesh_graph_idx = mesh_idx_vm
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
    # Third-party
    from torch_geometric.utils import degree

    pos = pyg_m2g.pos.cpu().numpy()
    dst = pyg_m2g.edge_index[1]
    num_nodes = pyg_m2g.num_nodes

    indeg = degree(dst, num_nodes=num_nodes)
    grid_mask_t = torch.as_tensor(is_grid, device=indeg.device)

    disc_grid = torch.where((indeg == 0) & grid_mask_t)[0].cpu().numpy()

    new_edges = []

    # Connect disconnected grid nodes to nearest mesh node
    # Note: We connect regardless of land filtering to ensure connectivity
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
