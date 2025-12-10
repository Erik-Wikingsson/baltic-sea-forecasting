# Third-party
import networkx
import numpy as np
import scipy
import torch
from torch_geometric.utils.convert import from_networkx

# Local
from . import utils as gutils


def mk_2d_graph(xy, nx, ny, land_mask):
    xm, xM = np.amin(xy[0][0, :]), np.amax(xy[0][0, :])
    ym, yM = np.amin(xy[1][:, 0]), np.amax(xy[1][:, 0])

    # avoid nodes on border
    dx = (xM - xm) / nx
    dy = (yM - ym) / ny
    lx = np.linspace(xm + dx / 2, xM - dx / 2, nx, dtype=np.float32)
    ly = np.linspace(ym + dy / 2, yM - dy / 2, ny, dtype=np.float32)

    mg = np.meshgrid(lx, ly)
    g = networkx.grid_2d_graph(len(ly), len(lx))

    # kdtree for nearest neighbor search of land nodes
    land_points = np.argwhere(land_mask.T).astype(np.float32)
    land_kdtree = scipy.spatial.KDTree(land_points)

    # add nodes excluding land
    for node in list(g.nodes):
        node_pos = np.array([mg[0][node], mg[1][node]], dtype=np.float32)
        dist, _ = land_kdtree.query(node_pos, k=1)
        if dist < np.sqrt(0.5):
            g.remove_node(node)
        else:
            g.nodes[node]["pos"] = node_pos

    # add diagonal edges if both nodes exist
    for x in range(nx - 1):
        for y in range(ny - 1):
            if g.has_node((x, y)) and g.has_node((x + 1, y + 1)):
                g.add_edge((x, y), (x + 1, y + 1))
            if g.has_node((x + 1, y)) and g.has_node((x, y + 1)):
                g.add_edge((x + 1, y), (x, y + 1))

    # turn into directed graph
    dg = networkx.DiGraph(g)

    # add node data
    for u, v in g.edges():
        d = np.sqrt(np.sum((g.nodes[u]["pos"] - g.nodes[v]["pos"]) ** 2))
        dg.edges[u, v]["len"] = d
        dg.edges[u, v]["vdiff"] = g.nodes[u]["pos"] - g.nodes[v]["pos"]
        dg.add_edge(v, u)
        dg.edges[v, u]["len"] = d
        dg.edges[v, u]["vdiff"] = g.nodes[v]["pos"] - g.nodes[u]["pos"]

    # add self edge if needed
    for v, degree in list(dg.degree()):
        if degree <= 1:
            dg.add_edge(v, v, len=0, vdiff=np.array([0, 0]))

    return dg


def build_regular_mesh_graph(
    xy,
    land_mask,
    max_mesh_levels,
    hierarchical,
    mesh_plot_function,
):

    save_graphs = {}

    # graph geometry
    nx = 3  # number of children = nx**2
    nlev = int(np.log(max(xy.shape)) / np.log(nx))
    nleaf = nx**nlev  # leaves at the bottom = nleaf**2

    mesh_levels = nlev - 1
    if max_mesh_levels:
        # Limit the levels in mesh graph
        mesh_levels = min(mesh_levels, max_mesh_levels)

    print(f"nlev: {nlev}, nleaf: {nleaf}, mesh_levels: {mesh_levels}")

    # multi resolution tree levels
    G = []
    for lev in range(1, mesh_levels + 1):
        n = int(nleaf / (nx**lev))
        g = mk_2d_graph(xy, n, n, land_mask)
        if mesh_plot_function is not None:
            mesh_plot_function(from_networkx(g), f"Mesh graph, level {lev}")

        G.append(g)

    if hierarchical:
        # Relabel nodes of each level with level index first
        G = [
            gutils.prepend_node_index(graph, level_i)
            for level_i, graph in enumerate(G)
        ]

        num_nodes_level = np.array([len(g_level.nodes) for g_level in G])
        # First node index in each level in the hierarchical graph
        first_index_level = np.concatenate(
            (np.zeros(1, dtype=int), np.cumsum(num_nodes_level[:-1]))
        )

        # Create inter-level mesh edges
        up_graphs = []
        down_graphs = []
        for from_level, to_level, G_from, G_to, start_index in zip(
            range(1, mesh_levels),
            range(0, mesh_levels - 1),
            G[1:],
            G[:-1],
            first_index_level[: mesh_levels - 1],
        ):
            # start out from graph at from level
            G_down = G_from.copy()
            G_down.clear_edges()
            G_down = networkx.DiGraph(G_down)

            # Add nodes of to level
            G_down.add_nodes_from(G_to.nodes(data=True))

            # build kd tree for mesh point pos
            # order in vm should be same as in vm_xy
            v_to_list = list(G_to.nodes)
            v_from_list = list(G_from.nodes)
            v_from_xy = np.array([xy for _, xy in G_from.nodes.data("pos")])
            kdt_m = scipy.spatial.KDTree(v_from_xy)

            # add edges between levels
            for v in v_to_list:
                # find 1(?) nearest neighbours (index to vm_xy)
                neigh_idx = kdt_m.query(G_down.nodes[v]["pos"], 1)[1]
                u = v_from_list[neigh_idx]

                # add edge from mesh to grid
                G_down.add_edge(u, v)
                d = np.sqrt(
                    np.sum(
                        (G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]) ** 2
                    )
                )
                G_down.edges[u, v]["len"] = d
                G_down.edges[u, v]["vdiff"] = (
                    G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]
                )

            # relabel nodes to integers (sorted)
            G_down_int = networkx.convert_node_labels_to_integers(
                G_down, first_label=start_index, ordering="sorted"
            )  # Issue with sorting here
            G_down_int = gutils.sort_nodes_internally(G_down_int)
            pyg_down = gutils.from_networkx_with_start_index(
                G_down_int, start_index
            )

            # Create up graph, invert downwards edges
            up_edges = torch.stack(
                (pyg_down.edge_index[1], pyg_down.edge_index[0]), dim=0
            )
            pyg_up = pyg_down.clone()
            pyg_up.edge_index = up_edges

            up_graphs.append(pyg_up)
            down_graphs.append(pyg_down)

            if mesh_plot_function is not None:
                mesh_plot_function(
                    pyg_down,
                    f"Down graph, {from_level} -> {to_level}",
                )
                mesh_plot_function(
                    pyg_down,
                    f"Up graph, {to_level} -> {from_level}",
                )

        # Save up and down edges
        save_graphs["mesh_up"] = up_graphs
        save_graphs["mesh_down"] = down_graphs

        # Extract intra-level edges for m2m
        m2m_graphs = [
            gutils.from_networkx_with_start_index(
                networkx.convert_node_labels_to_integers(
                    level_graph, first_label=start_index, ordering="sorted"
                ),
                start_index,
            )
            for level_graph, start_index in zip(G, first_index_level)
        ]

        mesh_pos = [graph.pos.to(torch.float32) for graph in m2m_graphs]

        # For use in g2m and m2g
        G_bottom_mesh = G[0]
    else:
        # TODO Non-hierarchical implementation
        raise NotImplementedError(
            "Non-hierarchical mesh graph not implemented with proper "
            "oceanographic masking"
        )

        # combine all levels to one graph
        G_tot = G[0]
        for lev in range(1, len(G)):
            nodes = list(G[lev - 1].nodes)
            n = int(np.sqrt(len(nodes)))
            ij = (
                np.array(nodes)
                .reshape((n, n, 2))[1::nx, 1::nx, :]
                .reshape(int(n / nx) ** 2, 2)
            )
            ij = [tuple(x) for x in ij]
            G[lev] = networkx.relabel_nodes(G[lev], dict(zip(G[lev].nodes, ij)))
            G_tot = networkx.compose(G_tot, G[lev])

        # Relabel mesh nodes to start with 0
        G_tot = gutils.prepend_node_index(G_tot, 0)

        # relabel nodes to integers (sorted)
        G_int = networkx.convert_node_labels_to_integers(
            G_tot, first_label=0, ordering="sorted"
        )

        # Graph to use in g2m and m2g
        G_bottom_mesh = G_tot

        # export the nx graph to PyTorch geometric
        pyg_m2m = from_networkx(G_int)
        m2m_graphs = [pyg_m2m]
        mesh_pos = [pyg_m2m.pos.to(torch.float32)]

        if mesh_plot_function is not None:
            mesh_plot_function(pyg_m2m, "Mesh-to-mesh")

    save_graphs["m2m"] = m2m_graphs
    return mesh_pos, G_bottom_mesh, save_graphs
