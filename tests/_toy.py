"""Small attributed graphs for tests that must not depend on RDKit."""

import networkx as nx


def attributed(atoms: list[tuple[int, int, int]], bonds: list[tuple[int, int, str]]) -> nx.Graph:
    """Build an attributed graph from (element, hcount, cycles) atoms and (u, v, order) bonds."""
    g = nx.Graph()
    for i, (element, hcount, cycles) in enumerate(atoms):
        g.add_node(
            i,
            element=element,
            charge=0,
            isotope=0,
            hcount=hcount,
            radicals=0,
            aromatic=False,
            cycles=cycles,
            index=i,
        )
    for u, v, order in bonds:
        aromatic = order == "AROMATIC"
        g.add_edge(u, v, order=order, aromatic=aromatic)
        if aromatic:
            g.nodes[u]["aromatic"] = g.nodes[v]["aromatic"] = True
    return g


def ring(element: int, hcount: int, size: int, order: str) -> nx.Graph:
    """A monocyclic ring of identical atoms."""
    return attributed(
        [(element, hcount, 1)] * size, [(i, (i + 1) % size, order) for i in range(size)]
    )
