from atommapper.numbering import assign_identifiers

SOURCE_RANK = {0: 1, 1: 0, 2: 2}
TARGET_RANK = {0: 0, 1: 2, 3: 1}


def test_conserved_first_then_source_only_then_target_only():
    numbering = assign_identifiers({0: 1, 1: 0}, [0, 1, 2], [0, 1, 3], SOURCE_RANK, TARGET_RANK)
    # conserved in source canonical order: vertex 1 (rank 0) then vertex 0 (rank 1)
    assert numbering.source == {1: 1, 0: 2, 2: 3}
    assert numbering.target == {0: 1, 1: 2, 3: 4}


def test_identifiers_are_positive_unique_and_shared_only_by_conserved_vertices():
    numbering = assign_identifiers({0: 1, 1: 0}, [0, 1, 2], [0, 1, 3], SOURCE_RANK, TARGET_RANK)
    assert all(i > 0 for i in [*numbering.source.values(), *numbering.target.values()])
    shared = set(numbering.source.values()) & set(numbering.target.values())
    assert shared == {1, 2}


def test_empty_mapping_numbers_every_vertex_one_sided():
    numbering = assign_identifiers({}, [0, 1], [0], {0: 1, 1: 0}, {0: 0})
    assert numbering.source == {1: 1, 0: 2}
    assert numbering.target == {0: 3}
