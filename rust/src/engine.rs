//! Connected common edge subgraphs of labelled graphs, with a step budget.
//!
//! The search runs on the bond-pair compatibility graph (the modular product of the two
//! line graphs): a vertex is a pair of bonds, one per graph, with the same end labels, and
//! two vertices are joined by a c-edge when the bonds meet in both graphs at atoms of the
//! same label, or by a d-edge when they meet in neither. Koch's algorithm (2001) lists the
//! maximal cliques connected through c-edges: the connected common fragments. A clique is
//! a common edge subgraph unless it maps a triangle onto a three-armed star, which is
//! checked on each clique.
//!
//! The search stops on a count of steps, so a search stopped by its budget returns the same
//! partial result on every machine. An optional deadline stops it on the clock as well, read
//! every `CLOCK_EVERY` steps (fewer on large compatibility graphs, whose steps cost more,
//! and also while those are built); a search the clock stopped depends on machine speed
//! and load and says so (`Stop::Time`). Ties are broken by input order: callers that want results
//! independent of atom numbering pass atoms and bonds in a canonical order.
//!
//! A step does not cost the same on every pair (it reads bitsets as long as the compatibility
//! graph is large), so the engine also counts its work in units that do (`Work`: the steps
//! and what they read) and stops when their cost, at the caller's prices (`WorkCosts`),
//! passes `FragmentLimits::max_work` (`Stop::Work`). That stop reads no clock: like the
//! step budget, it returns the same result on every machine. The set-up of the
//! compatibility graph is charged before it is built, so a pair whose set-up alone costs
//! more than the limit stops at once.

use std::cell::Cell;
use std::collections::BTreeSet;
use std::time::Instant;

/// Steps between two readings of the clock: about a millisecond on drug-sized pairs.
pub const CLOCK_EVERY: u64 = 1024;

/// Compatibility graphs above this many vertices also read the clock while they are built
/// and before their first step: on them the set-up alone takes seconds (two alkane chains
/// of 140 and 141 carbons: 19,460 vertices, 1.7 s before the first step). Smaller graphs
/// are always built, so a zero time limit still returns the greedy clique of a drug-sized
/// pair.
pub const SETUP_CLOCK_PAIRS: usize = 4096;

/// Steps between two readings of the clock on a compatibility graph of `n` vertices: a
/// step costs more on a larger graph, so the clock is read more often, down to every step.
fn clock_every(n: usize) -> u64 {
    ((1u64 << 20) / n.max(1) as u64).clamp(1, CLOCK_EVERY)
}

/// Whether the deadline has passed.
#[inline]
fn passed(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|t| Instant::now() >= t)
}

/// Whether the set-up of a graph of `n` vertices should stop: it is large and the deadline
/// has passed.
fn setup_late(n: usize, deadline: Option<Instant>) -> bool {
    n > SETUP_CLOCK_PAIRS && passed(deadline)
}

/// Whether `steps` is a step at which the clock is read and the deadline has passed.
#[inline]
fn late(steps: u64, every: u64, deadline: Option<Instant>) -> bool {
    steps.is_multiple_of(every) && passed(deadline)
}

/// A labelled graph: one label per atom, bonds as atom index pairs.
pub struct Graph {
    pub labels: Vec<i64>,
    pub bonds: Vec<(usize, usize)>,
}

/// Why a search stopped before proving it had every fragment it lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The step budget ran out; the fragments are those found so far.
    Steps,
    /// More compatible bond pairs than `max_pairs`; nothing was searched.
    Pairs,
    /// The deadline passed; the fragments are those found so far, and depend on machine
    /// speed and load.
    Time,
    /// The work limit was passed (`FragmentLimits::max_work`); the fragments are those
    /// found so far, the same on every machine.
    Work,
}

/// The engine's work, in units whose cost does not depend on the pair.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Work {
    /// Pairs of vertices of the compatibility graph whose relation the set-up reads.
    pub relation_pairs: u64,
    /// Steps of the enumeration: what a call of the recursion costs whatever it reads.
    pub steps: u64,
    /// Bitset words (and single vertices) the enumeration reads.
    pub words: u64,
    /// Bonds of the cliques split into consistent parts, scored or turned into fragments.
    pub leaf_bonds: u64,
}

/// The price of each unit of `Work`, in the caller's unit (AtomMapper: picoseconds of its
/// virtual clock). All zero by default: the work is counted but costs nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkCosts {
    pub relation_pair: u64,
    pub step: u64,
    pub word: u64,
    pub leaf_bond: u64,
}

impl Work {
    /// The cost of the work at `costs`, saturating.
    pub fn cost(&self, costs: &WorkCosts) -> u64 {
        (self.relation_pairs.saturating_mul(costs.relation_pair))
            .saturating_add(self.steps.saturating_mul(costs.step))
            .saturating_add(self.words.saturating_mul(costs.word))
            .saturating_add(self.leaf_bonds.saturating_mul(costs.leaf_bond))
    }

    fn add(&mut self, other: Work) {
        self.relation_pairs += other.relation_pairs;
        self.steps += other.steps;
        self.words += other.words;
        self.leaf_bonds += other.leaf_bonds;
    }
}

/// The atom shared by two bonds, if any.
#[inline]
fn shared(a: (usize, usize), b: (usize, usize)) -> Option<usize> {
    if a.0 == b.0 || a.0 == b.1 {
        Some(a.0)
    } else if a.1 == b.0 || a.1 == b.1 {
        Some(a.1)
    } else {
        None
    }
}

/// Row-major bitset matrix.
struct BitMatrix {
    words: usize,
    data: Vec<u64>,
}

impl BitMatrix {
    fn new(rows: usize, words: usize) -> Self {
        BitMatrix {
            words,
            data: vec![0; rows * words],
        }
    }
    #[inline]
    fn row(&self, r: usize) -> &[u64] {
        &self.data[r * self.words..(r + 1) * self.words]
    }
    #[inline]
    fn set(&mut self, r: usize, c: usize) {
        self.data[r * self.words + (c >> 6)] |= 1u64 << (c & 63);
    }
}

/// Call `f` on every set bit, lowest first.
#[inline]
fn for_each_bit(bits: &[u64], mut f: impl FnMut(usize)) {
    for (w, &word) in bits.iter().enumerate() {
        let mut word = word;
        while word != 0 {
            let b = word.trailing_zeros() as usize;
            word &= word - 1;
            f((w << 6) + b);
        }
    }
}

impl Graph {
    /// Bond type: its two end labels, smaller first. A tuple, so any labels compare exactly.
    fn code(&self, b: usize) -> (i64, i64) {
        let (x, y) = (self.labels[self.bonds[b].0], self.labels[self.bonds[b].1]);
        (x.min(y), x.max(y))
    }

    /// Bond index triples forming triangles.
    fn triangles(&self) -> Vec<[usize; 3]> {
        let mut incident: Vec<Vec<usize>> = vec![Vec::new(); self.labels.len()];
        for (b, &(u, v)) in self.bonds.iter().enumerate() {
            incident[u].push(b);
            incident[v].push(b);
        }
        let mut out = BTreeSet::new();
        for (b, &(a, c)) in self.bonds.iter().enumerate() {
            for &b2 in &incident[a] {
                if b2 == b {
                    continue;
                }
                let other = if self.bonds[b2].0 == a {
                    self.bonds[b2].1
                } else {
                    self.bonds[b2].0
                };
                for &b3 in &incident[c] {
                    if b3 != b
                        && b3 != b2
                        && (self.bonds[b3].0 == other || self.bonds[b3].1 == other)
                    {
                        let mut t = [b, b2, b3];
                        t.sort_unstable();
                        out.insert(t);
                    }
                }
            }
        }
        out.into_iter().collect()
    }
}

/// The atom map a set of matched bonds induces, or None when it is not consistent.
///
/// An atom with two or more matched bonds goes to the atom their images share; an atom
/// with one goes to the far end of its bond's image from its neighbour's image. A bond
/// matched alone pairs its ends by label, lower atom first when both orders fit.
fn atom_map(bonds: &[(usize, usize)], g1: &Graph, g2: &Graph) -> Option<Vec<(usize, usize)>> {
    let mut image: Vec<Option<usize>> = vec![None; g1.labels.len()];
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); g1.labels.len()];
    for (k, &(i, _)) in bonds.iter().enumerate() {
        incident[g1.bonds[i].0].push(k);
        incident[g1.bonds[i].1].push(k);
    }
    for (a, ks) in incident.iter().enumerate() {
        if ks.len() >= 2 {
            image[a] = Some(shared(g2.bonds[bonds[ks[0]].1], g2.bonds[bonds[ks[1]].1])?);
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for &(i, j) in bonds {
            let (a, c) = g1.bonds[i];
            let (x, y) = g2.bonds[j];
            for (end, other) in [(a, c), (c, a)] {
                if image[end].is_none() {
                    if let Some(o) = image[other] {
                        image[end] = Some(if o == x { y } else { x });
                        changed = true;
                    }
                }
            }
        }
    }
    for &(i, j) in bonds {
        let (a, c) = g1.bonds[i];
        if image[a].is_none() && image[c].is_none() {
            let (x, y) = g2.bonds[j];
            let straight = g1.labels[a] == g2.labels[x] && g1.labels[c] == g2.labels[y];
            let (ia, ic) = if straight { (x, y) } else { (y, x) };
            image[a] = Some(ia);
            image[c] = Some(ic);
        }
    }
    // Check: labels agree, the map is one to one, and every matched bond maps onto its pair.
    let mut used = vec![false; g2.labels.len()];
    let mut out = Vec::new();
    for (a, img) in image.iter().enumerate() {
        if let Some(b) = *img {
            if g1.labels[a] != g2.labels[b] || used[b] {
                return None;
            }
            used[b] = true;
            out.push((a, b));
        }
    }
    for &(i, j) in bonds {
        let (a, c) = g1.bonds[i];
        let (ia, ic) = (image[a]?, image[c]?);
        let (x, y) = g2.bonds[j];
        if !((ia == x && ic == y) || (ia == y && ic == x)) {
            return None;
        }
    }
    Some(out)
}

// ---- Maximal connected common fragments (Koch 2001) ----

/// Limits of `maximal_fragments`.
#[derive(Clone, Copy)]
pub struct FragmentLimits {
    /// Search steps (calls of the recursion) for the whole enumeration.
    pub max_steps: u64,
    /// Maximum fragments kept; reaching it sets `truncated`. It ends a full enumeration;
    /// a best-only search drops further ties of the best so far and goes on, since a
    /// better fragment clears the list.
    pub max_fragments: usize,
    /// Ceiling on compatible bond pairs, which bounds memory (quadratic) and step cost.
    pub max_pairs: usize,
    /// Fragments with fewer matched bonds are not listed.
    pub min_bonds: usize,
    /// List only the largest fragments (every tied one): branches that cannot reach the
    /// largest consistent fragment found so far are pruned.
    pub best_only: bool,
    /// Wall-clock limit, read every `CLOCK_EVERY` steps; `None` never reads the clock.
    pub deadline: Option<Instant>,
    /// Limit on the cost of the work at `costs`, checked at every step and, for the
    /// set-up, before it starts; `None` for no limit.
    pub max_work: Option<u64>,
    /// The price of each unit of work.
    pub costs: WorkCosts,
    /// With rings: a trimmed score some common fragment is known to reach (from a search on
    /// finer labels, say). The search starts from it, so it cuts branches that cannot
    /// reach it from the start; the fragments listed are the same, since only those of the
    /// best score (at least this one) are kept. A fragment on finer labels may still score
    /// more than any part this search scores (it scores parts of maximal cliques, and the
    /// trim is not monotone: extending a fragment can unravel a ring), so a search that
    /// ends with no scored part at all runs again from zero, with the steps left.
    pub known_score: usize,
}

impl Default for FragmentLimits {
    fn default() -> Self {
        FragmentLimits {
            max_steps: 20_000_000,
            max_fragments: 4096,
            max_pairs: 20_000,
            min_bonds: 3,
            best_only: false,
            deadline: None,
            max_work: None,
            costs: WorkCosts::default(),
            known_score: 0,
        }
    }
}

/// Ring membership of one graph's atoms and bonds, for `ring_fragments`.
pub struct Rings {
    pub atoms: Vec<bool>,
    pub bonds: Vec<bool>,
}

/// One connected common edge subgraph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fragment {
    /// Matched bond pairs (bond of graph 1, bond of graph 2).
    pub bonds: Vec<(usize, usize)>,
    /// The atom correspondence the bonds induce, sorted by graph-1 atom.
    pub atoms: Vec<(usize, usize)>,
}

/// Every maximal connected common edge subgraph, as far as the limits allowed.
#[derive(Debug, Default)]
pub struct Fragments {
    pub fragments: Vec<Fragment>,
    pub steps: u64,
    pub stop: Option<Stop>,
    /// The cap was reached, or a fragment did not induce a consistent atom map: the list
    /// may not cover every common fragment.
    pub truncated: bool,
    /// Maximal cliques that mapped a triangle onto a three-armed star and were split into
    /// their consistent connected parts.
    pub split: usize,
    /// Vertices of the compatibility graph.
    pub pairs: usize,
    /// With rings, the best trimmed score found (or `known_score`, if none beat it).
    pub best_score: usize,
    /// The work done (and, when the set-up alone passed the limit, the set-up not done).
    pub work: Work,
}

impl BitMatrix {
    #[inline]
    fn get(&self, r: usize, c: usize) -> bool {
        self.data[r * self.words + (c >> 6)] >> (c & 63) & 1 == 1
    }
}

/// Compatible bond pairs `(vi[x], vj[x])`: same end labels, in (graph-1 bond, graph-2
/// bond) order.
fn compatible_pairs(g1: &Graph, g2: &Graph) -> (Vec<usize>, Vec<usize>) {
    let codes2: Vec<(i64, i64)> = (0..g2.bonds.len()).map(|b| g2.code(b)).collect();
    let (mut vi, mut vj) = (Vec::new(), Vec::new());
    for i in 0..g1.bonds.len() {
        let code1 = g1.code(i);
        for (j, &code2) in codes2.iter().enumerate() {
            if code2 == code1 {
                vi.push(i);
                vj.push(j);
            }
        }
    }
    (vi, vj)
}

/// The two relations of the compatibility graph: c-edges (the bonds meet in both graphs,
/// at atoms of the same label) and d-edges (they meet in neither); None when the deadline
/// passes first (read only on large graphs).
fn build_relations(
    vi: &[usize],
    vj: &[usize],
    g1: &Graph,
    g2: &Graph,
    deadline: Option<Instant>,
) -> Option<(BitMatrix, BitMatrix)> {
    let n = vi.len();
    let words = n.div_ceil(64);
    let (mut c, mut d) = (BitMatrix::new(n, words), BitMatrix::new(n, words));
    for x in 0..n {
        if x.is_multiple_of(64) && setup_late(n, deadline) {
            return None;
        }
        let (i, j) = (vi[x], vj[x]);
        for y in (x + 1)..n {
            let (k, l) = (vi[y], vj[y]);
            if i == k || j == l {
                continue;
            }
            match (
                shared(g1.bonds[i], g1.bonds[k]),
                shared(g2.bonds[j], g2.bonds[l]),
            ) {
                (None, None) => {
                    d.set(x, y);
                    d.set(y, x);
                }
                (Some(a), Some(b)) if g1.labels[a] == g2.labels[b] => {
                    c.set(x, y);
                    c.set(y, x);
                }
                _ => {}
            }
        }
    }
    Some((c, d))
}

/// Connected components of `members` under c-edges, each sorted, in order of first member.
fn c_components(members: &[usize], c: &BitMatrix) -> Vec<Vec<usize>> {
    let mut seen = vec![false; members.len()];
    let mut out = Vec::new();
    for start in 0..members.len() {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut component = vec![members[start]];
        let mut frontier = vec![start];
        while let Some(p) = frontier.pop() {
            for q in 0..members.len() {
                if !seen[q] && c.get(members[p], members[q]) {
                    seen[q] = true;
                    component.push(members[q]);
                    frontier.push(q);
                }
            }
        }
        component.sort_unstable();
        out.push(component);
    }
    out
}

/// Three members of `clique` that map a triangle of one graph onto a three-armed star of
/// the other, if any.
fn star_triple(
    clique: &[usize],
    vi: &[usize],
    vj: &[usize],
    g1: &Graph,
    g2: &Graph,
    tri1: &[[usize; 3]],
    tri2: &[[usize; 3]],
) -> Option<[usize; 3]> {
    let mut at1: Vec<Option<usize>> = vec![None; g1.bonds.len()];
    let mut at2: Vec<Option<usize>> = vec![None; g2.bonds.len()];
    for &x in clique {
        at1[vi[x]] = Some(x);
        at2[vj[x]] = Some(x);
    }
    let star = |triangles: &[[usize; 3]], at: &[Option<usize>], other: &[usize], g: &Graph| {
        triangles.iter().find_map(|t| {
            let xs = [at[t[0]]?, at[t[1]]?, at[t[2]]?];
            let (a, b, c) = (
                g.bonds[other[xs[0]]],
                g.bonds[other[xs[1]]],
                g.bonds[other[xs[2]]],
            );
            let centre = shared(a, b)?;
            (c.0 == centre || c.1 == centre).then_some(xs)
        })
    };
    star(tri1, &at1, vj, g2).or_else(|| star(tri2, &at2, vi, g1))
}

/// The consistent connected parts of a sorted clique: itself, or, when it maps a triangle
/// onto a three-armed star, the parts left by dropping each member of such a triple in
/// turn, recursively, down to `min_bonds` members.
#[allow(clippy::too_many_arguments)]
fn consistent_parts(
    sorted: Vec<usize>,
    vi: &[usize],
    vj: &[usize],
    g1: &Graph,
    g2: &Graph,
    tri1: &[[usize; 3]],
    tri2: &[[usize; 3]],
    c: &BitMatrix,
    min_bonds: usize,
) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut pending = vec![sorted];
    while let Some(part) = pending.pop() {
        match star_triple(&part, vi, vj, g1, g2, tri1, tri2) {
            None => out.push(part),
            Some(triple) => {
                for drop in triple {
                    let rest: Vec<usize> = part.iter().copied().filter(|&x| x != drop).collect();
                    for component in c_components(&rest, c) {
                        if component.len() >= min_bonds {
                            pending.push(component);
                        }
                    }
                }
            }
        }
    }
    out
}

/// Bond index of each atom pair of `g`, both orientations.
fn bond_index(g: &Graph) -> std::collections::HashMap<(usize, usize), usize> {
    let mut out = std::collections::HashMap::with_capacity(2 * g.bonds.len());
    for (b, &(u, v)) in g.bonds.iter().enumerate() {
        out.insert((u, v), b);
        out.insert((v, u), b);
    }
    out
}

/// Bonds of the largest piece left when the atom map `atoms` is trimmed to complete rings
/// (`ring_fragments`). The kept part is every graph-1 bond between mapped atoms whose
/// images are bonded in graph 2.
fn trimmed_bonds(
    atoms: &[(usize, usize)],
    g1: &Graph,
    r1: &Rings,
    r2: &Rings,
    partners2: &std::collections::HashMap<(usize, usize), usize>,
) -> usize {
    let mut image: Vec<Option<usize>> = vec![None; g1.labels.len()];
    for &(a, b) in atoms {
        image[a] = Some(b);
    }
    // Kept bonds: (atom, atom, ring bond in either graph).
    let mut kept: Vec<(usize, usize, bool)> = Vec::new();
    for (b, &(u, v)) in g1.bonds.iter().enumerate() {
        if let (Some(x), Some(y)) = (image[u], image[v]) {
            if let Some(&j) = partners2.get(&(x, y)) {
                kept.push((u, v, r1.bonds[b] || r2.bonds[j]));
            }
        }
    }
    let ring_atom = |a: usize| r1.atoms[a] || image[a].is_some_and(|x| r2.atoms[x]);
    let mut alive: Vec<bool> = image.iter().map(Option::is_some).collect();
    loop {
        kept.retain(|&(u, v, _)| alive[u] && alive[v]);
        let bridge = bridges(g1.labels.len(), &kept);
        let mut on_cycle = vec![false; g1.labels.len()];
        for (k, &(u, v, _)) in kept.iter().enumerate() {
            if !bridge[k] {
                on_cycle[u] = true;
                on_cycle[v] = true;
            }
        }
        let lonely: Vec<usize> = (0..alive.len())
            .filter(|&a| alive[a] && ring_atom(a) && !on_cycle[a])
            .collect();
        let drop: Vec<usize> = if lonely.is_empty() {
            kept.iter()
                .enumerate()
                .filter(|&(k, &(_, _, ring))| ring && bridge[k])
                .flat_map(|(_, &(u, v, _))| [u, v])
                .collect()
        } else {
            lonely
        };
        if drop.is_empty() {
            break;
        }
        for a in drop {
            alive[a] = false;
        }
    }
    // The largest piece, in bonds.
    let n = g1.labels.len();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v, _) in &kept {
        adjacency[u].push(v);
        adjacency[v].push(u);
    }
    let mut seen = vec![false; n];
    let mut best = 0;
    for start in 0..n {
        if seen[start] || adjacency[start].is_empty() {
            continue;
        }
        seen[start] = true;
        let (mut stack, mut degree_sum) = (vec![start], 0);
        while let Some(a) = stack.pop() {
            degree_sum += adjacency[a].len();
            for &w in &adjacency[a] {
                if !seen[w] {
                    seen[w] = true;
                    stack.push(w);
                }
            }
        }
        best = best.max(degree_sum / 2);
    }
    best
}

/// Which of `edges` (on `n` vertices, simple) are bridges (Tarjan, iterative).
fn bridges(n: usize, edges: &[(usize, usize, bool)]) -> Vec<bool> {
    let mut adjacency: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for (k, &(u, v, _)) in edges.iter().enumerate() {
        adjacency[u].push((v, k));
        adjacency[v].push((u, k));
    }
    let mut out = vec![false; edges.len()];
    let mut disc = vec![usize::MAX; n];
    let mut low = vec![0; n];
    let mut time = 0;
    for root in 0..n {
        if disc[root] != usize::MAX || adjacency[root].is_empty() {
            continue;
        }
        disc[root] = time;
        low[root] = time;
        time += 1;
        // (vertex, edge it was reached by, next neighbour to look at)
        let mut stack = vec![(root, usize::MAX, 0usize)];
        while let Some(&mut (v, via, ref mut next)) = stack.last_mut() {
            if let Some(&(w, k)) = adjacency[v].get(*next) {
                *next += 1;
                if k == via {
                    continue;
                }
                if disc[w] == usize::MAX {
                    disc[w] = time;
                    low[w] = time;
                    time += 1;
                    stack.push((w, k, 0));
                } else {
                    low[v] = low[v].min(disc[w]);
                }
            } else {
                stack.pop();
                if let Some(&(parent, _, _)) = stack.last() {
                    low[parent] = low[parent].min(low[v]);
                    if low[v] > disc[parent] {
                        out[via] = true;
                    }
                }
            }
        }
    }
    out
}

/// The consistent connected parts of a clique (`consistent_parts`).
type Parts<'a> = dyn Fn(&[usize]) -> Vec<Vec<usize>> + 'a;
/// The score of a consistent part (`trimmed_bonds`).
type Score<'a> = dyn Fn(&[usize]) -> usize + 'a;

struct Koch<'a> {
    c: &'a BitMatrix,
    d: &'a BitMatrix,
    words: usize,
    min_bonds: usize,
    /// With `best_only`, the size of the largest consistent clique found so far: the bound
    /// the search must reach. Inconsistent cliques (a triangle onto a star) never raise it,
    /// so the parts they split into are never pruned away.
    best_only: bool,
    best: usize,
    /// With `best_only`, a maximal clique is split into its consistent parts at once,
    /// and the parts compete by size, so the kept list holds only consistent fragments.
    parts: &'a Parts<'a>,
    /// With rings: the score of a consistent part (the bonds of its largest piece once
    /// trimmed to complete rings). The search then keeps the parts of the best score in
    /// `scored` and prunes against that score, not size.
    scorer: Option<&'a Score<'a>>,
    scored: Vec<(Vec<usize>, usize)>,
    best_score: usize,
    /// Per vertex: its graph-1 bond, its graph-2 bond and the class of their bond type,
    /// for the bound on how many more bonds the clique can take.
    vi: &'a [usize],
    vj: &'a [usize],
    class: &'a [usize],
    classes: usize,
    bonds1: usize,
    bonds2: usize,
    max_steps: u64,
    deadline: Option<Instant>,
    max_work: Option<u64>,
    costs: WorkCosts,
    /// The cost of the work done before the enumeration (the set-up).
    base_cost: u64,
    /// Bitset words read so far (`Work::words`); a cell, since the helpers that count
    /// borrow the search shared.
    words_read: Cell<u64>,
    leaf_bonds: u64,
    /// Steps between two readings of the clock (`clock_every`).
    every: u64,
    max_raw: usize,
    /// The current root: vertices below it were roots already, so every connected clique
    /// containing one of them has been listed.
    root: usize,
    steps: u64,
    stopped: bool,
    /// Stopped by the deadline, not the step budget.
    timed: bool,
    /// Stopped by the work limit.
    worked: bool,
    capped: bool,
    /// The best-only lists dropped a tie of the current best size or score.
    raw_full: bool,
    scored_full: bool,
    /// Maximal cliques that mapped a triangle onto a three-armed star.
    split: usize,
    raw: Vec<Vec<usize>>,
}

fn count_bits(bits: &[u64]) -> usize {
    bits.iter().map(|w| w.count_ones() as usize).sum()
}

impl Koch<'_> {
    /// Count `n` bitset words read.
    #[inline]
    fn read(&self, n: usize) {
        self.words_read.set(self.words_read.get() + n as u64);
    }

    /// The work of the enumeration so far.
    fn work(&self) -> Work {
        Work {
            relation_pairs: 0,
            steps: self.steps,
            words: self.words_read.get(),
            leaf_bonds: self.leaf_bonds,
        }
    }

    /// Whether the cost of the work so far, the set-up included, is past the limit.
    #[inline]
    fn over_work(&self) -> bool {
        self.max_work
            .is_some_and(|max| self.base_cost.saturating_add(self.work().cost(&self.costs)) > max)
    }

    /// One call of Koch's recursion: `p` extends the clique through a shared atom, `d`
    /// holds vertices compatible with every member but not yet connected to one, `s` holds
    /// vertices whose cliques were listed from an earlier root.
    fn extend(&mut self, clique: &mut Vec<usize>, mut p: Vec<u64>, d: Vec<u64>, mut s: Vec<u64>) {
        if self.stopped || self.capped {
            return;
        }
        self.steps += 1;
        if self.steps > self.max_steps {
            self.steps = self.max_steps;
            self.stopped = true;
            return;
        }
        if self.over_work() {
            self.stopped = true;
            self.worked = true;
            return;
        }
        if late(self.steps, self.every, self.deadline) {
            self.stopped = true;
            self.timed = true;
            return;
        }
        self.read(2 * self.words);
        let (np, ns) = (count_bits(&p), count_bits(&s));
        if np == 0 {
            if ns == 0 && clique.len() >= self.min_bonds {
                if !self.best_only {
                    if self.raw.len() < self.max_raw {
                        self.raw.push(clique.clone());
                    } else {
                        self.capped = true;
                    }
                    return;
                }
                self.leaf_bonds += clique.len() as u64;
                let parts = (self.parts)(clique);
                if parts.len() != 1 || parts[0].len() != clique.len() {
                    self.split += 1;
                }
                // A full list drops further ties of the best so far, but the search goes
                // on: a better fragment clears it.
                for part in parts {
                    if let Some(score) = self.scorer {
                        self.leaf_bonds += part.len() as u64;
                        let s = score(&part);
                        if s > 0 && s >= self.best_score {
                            if s > self.best_score {
                                self.best_score = s;
                                self.scored.retain(|(_, t)| *t >= s);
                                self.scored_full = false;
                            }
                            if self.scored.len() < self.max_raw {
                                self.scored.push((part.clone(), s));
                            } else {
                                self.scored_full = true;
                            }
                        }
                    }
                    if part.len() < self.best {
                        continue;
                    }
                    if part.len() > self.best {
                        self.best = part.len();
                        let best = self.best;
                        self.raw.retain(|r| r.len() >= best);
                        self.raw_full = false;
                    }
                    if self.raw.len() < self.max_raw {
                        self.raw.push(part);
                    } else {
                        self.raw_full = true;
                    }
                }
            }
            return;
        }
        // With a scorer, branches that cannot reach the best score go: the best score is
        // never above the best size, so the largest fragments are still all found.
        let bound = if self.scorer.is_some() {
            self.min_bonds.max(self.best_score)
        } else if self.best_only {
            self.min_bonds.max(self.best)
        } else {
            self.min_bonds
        };
        // A vertex of `d` that no path of c-edges through `p` and `d` reaches never joins,
        // here or below.
        // The bounds first on `d` as given, then, if they pass, on the part of `d` that can
        // still be reached: the closure is most of a node's cost, and the reached part is
        // a subset, so what the first test cuts the second would too.
        let mut d = d;
        self.read(self.words);
        if clique.len() + np + count_bits(&d) < bound || clique.len() + self.room(&p, &d) < bound {
            return;
        }
        let before = count_bits(&d);
        if before > 0 {
            let reach = self.closure(&p, &d);
            for w in 0..self.words {
                d[w] &= reach[w];
            }
            self.read(2 * self.words);
            let after = count_bits(&d);
            if after < before
                && (clique.len() + np + after < bound || clique.len() + self.room(&p, &d) < bound)
            {
                return;
            }
        }
        let mut members = Vec::with_capacity(np);
        for_each_bit(&self.branches(&p, &d, &s), |u| members.push(u));
        for u in members {
            p[u >> 6] &= !(1u64 << (u & 63));
            let (mut p2, mut d2, mut s2) = (p.clone(), d.clone(), s.clone());
            let mut reachable = Vec::new();
            let mut tested = 0;
            for_each_bit(&d2, |v| {
                tested += 1;
                if self.c.get(u, v) {
                    reachable.push(v);
                }
            });
            self.read(5 * self.words + tested);
            for v in reachable {
                d2[v >> 6] &= !(1u64 << (v & 63));
                if self.done(v) {
                    s2[v >> 6] |= 1u64 << (v & 63);
                } else {
                    p2[v >> 6] |= 1u64 << (v & 63);
                }
            }
            let (cr, dr) = (self.c.row(u), self.d.row(u));
            for w in 0..self.words {
                let n = cr[w] | dr[w];
                p2[w] &= n;
                d2[w] &= n;
                s2[w] &= n;
            }
            clique.push(u);
            self.extend(clique, p2, d2, s2);
            clique.pop();
            s[u >> 6] |= 1u64 << (u & 63);
            if self.stopped || self.capped {
                return;
            }
        }
    }

    /// `start` and the vertices of `within` reachable from it along c-edges.
    fn closure(&self, start: &[u64], within: &[u64]) -> Vec<u64> {
        let words = self.words;
        let mut seen = start.to_vec();
        let mut frontier = start.to_vec();
        let mut next = vec![0u64; words];
        loop {
            next.fill(0);
            self.read(3 * words);
            for_each_bit(&frontier, |v| {
                self.read(words);
                let row = self.c.row(v);
                for w in 0..words {
                    next[w] |= row[w] & within[w] & !seen[w];
                }
            });
            let mut grew = false;
            for w in 0..words {
                grew |= next[w] != 0;
                seen[w] |= next[w];
            }
            if !grew {
                return seen;
            }
            std::mem::swap(&mut frontier, &mut next);
        }
    }

    /// The members of `p` to branch on, under a pivot `u` from `p` or `s`. A maximal
    /// connected clique without `u` holds a vertex incompatible with `u` (else `u`, which
    /// meets the clique, would extend it). That vertex lies in `p`, or in `d`, where the
    /// clique reaches it from a member of `p` along c-edges: branching on the members of
    /// `p` incompatible with `u` or reaching such a vertex of `d` lists every clique.
    fn branches(&self, p: &[u64], d: &[u64], s: &[u64]) -> Vec<u64> {
        let mut pivot = None;
        let mut fewest = usize::MAX;
        // The first vertex with the fewest: none has fewer than zero, so the scan stops there.
        'scan: for bits in [p, s] {
            for (w0, &word) in bits.iter().enumerate() {
                let mut word = word;
                while word != 0 {
                    let u = w0 * 64 + word.trailing_zeros() as usize;
                    word &= word - 1;
                    self.read(self.words);
                    let (cr, dr) = (self.c.row(u), self.d.row(u));
                    let n: usize = (0..self.words)
                        .map(|w| (p[w] & !(cr[w] | dr[w])).count_ones() as usize)
                        .sum();
                    if n < fewest {
                        fewest = n;
                        pivot = Some(u);
                        if n == 0 {
                            break 'scan;
                        }
                    }
                }
            }
        }
        self.read(3 * self.words);
        let Some(u) = pivot else { return p.to_vec() };
        let (cr, dr) = (self.c.row(u), self.d.row(u));
        let away: Vec<u64> = (0..self.words).map(|w| d[w] & !(cr[w] | dr[w])).collect();
        if away.iter().all(|&w| w == 0) {
            return (0..self.words).map(|w| p[w] & !(cr[w] | dr[w])).collect();
        }
        let both: Vec<u64> = (0..self.words).map(|w| p[w] | d[w]).collect();
        let seen = self.closure(&away, &both);
        (0..self.words)
            .map(|w| p[w] & (seen[w] | !(cr[w] | dr[w])))
            .collect()
    }

    fn done(&self, v: usize) -> bool {
        v < self.root
    }

    /// An upper bound on the vertices the clique can still take from `p` and `d`: per
    /// bond type, the fewer of the distinct graph-1 and graph-2 bonds among them, since a
    /// clique uses each bond once.
    fn room(&self, p: &[u64], d: &[u64]) -> usize {
        let mut seen1 = vec![false; self.bonds1];
        let mut seen2 = vec![false; self.bonds2];
        let mut count1 = vec![0usize; self.classes];
        let mut count2 = vec![0usize; self.classes];
        self.read(2 * self.words);
        for bits in [p, d] {
            for_each_bit(bits, |x| {
                self.read(1);
                let k = self.class[x];
                if !seen1[self.vi[x]] {
                    seen1[self.vi[x]] = true;
                    count1[k] += 1;
                }
                if !seen2[self.vj[x]] {
                    seen2[self.vj[x]] = true;
                    count2[k] += 1;
                }
            });
        }
        count1.iter().zip(&count2).map(|(a, b)| *a.min(b)).sum()
    }
}

/// Every maximal connected common edge subgraph of `g1` and `g2` with at least
/// `limits.min_bonds` bonds, within `limits` (Koch, 2001). A maximal clique that maps a
/// triangle onto a three-armed star is split into its consistent connected parts, so that
/// every consistent connected common fragment of two or more bonds lies inside a listed one.
/// A one-bond fragment is listed in the orientation `atom_map` gives it; when its two ends
/// share a label, the reverse orientation is the caller's to add. Roots and branches are
/// taken in vertex order, so the result depends only on the input order.
pub fn maximal_fragments(g1: &Graph, g2: &Graph, limits: FragmentLimits) -> Fragments {
    fragments_impl(g1, g2, None, limits)
}

/// The largest connected common fragments, as `maximal_fragments` with `best_only`, and
/// those whose largest piece, once trimmed to complete rings, has the most bonds (every
/// tie of either, up to `limits.max_fragments` each). The trim keeps a kept
/// ring atom only on a cycle of the kept part and a kept ring bond only off its bridges
/// (a ring atom or bond is one in either graph): it drops ring atoms on no cycle first,
/// which breaks no other cycle, and only when none is left both ends of ring bonds that
/// are bridges, until nothing is dropped. The search prunes branches whose fragments
/// cannot reach the best trimmed size; `limits.best_only` is implied.
pub fn ring_fragments(
    g1: &Graph,
    g2: &Graph,
    rings1: &Rings,
    rings2: &Rings,
    limits: FragmentLimits,
) -> Fragments {
    let limits = FragmentLimits {
        best_only: true,
        ..limits
    };
    fragments_impl(g1, g2, Some((rings1, rings2)), limits)
}

fn fragments_impl(
    g1: &Graph,
    g2: &Graph,
    rings: Option<(&Rings, &Rings)>,
    limits: FragmentLimits,
) -> Fragments {
    if g1.bonds.len() > g2.bonds.len() {
        let mut out = fragments_impl(g2, g1, rings.map(|(a, b)| (b, a)), limits);
        for fragment in out.fragments.iter_mut() {
            for pair in fragment.bonds.iter_mut() {
                *pair = (pair.1, pair.0);
            }
            for pair in fragment.atoms.iter_mut() {
                *pair = (pair.1, pair.0);
            }
            fragment.atoms.sort_unstable();
            fragment.bonds.sort_unstable();
        }
        return out;
    }
    let (vi, vj) = compatible_pairs(g1, g2);
    let n = vi.len();
    if n > limits.max_pairs {
        return Fragments {
            stop: Some(Stop::Pairs),
            pairs: n,
            ..Fragments::default()
        };
    }
    if n == 0 {
        return Fragments::default();
    }
    // The set-up reads the relation of every two vertices: its work is known before it
    // starts, and a pair whose set-up alone passes the work limit is not set up.
    let setup = Work {
        relation_pairs: (n as u64) * (n as u64 - 1) / 2,
        ..Work::default()
    };
    let base_cost = setup.cost(&limits.costs);
    if limits.max_work.is_some_and(|max| base_cost > max) {
        return Fragments {
            stop: Some(Stop::Work),
            pairs: n,
            work: setup,
            ..Fragments::default()
        };
    }
    let Some((c, d)) = build_relations(&vi, &vj, g1, g2, limits.deadline) else {
        return Fragments {
            stop: Some(Stop::Time),
            pairs: n,
            work: setup,
            ..Fragments::default()
        };
    };
    let words = n.div_ceil(64);
    let (tri1, tri2) = (g1.triangles(), g2.triangles());
    let mut codes: Vec<(i64, i64)> = vi.iter().map(|&i| g1.code(i)).collect();
    codes.sort_unstable();
    codes.dedup();
    let class: Vec<usize> = vi
        .iter()
        .map(|&i| codes.binary_search(&g1.code(i)).unwrap())
        .collect();
    let parts = |clique: &[usize]| {
        let mut sorted = clique.to_vec();
        sorted.sort_unstable();
        consistent_parts(
            sorted,
            &vi,
            &vj,
            g1,
            g2,
            &tri1,
            &tri2,
            &c,
            limits.min_bonds.max(1),
        )
    };
    let partners2 = bond_index(g2);
    let score = |part: &[usize]| {
        let (r1, r2) = rings.expect("scored only with rings");
        let bonds: Vec<(usize, usize)> = part.iter().map(|&x| (vi[x], vj[x])).collect();
        atom_map(&bonds, g1, g2).map_or(0, |atoms| trimmed_bonds(&atoms, g1, r1, r2, &partners2))
    };
    let mut koch = Koch {
        c: &c,
        d: &d,
        words,
        min_bonds: limits.min_bonds.max(1),
        best_only: limits.best_only,
        best: 0,
        parts: &parts,
        scorer: if rings.is_some() { Some(&score) } else { None },
        scored: Vec::new(),
        best_score: if rings.is_some() {
            limits.known_score
        } else {
            0
        },
        vi: &vi,
        vj: &vj,
        class: &class,
        classes: codes.len(),
        bonds1: g1.bonds.len(),
        bonds2: g2.bonds.len(),
        max_steps: limits.max_steps,
        deadline: limits.deadline,
        max_work: limits.max_work,
        costs: limits.costs,
        base_cost,
        words_read: Cell::new(0),
        leaf_bonds: 0,
        every: clock_every(n),
        max_raw: limits.max_fragments,
        root: 0,
        steps: 0,
        stopped: false,
        timed: false,
        worked: false,
        capped: false,
        raw_full: false,
        scored_full: false,
        split: 0,
        raw: Vec::new(),
    };
    for u in 0..n {
        koch.root = u;
        let (mut p, mut dd, mut s) = (vec![0u64; words], vec![0u64; words], vec![0u64; words]);
        koch.read(n);
        for v in 0..n {
            if c.get(u, v) {
                if v < u {
                    s[v >> 6] |= 1u64 << (v & 63);
                } else {
                    p[v >> 6] |= 1u64 << (v & 63);
                }
            } else if d.get(u, v) {
                dd[v >> 6] |= 1u64 << (v & 63);
            }
        }
        let mut clique = vec![u];
        koch.extend(&mut clique, p, dd, s);
        if koch.stopped || koch.capped {
            break;
        }
    }
    if rings.is_some()
        && limits.known_score > 0
        && koch.scored.is_empty()
        && !koch.stopped
        && !koch.capped
    {
        // No part reached the known score: search again without it (`known_score`).
        let mut work = setup;
        work.add(koch.work());
        let again = FragmentLimits {
            known_score: 0,
            max_steps: limits.max_steps - koch.steps,
            max_work: limits
                .max_work
                .map(|max| max.saturating_sub(work.cost(&limits.costs))),
            ..limits
        };
        let mut out = fragments_impl(g1, g2, rings, again);
        out.steps += koch.steps;
        out.work.add(work);
        return out;
    }
    let mut out = Fragments {
        steps: koch.steps,
        stop: koch.stopped.then_some(if koch.timed {
            Stop::Time
        } else if koch.worked {
            Stop::Work
        } else {
            Stop::Steps
        }),
        truncated: koch.capped || koch.raw_full || koch.scored_full,
        split: koch.split,
        pairs: n,
        best_score: koch.best_score,
        work: setup,
        ..Fragments::default()
    };
    out.work.add(koch.work());
    // What follows the enumeration is counted too; nothing stops it.
    let mut kept: BTreeSet<Vec<usize>> = BTreeSet::new();
    if rings.is_some() {
        kept.extend(koch.scored.iter().map(|(part, _)| part.clone()));
    }
    for clique in &koch.raw {
        out.work.leaf_bonds += clique.len() as u64;
        let mut sorted = clique.clone();
        sorted.sort_unstable();
        if star_triple(&sorted, &vi, &vj, g1, g2, &tri1, &tri2).is_some() {
            out.split += 1;
        }
        kept.extend(consistent_parts(
            sorted,
            &vi,
            &vj,
            g1,
            g2,
            &tri1,
            &tri2,
            &c,
            koch.min_bonds,
        ));
    }
    // Split parts can lie inside another kept part; listing them would only crowd the cap.
    let kept: Vec<Vec<usize>> = kept.into_iter().collect();
    // Parts kept for their score stay even inside a larger part, whose trim may differ.
    let scored: BTreeSet<&Vec<usize>> = koch.scored.iter().map(|(part, _)| part).collect();
    let inside = |a: &Vec<usize>| {
        !scored.contains(a)
            && kept
                .iter()
                .any(|b| b.len() > a.len() && a.iter().all(|x| b.binary_search(x).is_ok()))
    };
    out.work.words += (kept.len() * kept.len()) as u64;
    let mut maximal: Vec<&Vec<usize>> = kept.iter().filter(|a| !inside(a)).collect();
    if limits.best_only {
        let largest = maximal.iter().map(|a| a.len()).max().unwrap_or(0);
        maximal.retain(|a| a.len() == largest || scored.contains(a));
    }
    for part in maximal {
        if out.fragments.len() >= limits.max_fragments {
            out.truncated = true;
            break;
        }
        out.work.leaf_bonds += part.len() as u64;
        let bonds: Vec<(usize, usize)> = part.iter().map(|&x| (vi[x], vj[x])).collect();
        match atom_map(&bonds, g1, g2) {
            Some(atoms) => out.fragments.push(Fragment { bonds, atoms }),
            None => out.truncated = true,
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(labels: &[i64], bonds: &[(usize, usize)]) -> Graph {
        Graph {
            labels: labels.to_vec(),
            bonds: bonds.to_vec(),
        }
    }

    fn ring(n: usize, label: i64) -> Graph {
        graph(
            &vec![label; n],
            &(0..n).map(|i| (i, (i + 1) % n)).collect::<Vec<_>>(),
        )
    }

    #[test]
    fn a_passed_deadline_stops_the_set_up_of_a_large_graph() {
        // 10,000 bond pairs: building the compatibility graph alone would take a while.
        let (g1, g2) = (ring(100, 6), ring(100, 6));
        let start = Instant::now();
        let out = maximal_fragments(
            &g1,
            &g2,
            FragmentLimits {
                deadline: Some(start),
                ..FragmentLimits::default()
            },
        );
        assert_eq!(
            (out.stop, out.steps, out.pairs),
            (Some(Stop::Time), 0, 10_000)
        );
        assert!(start.elapsed().as_millis() < 100);
    }

    // ---- maximal_fragments ----

    /// Every valid fragment by brute force: subsets of compatible bond pairs that are
    /// cliques, connected through shared atoms, consistent, with at least `min_bonds`.
    fn valid_subsets(g1: &Graph, g2: &Graph, min_bonds: usize) -> Vec<Vec<(usize, usize)>> {
        let (vi, vj) = compatible_pairs(g1, g2);
        let n = vi.len();
        assert!(n <= 20, "oracle limited to 20 bond pairs, got {n}");
        let (c, d) = build_relations(&vi, &vj, g1, g2, None).unwrap();
        let (tri1, tri2) = (g1.triangles(), g2.triangles());
        let mut out = Vec::new();
        for mask in 1u32..(1u32 << n) {
            let members: Vec<usize> = (0..n).filter(|&x| (mask >> x) & 1 == 1).collect();
            if members.len() < min_bonds {
                continue;
            }
            let clique = members.iter().all(|&x| {
                members
                    .iter()
                    .all(|&y| x == y || c.get(x, y) || d.get(x, y))
            });
            if !clique || c_components(&members, &c).len() != 1 {
                continue;
            }
            if star_triple(&members, &vi, &vj, g1, g2, &tri1, &tri2).is_some() {
                continue;
            }
            let mut bonds: Vec<(usize, usize)> = members.iter().map(|&x| (vi[x], vj[x])).collect();
            bonds.sort_unstable();
            out.push(bonds);
        }
        out
    }

    fn subset(a: &[(usize, usize)], b: &[(usize, usize)]) -> bool {
        a.iter().all(|x| b.contains(x))
    }

    fn enumerated(out: &Fragments) -> BTreeSet<Vec<(usize, usize)>> {
        out.fragments
            .iter()
            .map(|f| {
                let mut b = f.bonds.clone();
                b.sort_unstable();
                b
            })
            .collect()
    }

    #[test]
    fn a_full_best_only_list_does_not_end_the_search() {
        // the triangle of the second graph maps onto stars of the first in many maximal
        // cliques, whose consistent parts are small; with room for one fragment, the
        // search must still reach the four-bond one
        let cases = [(
            graph(
                &[6; 7],
                &[(0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (1, 3), (4, 5)],
            ),
            graph(&[6; 4], &[(0, 1), (0, 2), (1, 2), (2, 3)]),
        )];
        for (g1, g2) in cases {
            let unlimited = FragmentLimits {
                best_only: true,
                ..all_limits(1)
            };
            let largest = maximal_fragments(&g1, &g2, unlimited).fragments[0]
                .bonds
                .len();
            let one = maximal_fragments(
                &g1,
                &g2,
                FragmentLimits {
                    max_fragments: 1,
                    ..unlimited
                },
            );
            assert_eq!(one.fragments.len(), 1);
            assert_eq!(one.fragments[0].bonds.len(), largest);
            assert!(one.truncated);
            assert_eq!(one.stop, None);
        }
    }

    fn rings_of(g: &Graph, ring_atoms: &[usize]) -> Rings {
        let atoms: Vec<bool> = (0..g.labels.len())
            .map(|a| ring_atoms.contains(&a))
            .collect();
        let bonds = g.bonds.iter().map(|&(u, v)| atoms[u] && atoms[v]).collect();
        Rings { atoms, bonds }
    }

    #[test]
    fn the_trim_keeps_whole_rings_and_their_substituents_only() {
        // toluene: ring 0-5, methyl 6 on atom 0
        let toluene = graph(
            &[1, 1, 1, 1, 1, 1, 0],
            &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6)],
        );
        let rings = rings_of(&toluene, &[0, 1, 2, 3, 4, 5]);
        let partners = bond_index(&toluene);
        let identity: Vec<(usize, usize)> = (0..7).map(|a| (a, a)).collect();
        assert_eq!(
            trimmed_bonds(&identity, &toluene, &rings, &rings, &partners),
            7
        );
        // without atom 3 the ring is a path: every ring atom goes, the methyl stands alone
        let open: Vec<(usize, usize)> = identity.iter().copied().filter(|&(a, _)| a != 3).collect();
        assert_eq!(trimmed_bonds(&open, &toluene, &rings, &rings, &partners), 0);
    }

    #[test]
    fn ring_fragments_also_list_the_largest_fragments() {
        let toluene = graph(
            &[1, 1, 1, 1, 1, 1, 0],
            &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6)],
        );
        // a seven-membered ring with a methyl: the largest fragment threads the ring
        let other = graph(
            &[1, 1, 1, 1, 1, 1, 1, 0],
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 0),
                (0, 7),
            ],
        );
        let limits = FragmentLimits {
            best_only: true,
            ..all_limits(1)
        };
        let largest = maximal_fragments(&toluene, &other, limits);
        let r1 = rings_of(&toluene, &[0, 1, 2, 3, 4, 5]);
        let r2 = rings_of(&other, &[0, 1, 2, 3, 4, 5, 6]);
        let scored = ring_fragments(&toluene, &other, &r1, &r2, limits);
        let listed: BTreeSet<_> = scored.fragments.iter().map(|f| f.bonds.clone()).collect();
        assert!(!largest.fragments.is_empty());
        assert!(largest.fragments.iter().all(|f| listed.contains(&f.bonds)));
    }

    #[test]
    fn a_known_score_lists_the_same_fragments_in_fewer_steps() {
        // naphthalene against indane-like fused rings with chains: many fragments, and
        // a known score (here the best one) cuts branches without changing the list
        let g1 = graph(
            &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 0),
                (4, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 5),
                (0, 10),
                (10, 11),
                (8, 12),
            ],
        );
        let g2 = graph(
            &[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 0),
                (4, 6),
                (6, 7),
                (7, 8),
                (8, 5),
                (0, 9),
                (9, 10),
                (7, 11),
            ],
        );
        let r1 = rings_of(&g1, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let r2 = rings_of(&g2, &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
        let plain = ring_fragments(&g1, &g2, &r1, &r2, all_limits(1));
        assert!(plain.stop.is_none() && plain.best_score > 0);
        for known in [1, plain.best_score / 2, plain.best_score] {
            let limits = FragmentLimits {
                known_score: known,
                ..all_limits(1)
            };
            let out = ring_fragments(&g1, &g2, &r1, &r2, limits);
            assert_eq!(out.fragments, plain.fragments);
            assert_eq!(out.best_score, plain.best_score);
            assert!(out.steps <= plain.steps);
        }
        // A known score no part of this search reaches (possible when it came from finer
        // labels, since the trim is not monotone) makes the search run again from zero.
        let limits = FragmentLimits {
            known_score: plain.best_score + 1,
            ..all_limits(1)
        };
        let out = ring_fragments(&g1, &g2, &r1, &r2, limits);
        assert_eq!(out.fragments, plain.fragments);
        assert_eq!(out.best_score, plain.best_score);
        assert!(out.steps > plain.steps);
    }

    fn all_limits(min_bonds: usize) -> FragmentLimits {
        FragmentLimits {
            min_bonds,
            ..FragmentLimits::default()
        }
    }

    #[test]
    fn fragments_match_the_brute_force_maximal_set() {
        let cases = [
            (
                graph(&[6, 6, 7, 6, 6], &[(0, 1), (1, 2), (2, 3), (3, 4)]),
                graph(&[6, 6, 6, 7, 6], &[(0, 1), (1, 2), (2, 3), (3, 4)]),
            ),
            (
                ring(5, 6),
                graph(&[6; 5], &[(0, 1), (1, 2), (2, 3), (3, 4)]),
            ),
            (
                graph(&[6, 6, 6, 8, 6], &[(0, 1), (1, 2), (2, 3), (3, 4)]),
                graph(&[6, 6, 6, 7, 6], &[(0, 1), (1, 2), (2, 3), (3, 4)]),
            ),
            (
                graph(&[6, 6, 6, 8], &[(0, 1), (0, 2), (0, 3)]),
                graph(&[6, 6, 8, 8], &[(0, 1), (0, 2), (0, 3)]),
            ),
        ];
        for (g1, g2) in &cases {
            let out = maximal_fragments(g1, g2, all_limits(1));
            assert!(out.stop.is_none() && !out.truncated && out.split == 0);
            let valid = valid_subsets(g1, g2, 1);
            let maximal: BTreeSet<Vec<(usize, usize)>> = valid
                .iter()
                .filter(|a| !valid.iter().any(|b| b.len() > a.len() && subset(a, b)))
                .cloned()
                .collect();
            assert_eq!(enumerated(&out), maximal);
            for f in &out.fragments {
                assert!(f.atoms.iter().all(|&(a, b)| g1.labels[a] == g2.labels[b]));
            }
        }
    }

    #[test]
    fn every_valid_fragment_is_covered() {
        let cases = [
            (
                graph(&[6, 6, 6], &[(0, 1), (1, 2), (2, 0)]),
                graph(&[6, 6, 6, 6], &[(0, 1), (0, 2), (0, 3)]),
            ),
            (
                graph(&[6, 6, 6, 6], &[(0, 1), (1, 2), (2, 0), (2, 3)]),
                graph(&[6, 6, 6, 6, 6], &[(0, 1), (0, 2), (0, 3), (1, 4)]),
            ),
        ];
        for (g1, g2) in &cases {
            let out = maximal_fragments(g1, g2, all_limits(1));
            assert!(out.stop.is_none() && !out.truncated);
            assert!(out.split > 0);
            let valid = valid_subsets(g1, g2, 1);
            let listed = enumerated(&out);
            for v in &valid {
                assert!(listed.iter().any(|f| subset(v, f)), "{v:?} not covered");
            }
            for f in &listed {
                assert!(valid.contains(f), "{f:?} is not a valid fragment");
            }
        }
    }

    #[test]
    fn the_floor_drops_small_fragments() {
        let none = maximal_fragments(&ring(6, 6), &ring(6, 6), all_limits(7));
        assert!(none.fragments.is_empty() && none.stop.is_none());
        let out = maximal_fragments(&ring(6, 6), &ring(6, 6), all_limits(6));
        assert_eq!(out.fragments.len(), 12);
        assert!(out
            .fragments
            .iter()
            .all(|f| f.bonds.len() == 6 && f.atoms.len() == 6));
    }

    #[test]
    fn a_stopped_enumeration_is_reproducible() {
        let limits = FragmentLimits {
            max_steps: 50,
            ..all_limits(3)
        };
        let a = maximal_fragments(&ring(12, 6), &ring(12, 6), limits);
        let b = maximal_fragments(&ring(12, 6), &ring(12, 6), limits);
        assert_eq!(a.stop, Some(Stop::Steps));
        assert_eq!(a.steps, 50);
        assert_eq!(a.fragments, b.fragments);
    }

    #[test]
    fn a_passed_deadline_stops_the_enumeration_at_the_first_clock_reading() {
        let (g1, g2) = (ring(30, 6), ring(30, 6));
        let full = maximal_fragments(&g1, &g2, all_limits(3));
        assert!(full.stop.is_none() && full.steps > CLOCK_EVERY);
        let past = FragmentLimits {
            deadline: Some(Instant::now()),
            ..all_limits(3)
        };
        let out = maximal_fragments(&g1, &g2, past);
        assert_eq!(out.stop, Some(Stop::Time));
        assert_eq!(out.steps, CLOCK_EVERY);
        let far = FragmentLimits {
            deadline: Some(Instant::now() + std::time::Duration::from_secs(3600)),
            ..all_limits(3)
        };
        let out = maximal_fragments(&g1, &g2, far);
        assert_eq!((out.stop, out.steps), (None, full.steps));
        assert_eq!(out.fragments, full.fragments);
    }

    #[test]
    fn the_work_limit_stops_the_enumeration_at_the_same_step_every_time() {
        let (g1, g2) = (ring(30, 6), ring(30, 6));
        let costs = WorkCosts {
            relation_pair: 3,
            step: 7,
            word: 2,
            leaf_bond: 5,
        };
        let free = FragmentLimits {
            costs,
            ..all_limits(3)
        };
        let full = maximal_fragments(&g1, &g2, free);
        // Without a limit the work is counted and stops nothing.
        assert!(full.stop.is_none());
        assert_eq!(full.work.relation_pairs, 900 * 899 / 2);
        assert_eq!(full.work.steps, full.steps);
        assert!(full.work.words > 0 && full.work.leaf_bonds > 0);
        let half = FragmentLimits {
            max_work: Some(full.work.cost(&costs) / 2),
            ..free
        };
        let out = maximal_fragments(&g1, &g2, half);
        assert_eq!(out.stop, Some(Stop::Work));
        assert!(out.steps > 0 && out.steps < full.steps);
        // The stop reads no clock: the same steps, work and fragments again.
        let again = maximal_fragments(&g1, &g2, half);
        assert_eq!((again.stop, again.steps), (out.stop, out.steps));
        assert_eq!(again.work, out.work);
        assert_eq!(again.fragments, out.fragments);
        // A limit the search does not reach changes nothing.
        let enough = FragmentLimits {
            max_work: Some(full.work.cost(&costs)),
            ..free
        };
        let out = maximal_fragments(&g1, &g2, enough);
        assert_eq!((out.stop, out.steps), (None, full.steps));
        assert_eq!(out.fragments, full.fragments);
    }

    #[test]
    fn a_set_up_that_costs_more_than_the_work_limit_is_not_done() {
        // 10,000 bond pairs: the set-up reads 49,995,000 relations, known beforehand.
        let (g1, g2) = (ring(100, 6), ring(100, 6));
        let limits = FragmentLimits {
            max_work: Some(49_994_999),
            costs: WorkCosts {
                relation_pair: 1,
                ..WorkCosts::default()
            },
            ..FragmentLimits::default()
        };
        let start = Instant::now();
        let out = maximal_fragments(&g1, &g2, limits);
        assert_eq!(
            (out.stop, out.steps, out.pairs),
            (Some(Stop::Work), 0, 10_000)
        );
        assert_eq!(out.work.relation_pairs, 49_995_000);
        assert_eq!((out.work.words, out.work.leaf_bonds), (0, 0));
        assert!(start.elapsed().as_millis() < 100);
        // One unit more and the set-up is done: the search starts.
        let out = maximal_fragments(
            &g1,
            &g2,
            FragmentLimits {
                max_work: Some(49_995_000),
                max_steps: 10,
                ..limits
            },
        );
        assert_eq!((out.stop, out.steps), (Some(Stop::Steps), 10));
    }

    #[test]
    fn the_fragment_cap_truncates() {
        let limits = FragmentLimits {
            max_fragments: 5,
            ..all_limits(6)
        };
        let out = maximal_fragments(&ring(6, 6), &ring(6, 6), limits);
        assert_eq!(out.fragments.len(), 5);
        assert!(out.truncated);
    }

    #[test]
    fn too_many_bond_pairs_stop_before_searching() {
        let limits = FragmentLimits {
            max_pairs: 10,
            ..all_limits(3)
        };
        let out = maximal_fragments(&ring(6, 6), &ring(6, 6), limits);
        assert_eq!(out.stop, Some(Stop::Pairs));
        assert_eq!(out.pairs, 36);
        assert!(out.fragments.is_empty());
    }

    #[test]
    fn swapping_the_graphs_mirrors_the_fragments() {
        let g1 = graph(&[6, 6, 6, 8, 6], &[(0, 1), (1, 2), (2, 3), (1, 4)]);
        let g2 = graph(
            &[6, 6, 8, 6, 6, 6],
            &[(0, 1), (1, 2), (0, 3), (3, 4), (4, 5)],
        );
        let a = maximal_fragments(&g1, &g2, all_limits(1));
        let b = maximal_fragments(&g2, &g1, all_limits(1));
        let mirror: BTreeSet<Vec<(usize, usize)>> = b
            .fragments
            .iter()
            .map(|c| {
                let mut atoms: Vec<_> = c.atoms.iter().map(|&(x, y)| (y, x)).collect();
                atoms.sort_unstable();
                atoms
            })
            .collect();
        let direct: BTreeSet<_> = a.fragments.iter().map(|c| c.atoms.clone()).collect();
        assert_eq!(direct, mirror);
    }

    #[test]
    fn split_parts_inside_another_fragment_are_not_listed() {
        // K4 against K4: every maximal clique holds triangles mapped onto stars, and
        // splitting them yields parts that lie inside other parts.
        let k4 = graph(&[6; 4], &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
        let out = maximal_fragments(&k4, &k4, all_limits(3));
        assert!(out.split > 0);
        let listed = enumerated(&out);
        for a in &listed {
            assert!(
                !listed.iter().any(|b| b.len() > a.len() && subset(a, b)),
                "{a:?} lies inside another listed fragment"
            );
        }
    }

    #[test]
    fn best_only_keeps_the_largest_connected_fragments() {
        // 2,3-dimethylbutane against n-hexane: the largest connected common fragment is
        // a 4-atom chain (3 bonds), though two 2-bond pieces would give a 4-bond
        // disconnected one
        let dmb = graph(&[6; 6], &[(0, 1), (0, 2), (0, 3), (3, 4), (3, 5)]);
        let hexane = graph(&[6; 6], &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]);
        let limits = FragmentLimits {
            best_only: true,
            ..all_limits(1)
        };
        let out = maximal_fragments(&dmb, &hexane, limits);
        assert!(!out.fragments.is_empty());
        assert!(out.fragments.iter().all(|f| f.bonds.len() == 3));
    }

    #[test]
    fn best_only_lists_exactly_the_largest_of_the_full_enumeration() {
        let cases = [
            (ring(6, 6), ring(5, 6)),
            (
                graph(
                    &[6, 6, 7, 6, 6, 8],
                    &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (2, 5)],
                ),
                graph(
                    &[6, 6, 6, 7, 6, 8, 6],
                    &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (3, 5), (5, 6)],
                ),
            ),
            (
                graph(&[6; 4], &[(0, 1), (1, 2), (2, 0), (0, 3)]),
                graph(&[6; 4], &[(0, 1), (0, 2), (0, 3), (1, 2)]),
            ),
        ];
        for (g1, g2) in cases {
            let all = maximal_fragments(&g1, &g2, all_limits(1));
            let largest = all.fragments.iter().map(|f| f.bonds.len()).max().unwrap();
            let mut expected: Vec<_> = all
                .fragments
                .iter()
                .filter(|f| f.bonds.len() == largest)
                .map(|f| f.bonds.clone())
                .collect();
            let best = maximal_fragments(
                &g1,
                &g2,
                FragmentLimits {
                    best_only: true,
                    ..all_limits(1)
                },
            );
            let mut got: Vec<_> = best.fragments.iter().map(|f| f.bonds.clone()).collect();
            expected.sort();
            got.sort();
            assert_eq!(got, expected);
            assert!(best.steps <= all.steps);
        }
    }
}
