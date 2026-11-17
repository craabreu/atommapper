//! AtomMapper's fragment search engine: step-budgeted connected common fragment search.
//!
//! The engine is plain Rust (`engine`); the Python binding below, built by maturin as
//! `atommapper._mces`, only converts arguments and results.

pub mod engine;

#[cfg(feature = "python")]
mod python {
    use crate::engine::{maximal_fragments, ring_fragments, FragmentLimits, Graph, Rings, Stop};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use std::time::{Duration, Instant};

    type Fragment = (Vec<(usize, usize)>, Vec<usize>);

    fn checked(labels: Vec<i64>, bonds: Vec<(usize, usize)>, which: &str) -> PyResult<Graph> {
        let mut seen = std::collections::HashSet::new();
        for &(a, b) in &bonds {
            if a >= labels.len() || b >= labels.len() || a == b {
                return Err(PyValueError::new_err(format!(
                    "graph {which}: bond ({a}, {b}) is out of range or a self-loop"
                )));
            }
            if !seen.insert((a.min(b), a.max(b))) {
                return Err(PyValueError::new_err(format!(
                    "graph {which}: bond ({a}, {b}) is given twice"
                )));
            }
        }
        Ok(Graph { labels, bonds })
    }

    /// The deadline `seconds` from now; `None` for no limit or one too far to represent.
    fn deadline(seconds: Option<f64>) -> PyResult<Option<Instant>> {
        let Some(seconds) = seconds else {
            return Ok(None);
        };
        if seconds.is_nan() {
            return Err(PyValueError::new_err("time_limit_s must be a number"));
        }
        Ok(Duration::try_from_secs_f64(seconds.max(0.0))
            .ok()
            .and_then(|d| Instant::now().checked_add(d)))
    }

    fn stop_name(stop: Stop) -> &'static str {
        match stop {
            Stop::Steps => "steps",
            Stop::Pairs => "pairs",
            Stop::Time => "time",
        }
    }

    type FragmentsOut = (
        Vec<Fragment>,
        u64,
        Option<&'static str>,
        bool,
        usize,
        usize,
        usize,
    );
    type RingFlags = (Vec<bool>, Vec<bool>, Vec<bool>, Vec<bool>);

    /// Every maximal connected common edge subgraph with at least ``min_bonds`` bonds,
    /// within a step budget (Koch's enumeration); with ``best_only``, only the largest
    /// ones (the maximum connected common edge subgraphs, every tie).
    ///
    /// Returns ``(fragments, steps, stop, truncated, split, pairs, best_score)``. Each
    /// fragment is ``(atoms, bonds)``: the atom pairs it induces, sorted by graph-1 atom,
    /// and the graph-1 bond indices it matches. ``stop`` is ``None``, ``"steps"``,
    /// ``"pairs"`` (more compatible bond pairs than ``max_pairs``; nothing was searched)
    /// or ``"time"`` (``time_limit_s`` seconds passed, read every ``CLOCK_EVERY`` steps,
    /// more often on large search graphs and while those are built; without it the clock
    /// is never read). Labels are any integers; bonds must be distinct and within range.
    /// ``truncated`` means the list may not cover every common fragment (the cap was
    /// reached); ``split`` counts maximal cliques that mapped a triangle onto a
    /// three-armed star and were replaced by their consistent connected parts.
    ///
    /// With ``rings`` (ring flags of graph 1's atoms and bonds, then graph 2's), the
    /// largest fragments and those whose largest piece keeps the most bonds once trimmed
    /// to complete rings are listed, every tie (``engine::ring_fragments``); ``best_only``
    /// is implied; ``known_score`` is a trimmed score some common fragment is known to
    /// reach (``FragmentLimits::known_score``), and ``best_score`` the best found.
    #[pyfunction]
    #[pyo3(signature = (labels1, bonds1, labels2, bonds2, *, max_steps, max_fragments, min_bonds, max_pairs=20_000, best_only=false, rings=None, time_limit_s=None, known_score=0))]
    #[allow(clippy::too_many_arguments)]
    fn fragments(
        py: Python<'_>,
        labels1: Vec<i64>,
        bonds1: Vec<(usize, usize)>,
        labels2: Vec<i64>,
        bonds2: Vec<(usize, usize)>,
        max_steps: u64,
        max_fragments: usize,
        min_bonds: usize,
        max_pairs: usize,
        best_only: bool,
        rings: Option<RingFlags>,
        time_limit_s: Option<f64>,
        known_score: usize,
    ) -> PyResult<FragmentsOut> {
        if max_fragments == 0 || min_bonds == 0 {
            return Err(PyValueError::new_err(
                "max_fragments and min_bonds must be at least 1",
            ));
        }
        let g1 = checked(labels1, bonds1, "1")?;
        let g2 = checked(labels2, bonds2, "2")?;
        let limits = FragmentLimits {
            max_steps,
            max_fragments,
            max_pairs,
            min_bonds,
            best_only,
            deadline: deadline(time_limit_s)?,
            known_score,
        };
        let out = match rings {
            None => py.allow_threads(|| maximal_fragments(&g1, &g2, limits)),
            Some((atoms1, ring_bonds1, atoms2, ring_bonds2)) => {
                if atoms1.len() != g1.labels.len()
                    || ring_bonds1.len() != g1.bonds.len()
                    || atoms2.len() != g2.labels.len()
                    || ring_bonds2.len() != g2.bonds.len()
                {
                    return Err(PyValueError::new_err(
                        "rings must flag every atom and bond of both graphs",
                    ));
                }
                let r1 = Rings {
                    atoms: atoms1,
                    bonds: ring_bonds1,
                };
                let r2 = Rings {
                    atoms: atoms2,
                    bonds: ring_bonds2,
                };
                py.allow_threads(|| ring_fragments(&g1, &g2, &r1, &r2, limits))
            }
        };
        let fragments = out
            .fragments
            .into_iter()
            .map(|c| (c.atoms, c.bonds.into_iter().map(|(i, _)| i).collect()))
            .collect();
        let stop = out.stop.map(stop_name);
        Ok((
            fragments,
            out.steps,
            stop,
            out.truncated,
            out.split,
            out.pairs,
            out.best_score,
        ))
    }

    #[pymodule]
    fn _mces(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(fragments, m)?)?;
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }
}
