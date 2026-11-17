"""
Mesh topology perturbations for the robustness evaluation (eval_CBD_robust.py).

Every perturbation is represented as a sparse linear operator P [V', V] plus a
new face array F' [F', 3]:

    perturbed_vertices = P @ original_vertices

Because the same P applies to the neutral mesh and to every expression frame of
that identity, the ground truth on the perturbed mesh is exactly P @ GT.

    hole      : P selects the vertices that survive after removing k-ring
                patches around seed vertices (rows are one-hot).
    decimate  : igl.decimate (edge collapse), then every output vertex is
                snapped to the original vertex it was born from, so P is again
                a one-hot selection.
    subdiv    : midpoint subdivision without smoothing (each triangle -> 4).
                Rows [0, V) are the original vertices (identity), rows [V, V+E)
                are edge midpoints (0.5 / 0.5).
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import igl


@dataclass
class Perturbation:
    name: str
    P: sp.csr_matrix              # [V', V]
    faces: np.ndarray             # [F', 3] int64
    orig_rows: np.ndarray = None  # rows of P that are exact original vertices (subdiv only)
    info: dict = field(default_factory=dict)

    @property
    def n_vert(self):
        return self.P.shape[0]


def identity_perturbation(n_vert, faces):
    P = sp.identity(n_vert, format='csr', dtype=np.float32)
    return Perturbation('clean', P, np.asarray(faces, dtype=np.int64))


def _selection(keep_idx, n_vert):
    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    rows = np.arange(len(keep_idx))
    return sp.csr_matrix(
        (np.ones(len(keep_idx), dtype=np.float32), (rows, keep_idx)),
        shape=(len(keep_idx), n_vert),
    )


def _compact(faces, keep_mask, n_vert):
    """Keep faces whose 3 vertices all survive, drop unreferenced vertices,
    re-index faces. Returns (keep_idx, new_faces)."""
    faces = np.asarray(faces, dtype=np.int64)
    f_keep = keep_mask[faces].all(axis=1)
    faces = faces[f_keep]
    used = np.zeros(n_vert, dtype=bool)
    used[faces.reshape(-1)] = True
    keep_idx = np.nonzero(used)[0]
    remap = -np.ones(n_vert, dtype=np.int64)
    remap[keep_idx] = np.arange(len(keep_idx))
    return keep_idx, remap[faces]


# ---------------------------------------------------------------------------
# hole
# ---------------------------------------------------------------------------
def make_hole(n_vert, faces, candidate_idx, n_holes, frac_per_hole, seed, name='hole'):
    """
    Remove `n_holes` patches. Each patch is grown by BFS rings (k-ring) from a
    random seed vertex in `candidate_idx` until it holds
    frac_per_hole * len(candidate_idx) vertices. Seed vertices are drawn
    without replacement from candidates that are not already removed.

    The patch depends only on topology + candidate set + seed, so two
    identities that share a topology get the identical hole.
    """
    faces = np.asarray(faces, dtype=np.int64)
    adj = igl.adjacency_list(faces)
    rng = np.random.RandomState(seed)
    target = max(1, int(round(frac_per_hole * len(candidate_idx))))
    removed = np.zeros(n_vert, dtype=bool)
    centers = []
    cand = np.asarray(candidate_idx)
    for _ in range(n_holes):
        free = cand[~removed[cand]]
        c = int(rng.choice(free))
        centers.append(c)
        patch = {c}
        frontier = [c]
        while len(patch) < target and frontier:
            nxt = []
            for v in frontier:
                for u in adj[v]:
                    if u not in patch:
                        patch.add(u)
                        nxt.append(u)
            frontier = nxt
        removed[list(patch)] = True
    keep_idx, new_faces = _compact(faces, ~removed, n_vert)
    info = dict(centers=centers, n_removed=int(n_vert - len(keep_idx)),
                n_boundary_loops=len(igl.all_boundary_loop(new_faces)))
    return Perturbation(name, _selection(keep_idx, n_vert), new_faces, info=info)


# ---------------------------------------------------------------------------
# decimate
# ---------------------------------------------------------------------------
def make_decimate(verts, faces, face_ratio, name='decimate'):
    """igl.decimate to face_ratio * |F| faces, then snap every output vertex
    to its birth vertex (I) so that the result is an exact vertex subset."""
    V = np.asarray(verts, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    ok, U, G, J, I = igl.decimate(V, F, int(F.shape[0] * face_ratio))
    if not ok:
        raise RuntimeError(f'igl.decimate failed (ratio={face_ratio})')
    I = np.asarray(I, dtype=np.int64)
    G = np.asarray(G, dtype=np.int64)
    # several output vertices could in principle share a birth vertex: merge them
    uniq, inv = np.unique(I, return_inverse=True)
    G = inv[G]
    nondegen = (G[:, 0] != G[:, 1]) & (G[:, 1] != G[:, 2]) & (G[:, 0] != G[:, 2])
    G = G[nondegen]
    used = np.zeros(len(uniq), dtype=bool)
    used[G.reshape(-1)] = True
    keep_local = np.nonzero(used)[0]
    remap = -np.ones(len(uniq), dtype=np.int64)
    remap[keep_local] = np.arange(len(keep_local))
    G = remap[G]
    keep_idx = uniq[keep_local]
    info = dict(n_merged=int(len(I) - len(uniq)), n_degenerate=int((~nondegen).sum()),
                snap_dist_max=float(np.linalg.norm(U - V[I], axis=1).max()))
    return Perturbation(name, _selection(keep_idx, V.shape[0]), G, info=info)


# ---------------------------------------------------------------------------
# subdivision
# ---------------------------------------------------------------------------
def make_subdiv(n_vert, faces, name='subdiv'):
    F = np.asarray(faces, dtype=np.int64)
    he = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)  # [3F, 2]
    he_sorted = np.sort(he, axis=1)
    edges, e_of_he = np.unique(he_sorted, axis=0, return_inverse=True)
    e_of_he = e_of_he.reshape(-1)
    nF = F.shape[0]
    m01 = n_vert + e_of_he[:nF]
    m12 = n_vert + e_of_he[nF:2 * nF]
    m20 = n_vert + e_of_he[2 * nF:]
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    new_faces = np.concatenate([
        np.stack([a, m01, m20], 1),
        np.stack([m01, b, m12], 1),
        np.stack([m20, m12, c], 1),
        np.stack([m01, m12, m20], 1),
    ], axis=0)
    nE = edges.shape[0]
    rows = np.concatenate([np.arange(n_vert), n_vert + np.arange(nE), n_vert + np.arange(nE)])
    cols = np.concatenate([np.arange(n_vert), edges[:, 0], edges[:, 1]])
    vals = np.concatenate([np.ones(n_vert), np.full(nE, 0.5), np.full(nE, 0.5)]).astype(np.float32)
    P = sp.csr_matrix((vals, (rows, cols)), shape=(n_vert + nE, n_vert))
    return Perturbation(name, P, new_faces, orig_rows=np.arange(n_vert))


# ---------------------------------------------------------------------------
# condition registry
# ---------------------------------------------------------------------------
# hole sizes are fractions of the face-region (inner mask) vertex count
CONDITIONS = {
    'hole_s':  dict(kind='hole', n_holes=1, frac=0.02, seed=0),
    'hole_l':  dict(kind='hole', n_holes=1, frac=0.10, seed=0),
    'hole_m':  dict(kind='hole', n_holes=5, frac=0.02, seed=1),
    'dec50':   dict(kind='decimate', ratio=0.50),
    'dec25':   dict(kind='decimate', ratio=0.25),
    'subdiv1': dict(kind='subdiv'),
}
CONDITION_ORDER = ['clean'] + list(CONDITIONS.keys())


def build_perturbation(cond, verts, faces, hole_candidates=None):
    """
    Args:
        cond: key of CONDITIONS or 'clean'
        verts: [V, 3] neutral vertices of this identity (used by decimate)
        faces: [F, 3]
        hole_candidates: vertex indices allowed as hole centers / counted for
            the hole size (face region). Shared across identities of one
            dataset so the hole is identical for the same topology.
    """
    n_vert = len(verts)
    if cond == 'clean':
        return identity_perturbation(n_vert, faces)
    spec = CONDITIONS[cond]
    if spec['kind'] == 'hole':
        return make_hole(n_vert, faces, hole_candidates, spec['n_holes'], spec['frac'],
                         spec['seed'], name=cond)
    if spec['kind'] == 'decimate':
        return make_decimate(verts, faces, spec['ratio'], name=cond)
    if spec['kind'] == 'subdiv':
        return make_subdiv(n_vert, faces, name=cond)
    raise KeyError(cond)
