"""
Vertex-subsampling augmentation for robustness!
approximate decimated / non-uniform / holed meshes
"""
import torch


@torch.no_grad()
def sample_vertex_indices(
        template,                    # (N, 3) neutral vertices of the batch identity
        aug_prob=0.7,                # prob. of augmenting at all (else full mesh)
        min_ratio=0.1,               # [sparse] keep ratio ~ U(min_ratio, 1)
        min_points=256,              # never keep fewer than this..!
        hole_prob=0.5,               # [hole]: prob. of cutting holes
        max_holes=3,                 # [hole]: k ~ U{1..max_holes}
        hole_radius=(0.02, 0.08),    # [hole]: r ~ U(range) * bbox diagonal
        density_prob=0.5,            # [non-uniform]: prob. of non-uniform density
        num_rbf=4,                   # [non-uniform]: number of RBF bumps
        rbf_sigma=(0.05, 0.2),       # [non-uniform]: sigma ~ U(range) * bbox diagonal
        rbf_amp=2.0,                 # [non-uniform]: amplitude ~ U(-amp, amp) in log-density
        generator=None,
    ):
    """
    Returns (idx, augmented). idx is a 1D LongTensor of kept vertex indices
    in random order (on template.device). If not augmented, idx = arange(N)
    (original order, so mesh-topology losses such as Laplacian stay valid).
    """
    device = template.device
    N = template.shape[0]

    def rand(*shape):
        return torch.rand(*shape, device=device, generator=generator)

    if rand(1).item() >= aug_prob:
        return torch.arange(N, device=device), False

    diag = (template.max(0).values - template.min(0).values).norm()
    keep = torch.ones(N, dtype=torch.bool, device=device)

    # holes ---------------------------------------------------------------
    if rand(1).item() < hole_prob:
        k = int(torch.randint(1, max_holes + 1, (1,), device=device, generator=generator).item())
        
        centers = template[torch.randint(0, N, (k,), device=device, generator=generator)]
        radii = (hole_radius[0] + (hole_radius[1] - hole_radius[0]) * rand(k)) * diag
        dist = torch.cdist(template, centers)                # (N, k)
        hole = (dist < radii[None]).any(-1)
        
        # do not let holes eat the mesh below min_points
        if (keep & ~hole).sum() >= min_points:
            keep &= ~hole

    cand = keep.nonzero(as_tuple=True)[0] # remaining vertices
    M = cand.numel()

    # non-uniform sampling probability --------------------------------------
    if rand(1).item() < density_prob:
        pts = template[cand]
        centers = pts[torch.randint(0, M, (num_rbf,), device=device, generator=generator)]
        sigma = (rbf_sigma[0] + (rbf_sigma[1] - rbf_sigma[0]) * rand(num_rbf)) * diag
        amp = (2 * rand(num_rbf) - 1) * rbf_amp
        d2 = torch.cdist(pts, centers).square()               # (M, num_rbf)
        
        log_p = (amp[None] * torch.exp(-d2 / (2 * sigma[None] ** 2))).sum(-1)
        prob = torch.exp(log_p - log_p.max())
    else:
        prob = torch.ones(M, device=device)

    # sparse keep ratio ----------------------------------------------------
    ratio = min_ratio + (1 - min_ratio) * rand(1).item()
    n_keep = min(M, max(min_points, int(round(ratio * M))))
    
    sel = torch.multinomial(prob, n_keep, replacement=False, generator=generator)
    return cand[sel], True
