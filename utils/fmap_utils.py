import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def calculate_geodesic_error(dist_x, corr_x, corr_y, p2p, return_mean=True):
    """
    Calculate the geodesic error between predicted correspondence and gt correspondence

    Args:
        dist_x (np.ndarray): Geodesic distance matrix of shape x. shape [Vx, Vx]
        corr_x (np.ndarray): Ground truth correspondences of shape x. shape [V]
        corr_y (np.ndarray): Ground truth correspondences of shape y. shape [V]
        p2p (np.ndarray): Point-to-point map (shape y -> shape x). shape [Vy]
        return_mean (bool, optional): Average the geodesic error. Default True.
    Returns:
        avg_geodesic_error (np.ndarray): Average geodesic error.
    """
    ind21 = np.stack([corr_x, p2p[corr_y]], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
    geo_err = np.take(dist_x, ind21)
    if return_mean:
        return geo_err.mean()
    else:
        return geo_err
    
def cdot(X, Y, dim):
    assert X.dim() == Y.dim()
    return torch.sum(torch.mul(X, Y), dim=dim)

class DirichletLoss(nn.Module):
    def __init__(self, normalize=False, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.normalize = normalize

    def forward(self, feats, L):
        assert feats.dim() == 3

        if self.normalize:
            feats = F.normalize(feats, p=2, dim=-1)

        de = cdot(feats, torch.bmm(L, feats), dim=1)
        loss = torch.mean(de)

        return self.loss_weight * loss
    
class SquaredFrobeniusLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, a, b):
        loss = torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1))
        return self.loss_weight * torch.mean(loss)

class SURFMNetLoss(nn.Module):
    """
    Loss as presented in the SURFMNet paper.
    Orthogonality + Bijectivity + Laplacian Commutativity
    """

    def __init__(self, w_bij=1.0, w_orth=1.0, w_lap=1e-3):
        """
        Init SURFMNetLoss

        Args:
            w_bij (float, optional): Bijectivity penalty weight. Default 1e3.
            w_orth (float, optional): Orthogonality penalty weight. Default 1e3.
            w_lap (float, optional): Laplacian commutativity penalty weight. Default 1.0.
        """
        super(SURFMNetLoss, self).__init__()
        assert w_bij >= 0 and w_orth >= 0 and w_lap >= 0
        self.w_bij = w_bij
        self.w_orth = w_orth
        self.w_lap = w_lap

    def forward(self, C12, C21, evals_1, evals_2):
        """
        Compute bijectivity loss + orthogonality loss
                            + Laplacian commutativity loss
                            + descriptor preservation via commutativity loss

        Args:
            C12 (torch.Tensor): matrix representation of functional map (1->2). Shape: [N, K, K]
            C21 (torch.Tensor): matrix representation of functional map (2->1). Shape: [N, K, K]
            evals_1 (torch.Tensor): eigenvalues of shape 1. Shape [N, K]
            evals_2 (torch.Tensor): eigenvalues of shape 2. Shape [N, K]
        """
        criterion = SquaredFrobeniusLoss()
        eye = torch.eye(C12.shape[1], C12.shape[2], device=C12.device).unsqueeze(0)
        eye_batch = torch.repeat_interleave(eye, repeats=C12.shape[0], dim=0)

        # losses = dict()
        # Bijectivity penalty
        if self.w_bij > 0:
            bijectivity_loss = criterion(torch.bmm(C12, C21), eye_batch) + criterion(torch.bmm(C21, C12), eye_batch)
            bijectivity_loss *= self.w_bij
            # losses['l_bij'] = bijectivity_loss
            l_bij = bijectivity_loss

        # Orthogonality penalty
        if self.w_orth > 0:
            orthogonality_loss = criterion(torch.bmm(C12.transpose(1, 2), C12), eye_batch) + \
                                 criterion(torch.bmm(C21.transpose(1, 2), C21), eye_batch)
            orthogonality_loss *= self.w_orth
            # losses['l_orth'] = orthogonality_loss
            l_orth = orthogonality_loss

        # Laplacian commutativity penalty
        if self.w_lap > 0:
            laplacian_loss = criterion(torch.einsum('abc,ac->abc', C12, evals_1),
                                       torch.einsum('ab,abc->abc', evals_2, C12))
            laplacian_loss += criterion(torch.einsum('abc,ac->abc', C21, evals_2),
                                        torch.einsum('ab,abc->abc', evals_1, C21))
            laplacian_loss *= self.w_lap
            # losses['l_lap'] = laplacian_loss
            l_lap = laplacian_loss

        # return losses
        return l_bij, l_orth, l_lap


class PartialFmapsLoss(nn.Module):
    def __init__(self, w_bij=1.0, w_orth=1.0):
        """
        Init PartialFmapsLoss
        Args:
            w_bij (float, optional): Bijectivity penalty weight. Default 1.0.
            w_orth (float, optional): Orthogonality penalty weight. Default 1.0.
        """
        super(PartialFmapsLoss, self).__init__()
        assert w_bij >= 0 and w_orth >= 0, 'Loss weight should be non-negative.'
        self.w_bij = w_bij
        self.w_orth = w_orth

    def forward(self, C_fp, C_pf, evals_full, evals_partial):
        assert C_fp.shape[0] == 1, 'Currently, only support batch size = 1'
        criterion = SquaredFrobeniusLoss()
        C_fp, C_pf = C_fp[0], C_pf[0]
        evals_full, evals_partial = evals_full[0], evals_partial[0]

        # compute area ratio between full shape and partial shape r
        r = min((evals_partial < evals_full.max()).sum(), C_fp.shape[0] - 1)
        eye = torch.zeros_like(C_fp)
        eye[torch.arange(0, r + 1), torch.arange(0, r + 1)] = 1.0

        if self.w_bij > 0:
            bijectivity_loss = self.w_bij * criterion(torch.matmul(C_fp, C_pf), eye)
        else:
            bijectivity_loss = 0.0

        if self.w_orth > 0:
            orthogonality_loss = self.w_bij * criterion(torch.matmul(C_fp, C_fp.t()), eye)
        else:
            orthogonality_loss = 0.0

        return {'l_bij': bijectivity_loss, 'l_orth': orthogonality_loss}
    

class Similarity(nn.Module):
    def __init__(self, normalise_dim=-1, tau=0.2, hard=False):
        super(Similarity, self).__init__()
        self.dim = normalise_dim
        self.tau = tau
        self.hard = hard

    def forward(self, log_alpha):
        log_alpha = log_alpha / self.tau
        alpha = torch.exp(log_alpha - (torch.logsumexp(log_alpha, dim=self.dim, keepdim=True)))

        if self.hard:
            # Straight through.
            index = alpha.max(self.dim, keepdim=True)[1]
            alpha_hard = torch.zeros_like(alpha, memory_format=torch.legacy_contiguous_format).scatter_(self.dim, index, 1.0)
            ret = alpha_hard - alpha.detach() + alpha
        else:
            ret = alpha
        return ret


def _get_mask(evals1, evals2, resolvant_gamma):
    scaling_factor = max(torch.max(evals1), torch.max(evals2))
    evals1, evals2 = evals1 / scaling_factor, evals2 / scaling_factor
    evals_gamma1 = (evals1 ** resolvant_gamma)[None, :]
    evals_gamma2 = (evals2 ** resolvant_gamma)[:, None]

    M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
    M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
    return M_re.square() + M_im.square()

def get_mask(evals1, evals2, resolvant_gamma):
    masks = []
    for bs in range(evals1.shape[0]):
        masks.append(_get_mask(evals1[bs], evals2[bs], resolvant_gamma))
    return torch.stack(masks, dim=0)


    
class RegularizedFMNet(nn.Module):
    """Compute the functional map matrix representation in DPFM"""
    def __init__(self, lmbda=100, resolvant_gamma=0.5, bidirectional=False):
        super(RegularizedFMNet, self).__init__()
        self.lmbda = lmbda
        self.resolvant_gamma = resolvant_gamma
        self.bidirectional = bidirectional

    def compute_functional_map(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        A = torch.bmm(evecs_trans_x, feat_x)  # [B, K, C]
        B = torch.bmm(evecs_trans_y, feat_y)  # [B, K, C]

        D = get_mask(evals_x, evals_y, self.resolvant_gamma)  # [B, K, K]

        A_t = A.transpose(1, 2)  # [B, C, K]
        A_A_t = torch.bmm(A, A_t)  # [B, K, K]
        B_A_t = torch.bmm(B, A_t)  # [B, K, K]

        C_i = []
        for i in range(evals_x.shape[1]):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.shape[0])], dim=0)
            C = torch.bmm(torch.inverse(A_A_t + self.lmbda * D_i), B_A_t[:, [i], :].transpose(1, 2))
            C_i.append(C.transpose(1, 2))

        Cxy = torch.cat(C_i, dim=1)
        return Cxy

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        """
        Forward pass to compute functional map
        Args:
            feat_x (torch.Tensor): feature vector of shape x. [B, Vx, C].
            feat_y (torch.Tensor): feature vector of shape y. [B, Vy, C].
            evals_x (torch.Tensor): eigenvalues of shape x. [B, K].
            evals_y (torch.Tensor): eigenvalues of shape y. [B, K].
            evecs_trans_x (torch.Tensor): pseudo inverse of eigenvectors of shape x. [B, K, Vx].
            evecs_trans_y (torch.Tensor): pseudo inverse of eigenvectors of shape y. [B, K, Vy].

        Returns:
            C (torch.Tensor): functional map from shape x to shape y. [B, K, K].
        """
        Cxy = self.compute_functional_map(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        if self.bidirectional:
            Cyx = self.compute_functional_map(feat_y, feat_x, evals_y, evals_x, evecs_trans_y, evecs_trans_x)
        else:
            Cyx = None

        return Cxy, Cyx