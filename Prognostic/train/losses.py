import numpy as np
from scipy import stats
prev_bs = 0
prior_dist = None
import torch
import torch.nn.functional as F
import torch.nn as nn

def cox_cost(logits, at_risk, observed,failures,ties):
    # only uncensored data / dead patients were calculated
    # logits: output sort by real survival time
    logL = 0
    # pre-calculate cumsum
    cumsum_logits = torch.cumsum(logits, dim=0)
    # dim不变，后面依次相加
    hazard_ratio = torch.exp(logits)
    cumsum_hazard_ratio = torch.cumsum(hazard_ratio, dim=0)
    if ties == 'noties':
        # no ties
        log_risk = torch.log(cumsum_hazard_ratio)

        likelihood = logits - log_risk

        uncensored_likelihood = likelihood * observed.float()

        logL = -1 * uncensored_likelihood.sum()
    else:
        # Loop for death times
            print(failures)

            for t in failures:
                tfail = failures[t]
                trisk = at_risk[t]
                d = len(tfail)
                dr = len(trisk)

                logL += -cumsum_logits[tfail[-1]] + (0 if tfail[0] == 0 else cumsum_logits[tfail[0] - 1])

                if ties == 'breslow':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    logL += torch.log(s) * d
                elif ties == 'efron':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    r = cumsum_hazard_ratio[tfail[-1]] - (0 if tfail[0] == 0 else cumsum_hazard_ratio[tfail[0] - 1])

                    for j in range(d):

                        logL += torch.log(s - j * r / d)

                else:
                    raise NotImplementedError('tie breaking method not recognized')

    return logL

def _neg_partial_log(prediction, T, E):
    """
    calculate cox loss, Pytorch implementation by Huang, https://github.com/huangzhii/SALMON
    :param X: variables
    :param T: Time
    :param E: Status
    :return: neg log of the likelihood
    """

    current_batch_len = len(prediction)
    # print(current_batch_len)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.cuda()
    train_ystatus = E

    theta = prediction.reshape(-1)
    exp_theta = torch.exp(theta)

    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

    return loss_nn




def GaussianKernel(radius, std):
    """
    Generate a gaussian blur kernel based on the given radius and std.

    Args
    ----------
    radius: int
        Radius of the Gaussian kernel. Center not included.
    std: float
        Standard deviation of the Gaussian kernel.

    Returns
    ----------
    weight: torch.FloatTensor, [2 * radius + 1, 2 * radius + 1]
        Output Gaussian kernel.

    """
    size = 2 * radius + 1
    weight = torch.ones(size, )
    weight.requires_grad = False
    for j in range(-radius, radius + 1):
        dis = j * j
        weight[j + radius] = np.exp(-dis / (2 * std * std))
    weight = weight / weight.sum()
    return weight


def update_prior_dist(batch_size, alpha, beta):
    """
    Update the samples of prior distribution due to the change of batchsize.

    Args
    ----------
    batch_size: int
        Current batch size.
    alpha: float
        Parameter of Beta distribution.
    beta: float
        Parameter of Beta distribution.

    """
    global prior_dist
    grid_points = torch.arange(1., 2 * batch_size, 2.).float().cuda() / (2 * batch_size)
    grid_points_np = grid_points.cpu().numpy()
    grid_points_icdf = stats.beta.ppf(grid_points_np, a=alpha, b=beta)
    prior_dist = torch.tensor(grid_points_icdf).float().cuda().unsqueeze(1)


def shapingloss(assign, radius, num_parts, std=0.2, alpha=1, beta=0.01, eps=1e-5):
    """
    Wasserstein shaping loss for Bernoulli distribution.

    Args
    ----------
    assign: torch.cuda.FloatTensor, [batch_size, num_parts, height, width]
        Assignment map for grouping.
    radius: int
        Radius for the Gaussian kernel.
    std: float
        Standard deviation for the Gaussian kernel.
    num_parts: int
        Number of object parts in the current model.
    alpha: float
        Parameter of Beta distribution.
    beta: float
        Parameter of Beta distribution.
    eps:
        Epsilon for rescaling the distribution.

    Returns
    ----------
    loss: torch.cuda.FloatTensor, [1, ]
        Average Wasserstein shaping loss for the current minibatch.

    """
    global prev_bs, prior_dist
    batch_size = assign.shape[0]

    # Gaussian blur
    if radius == 0:
        assign_smooth = assign
    else:
        weight = GaussianKernel(radius, std)
        weight = weight.contiguous().view(1, 1, 2*radius+1).expand(
            num_parts, 1, 2*radius+1
        ).cuda()
        assign_smooth = F.conv1d(assign.unsqueeze(2), weight, groups=num_parts)

    # pool the assignment maps into [batch_size, num_parts] for the empirical distribution of part occurence
    # part_occ = F.adaptive_max_pool2d(assign_smooth, (1, 1)).squeeze(2).squeeze(2)
    emp_dist, _ = assign_smooth.sort(dim=0, descending=False)

    # the Beta prior
    if batch_size != prev_bs:
        update_prior_dist(batch_size, alpha, beta)

    # rescale the distribution
    emp_dist = (emp_dist + eps).log()
    prior_dist = (prior_dist + eps).log()

    # return the loss
    output_nk = (emp_dist - prior_dist).abs()
    loss = output_nk.mean()
    return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, targets):
        """
        Args:
        	inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        	targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.long().unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def cosine_dist(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return 1-cosine


def _batch_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if(indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


def _batch_mid_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
    hard_p = sorted_mat_distance[:, 1]
    hard_p_indice = positive_indices[:, 1]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if(indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n

class TripletLoss(nn.Module):
    '''
    Compute Triplet loss augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    '''

    def __init__(self, margin, normalize_feature=False, mid_hard=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()
        self.mid_hard = mid_hard

    def forward(self, emb, label, emb_=None):
        if emb_ is None:
            mat_dist = euclidean_dist(emb, emb)
            # mat_dist = cosine_dist(emb, emb)
            assert mat_dist.size(0) == mat_dist.size(1)
            N = mat_dist.size(0)
            mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
            if self.mid_hard:
                dist_ap, dist_an = _batch_mid_hard(mat_dist, mat_sim)
            else:
                dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
            assert dist_an.size(0)==dist_ap.size(0)
            y = torch.ones_like(dist_ap)
            loss = self.margin_loss(dist_an, dist_ap, y)
            prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
            return loss, prec
        else:
            mat_dist = euclidean_dist(emb, emb_)
            N = mat_dist.size(0)
            mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
            dist_ap, dist_an = _batch_hard(mat_dist, mat_sim)
            y = torch.ones_like(dist_ap)
            loss = self.margin_loss(dist_an, dist_ap, y)
            return loss

class SoftTripletLoss(nn.Module):
    def __init__(self, margin=None, normalize_feature=False, mid_hard=False):
        super(SoftTripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.mid_hard = mid_hard

    def forward(self, emb1, emb2, label):
        if self.normalize_feature:
            # equal to cosine similarity
            emb1 = F.normalize(emb1)
            emb2 = F.normalize(emb2)

        mat_dist = euclidean_dist(emb1, emb1)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()
        if self.mid_hard:
            dist_ap, dist_an, ap_idx, an_idx = _batch_mid_hard(mat_dist, mat_sim, indice=True)
        else:
            dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
        assert dist_an.size(0)==dist_ap.size(0)
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = F.log_softmax(triple_dist, dim=1)
        if (self.margin is not None):
            loss = (- self.margin * triple_dist[:,0] - (1 - self.margin) * triple_dist[:,1]).mean()
            return loss

        mat_dist_ref = euclidean_dist(emb2, emb2)
        dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N,1).expand(N,N))[:,0]
        dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N,1).expand(N,N))[:,0]
        triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
        triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

        loss = (- triple_dist_ref * triple_dist).mean(0).sum()
        return loss

# https://gist.github.com/dimartinot/6ab415ce215d1dd310e86a4a7079e43e
def get_quadruplet_mask(
        labels: torch.Tensor,  # [B]
        device: torch.device
):
    B = labels.size(0)

    # Make sure that i != j != k != l
    indices_equal = torch.eye(B, dtype=torch.bool).to(device=device)  # [B, B]
    indices_not_equal = ~indices_equal  # [B, B]
    i_not_equal_j = indices_not_equal.view(B, B, 1, 1)  # [B, B, 1, 1]
    j_not_equal_k = indices_not_equal.view(1, B, B, 1)  # [B, 1, 1, B]
    k_not_equal_l = indices_not_equal.view(1, 1, B, B)  # [1, 1, B, B]
    distinct_indices = i_not_equal_j & j_not_equal_k & k_not_equal_l  # [B, B, B, B]

    # Make sure that labels[i] == labels[j]
    #            and labels[j] != labels[k]
    #            and labels[k] != labels[l]
    labels_equal = labels.view(1, B) == labels.view(B, 1)  # [B, B]
    i_equal_j = labels_equal.view(B, B, 1, 1)  # [B, B, 1, 1]
    j_equal_k = labels_equal.view(1, B, B, 1)  # [1, B, B, 1]
    k_equal_l = labels_equal.view(1, 1, B, B)  # [1, 1, B, B]

    return (i_equal_j & ~j_equal_k & ~k_equal_l) & distinct_indices  # [B, B, B, B]

def RFS_batch_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if(indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n

class SoftListTripletLoss(nn.Module):
    def __init__(self, margin=None, normalize_feature=False, mid_hard=False, soft_thred=0.1):
        super(SoftListTripletLoss, self).__init__()
        self.soft_thred = soft_thred
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.mid_hard = mid_hard

    def forward_E0(self, mat_dist, label, events):
        # E1 之间尽量远
        events_mask = events == 0
        E0_label = label[events_mask]
        E0_mat_dist = mat_dist[events_mask.squeeze(), :][:,events_mask.squeeze()]
        E0_N = E0_mat_dist.size(0)
        E0_mat_sim = E0_label.expand(E0_N, E0_N).eq(E0_label.expand(E0_N, E0_N).t()).float()
        # E0_mat_sim = 1 - abs(E0_label.expand(E0_N, E0_N) - E0_label.expand(E0_N, E0_N).t())
        dist_ap, dist_an, ap_idx, an_idx = RFS_batch_hard(E0_mat_dist, E0_mat_sim, indice=True)

        assert dist_an.size(0)==dist_ap.size(0)
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = F.log_softmax(triple_dist, dim=1)
        if (self.margin is not None):
            # ap_v = label - label[ap_idx]
            # an_v = label - label[an_idx]
            # loss = (- self.margin * (1-abs(ap_v)) * triple_dist[:,0] - (1 - self.margin) * (1-abs(an_v)) * triple_dist[:,1]).mean()
            loss = (- self.margin * triple_dist[:,0] - (1 - self.margin) * triple_dist[:,1]).mean()
            return loss

    # https://gist.github.com/dimartinot/6ab415ce215d1dd310e86a4a7079e43e
    # https: // gist.github.com / tomekkorbak / bdea3fb841fcd390b58f2643eaaf365b
    def forward_E0E1(self, mat_dist, label, events):
        # E0 RFS time > E1 RFS time 之间尽量远
        events_mask = events == 1
        E1_label = label[events_mask]
        unique_E1_label = torch.unique(E1_label)


        quadruplet_loss = torch.tensor(0.0).to(label.device)
        for iE1_label in unique_E1_label:
            with torch.no_grad():
                # 当前E1样本到>RFS time E0的距离
                E0BE1_mask = (label >= iE1_label) & (~events_mask)
                # 当前E1样本到同样 E1 样本的距离
                iE1_mask = (iE1_label == label) & (events_mask)
                # 当前E1样本到 os time 较大的 E1 样本的距离
                iE1BE1_mask = (iE1_label < label) & (events_mask)

            E0BE1_dist = mat_dist[E0BE1_mask.squeeze(), :][:,E0BE1_mask.squeeze()]
            iE1E1_dist = mat_dist[iE1_mask.squeeze(), :][:,iE1_mask.squeeze()]
            iE1BE1_dist = mat_dist[iE1BE1_mask.squeeze(), :][:,iE1BE1_mask.squeeze()]
            if E0BE1_dist.size(0) > 0 and iE1E1_dist.size(0) > 0:
                an_dist = F.relu(self.margin + torch.mean(iE1E1_dist) - torch.mean(E0BE1_dist))
                quadruplet_loss += an_dist
            if iE1E1_dist.size(0) > 0 and iE1BE1_dist.size(0) > 0:
                ap_dist = F.relu(self.soft_thred + torch.mean(iE1E1_dist) - torch.mean(iE1BE1_dist))
                quadruplet_loss += ap_dist
        return quadruplet_loss

    def forward(self, emb1, label, events):
        if self.normalize_feature:
            # equal to cosine similarity
            emb1 = F.normalize(emb1)
        mat_dist = euclidean_dist(emb1, emb1)
        assert mat_dist.size(0) == mat_dist.size(1)

        # E0_tri_loss = self.forward_E0(mat_dist, label, events)
        E0E1_tri_loss = self.forward_E0E1(mat_dist, label, events)
        return E0E1_tri_loss

class E1TimeFitLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(E1TimeFitLoss, self).__init__()
        self.reduction = reduction

    def forward(self, predict, events, RFS_time):
        events_mask = events == 1
        return F.mse_loss(
            predict[events_mask],
            RFS_time[events_mask],
            reduction=self.reduction
        )