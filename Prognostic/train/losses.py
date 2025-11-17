import numpy as np
from scipy import stats

import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
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

class E1TimeFitLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(E1TimeFitLoss, self).__init__()
        self.reduction = reduction

    def forward(self, predict, events, RFS_time):

        events_mask = events == 1
        time_loss =  F.mse_loss(
            predict[events_mask],
            RFS_time[events_mask],
            reduction=self.reduction
        )
        # 检查loss是否为NaN
        if torch.isnan(time_loss) or torch.isinf(time_loss):
            # 创建零loss，但保留计算图结构
            time_loss = torch.tensor(0.0, device=time_loss.device, requires_grad=False)
            # nan_detected = True
        # else:
        #     nan_detected = False
        return time_loss