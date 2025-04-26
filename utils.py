import torch


def KLDiv(p, q):
    return torch.sum(p * (p.log() - q.log()), -1)

def NLL(p, X):
    return torch.nn.functional.cross_entropy(
            input=torch.log(p[:, :-1].transpose(1, 2)),
            target=X[:, 1:].cpu().long(),
            reduction="none",
        )

def pairwise_distance(preds):
    return torch.cdist(preds, preds).mean().item()