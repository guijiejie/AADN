import torch


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)
CUR_DEVICE = torch.device("cuda")


def attack_predict_or_gt(model, hazy, label, epsilon, alpha, attack_iters, criterion):
    alpha = alpha / 255.0
    epsilon = epsilon / 255.0

    upper_limit, lower_limit = 1, 0

    # torch.cuda.empty_cache()

    delta = torch.zeros_like(hazy).to(CUR_DEVICE)
    delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit - hazy, upper_limit - hazy)
    delta.requires_grad = True
    for _ in range(attack_iters):
        robust_output = model((hazy + delta))

        loss = criterion(robust_output, label)
        grad = torch.autograd.grad(loss, [delta])[0].detach()
        d = delta
        g = grad
        x = hazy
        d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data = d
    max_delta = delta.detach()
    return max_delta