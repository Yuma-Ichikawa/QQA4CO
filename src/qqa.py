import torch
from time import time

def optimize(num_node,
             loss_func,
             parallel_size=100,
             num_epoch=1000,
             lr=0.1,
             temp=1e-3,
             device="cpu",
             min_bg=-2,
             max_bg=0.1,
             curve_rate=4,
             div_param=0.1,
             check_interval=1000
             ):
    runtime_start=time()

    x = torch.rand((parallel_size, num_node), device=device, requires_grad=True)

    optimizer = torch.optim.AdamW([x], lr=lr)

    best_loss, idx_min = torch.min(loss_func(x.round()), dim=0)
    best_bit_string = (x[idx_min,:].detach() >= 0.5)*1

    for epoch in range(num_epoch):
        optimizer.zero_grad()
        bg = min_bg + (max_bg - min_bg) * epoch / num_epoch
        losses = loss_func(x)
        penalties = torch.sum(1 - (1 - 2 * x) ** curve_rate, dim=1)
        if parallel_size == 0:
            div = 0
        else:
            div = -x.std(dim=0).sum() * parallel_size
        loss = (torch.sum(losses + penalties * bg)) * (1 - div_param) + div * div_param
        loss.backward()
        optimizer.step()
        noise = torch.randn_like(x) * ((2*lr*temp)**(1/2))
        with torch.no_grad():
            x += noise
            x.clamp_(0, 1)
            losses_round = loss_func(x.round())
            losses_round_min, idx_min = torch.min(losses_round, dim=0)
        if losses_round_min.item() < best_loss:
            best_loss = losses_round_min.item()
            best_bit_string = (x[idx_min,:].detach() >= 0.5)*1

        if epoch % check_interval == 0 or epoch == num_epoch - 1:
            print(f"EPOCH:{epoch}, LOSS:{torch.sum(losses).item()}, PENALTY:{torch.sum(penalties).item()}, PARAM:{bg}")

    runtime = time()-runtime_start
    return best_bit_string, runtime

def optimize_categorical(num_node,
                         num_category,
                         loss_func,
                         parallel_size=100,
                         num_epoch=1000,
                         lr=0.1,
                         temp=1e-3,
                         device="cpu",
                         min_bg=-2,
                         max_bg=0.1,
                         curve_rate=4,
                         div_param=0.1,
                         check_interval=1000
                         ):
    runtime_start=time()

    x = torch.rand((parallel_size, num_node, num_category), device=device, requires_grad=True)

    optimizer = torch.optim.AdamW([x], lr=lr)

    max_indices = torch.argmax(x, dim=2)
    x_round = torch.zeros_like(x)
    x_round.scatter_(2, max_indices.unsqueeze(2), 1)
    best_loss, idx_min = torch.min(loss_func(x_round), dim=0)
    best_string = max_indices[idx_min].detach()

    for epoch in range(num_epoch):
        optimizer.zero_grad()
        bg = min_bg + (max_bg - min_bg) * epoch / num_epoch
        x_norm = x / (torch.sum(x, dim=2).unsqueeze(2))
        losses = loss_func(x_norm)
        penalties = torch.sum(1 - torch.sum((x_norm.shape[2] * x_norm - 1) ** curve_rate, dim=2) / ((x_norm.shape[2] - 1) ** curve_rate + x_norm.shape[2] - 1), dim=1)
        if parallel_size == 0:
            div = 0
        else:
            div = -x_norm.std(dim=0).mean(dim=1).sum() * parallel_size
        loss = (torch.sum(losses + penalties * bg)) * (1 - div_param) + div * div_param
        loss.backward()
        optimizer.step()
        noise = torch.randn_like(x) * ((2*lr*temp)**(1/2))
        with torch.no_grad():
            x += noise
            x.clamp_(1e-5, 1)
            max_indices = torch.argmax(x, dim=2)
            x_round = torch.zeros_like(x)
            x_round.scatter_(2, max_indices.unsqueeze(2), 1)
            losses_round = loss_func(x_round)
            losses_round_min, idx_min = torch.min(losses_round, dim=0)
        if losses_round_min.item() < best_loss:
            best_loss = losses_round_min.item()
            best_string = max_indices[idx_min].detach()

        if epoch % check_interval == 0 or epoch == num_epoch - 1:
            print(f"EPOCH:{epoch}, LOSS:{torch.sum(losses).item()}, PENALTY:{torch.sum(penalties).item()}, PARAM:{bg}")

    runtime = time()-runtime_start
    return best_string, runtime