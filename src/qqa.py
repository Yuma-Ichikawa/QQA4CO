import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time

def batch_annealing(
    problem,
    sol_size=100,
    learning_rate=1,
    temp=0,
    min_bg=-2,
    max_bg=0.1,
    curve_rate=2,
    div_param=0,
    num_epochs=int(1e4),
    check_interval=1000,
    device="cpu",
    plot_dynamics=False,
    # Add auto-div control
    auto_divparam=False,
    div_target=0.3,
    div_param_lr=0.001
):
    """
    Single-instance batch annealing with optional div control.
    * auto_divparam controls div_param so that (div_value / (sol_size * problem.num_nodes)) ~ div_target.
    * Enhanced plotting with a stylish look.
    * Logs in a more visually appealing format (mean loss/penalty).
    """
    runtime_start = time()

    x = torch.rand((sol_size, problem.num_nodes), device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([x], lr=learning_rate)

    best_obj = 1e9
    best_sol = x.round()
    losses_hist, losses_std_hist = [], []
    penalties_hist, penalties_std_hist = [], []
    div_hist = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        bg = min_bg + (max_bg - min_bg) * epoch / num_epochs

        # Evaluate losses, penalties, diversity
        losses = problem.loss_fn(x)  # shape (sol_size,)
        penalties = torch.sum(1 - (1 - 2 * x) ** curve_rate, dim=1)  # shape (sol_size,)
        div_value = x.std(dim=0).sum()  # single scalar
        div_term = -div_value * sol_size

        # Weighted sum using mean-based stats
        total_loss = (torch.mean(losses) + torch.mean(penalties) * bg) * (1 - div_param) + div_term * div_param
        total_loss.backward()
        optimizer.step()

        # Optional random noise
        noise = torch.randn_like(x) * ((2 * learning_rate * temp) ** 0.5)
        with torch.no_grad():
            x.add_(noise).clamp_(0, 1)

        # Check best (round)
        with torch.no_grad():
            losses_round = problem.loss_fn(x.round())
            best_in_batch, _ = torch.min(losses_round, dim=0)
            if best_in_batch.item() < best_obj:
                best_obj = best_in_batch.item()
                best_sol = x.round()

        # Auto-div
        if auto_divparam and sol_size > 1:
            ratio = div_value.item() / (sol_size * problem.num_nodes)
            diff = ratio - div_target
            div_param += div_param_lr * diff
            div_param = max(0.0, min(1.0, div_param))

        # For plotting
        arr_losses = losses.detach().cpu().numpy()
        arr_pen = penalties.detach().cpu().numpy()
        losses_hist.append(arr_losses.mean())
        losses_std_hist.append(arr_losses.std())
        penalties_hist.append(arr_pen.mean())
        penalties_std_hist.append(arr_pen.std())
        div_hist.append(div_value.item())

        # Stylish log
        if epoch % check_interval == 0 or epoch == num_epochs - 1:
            mean_loss_val = arr_losses.mean()
            mean_pen_val = arr_pen.mean()
            print("\n============================== [LOG] ===============================")
            print(f"[ EPOCH {epoch} ]")
            print(f"  Best Loss So Far : {best_obj:.4f}")
            print(f"  Mean(Loss)       : {mean_loss_val:.4f}")
            print(f"  Mean(Penalty)    : {mean_pen_val:.4f}")
            print(f"  BG               : {bg:.4f}")
            print(f"  DIV Value        : {div_value.item():.4f}")
            print(f"  div_param        : {div_param:.4f}")
            print("====================================================================")

    runtime = time() - runtime_start
    print("\n============================== [FINAL] ==============================")
    print(f"FINAL BEST LOSS : {best_obj:.4f}")
    print(f"RUNTIME         : {runtime:.2f} s")
    print("====================================================================")
    print(f"sol_size:{sol_size}, lr:{learning_rate}, temp:{temp}, "
          f"min_bg:{min_bg}, max_bg:{max_bg}, curve_rate:{curve_rate}, div_param:{div_param}")

    if plot_dynamics:
        epochs = np.arange(num_epochs)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), facecolor="white")
        fig.suptitle("Batch Annealing (Single-Instance)", fontsize=16, fontweight="bold")

        for ax in axs:
            ax.grid(ls="--", alpha=0.6)

        # 1) Loss
        mean_l = np.array(losses_hist)
        std_l = np.array(losses_std_hist)
        axs[0].plot(epochs, mean_l, label='Mean Loss', color='darkblue', lw=2)
        axs[0].fill_between(epochs, mean_l - std_l, mean_l + std_l, alpha=0.2, color='cornflowerblue')
        axs[0].set_title("Loss", fontsize=14)
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        # 2) Penalties
        mean_p = np.array(penalties_hist)
        std_p = np.array(penalties_std_hist)
        axs[1].plot(epochs, mean_p, label='Mean Penalty', color='firebrick', lw=2)
        axs[1].fill_between(epochs, mean_p - std_p, mean_p + std_p, alpha=0.2, color='salmon')
        axs[1].set_title("Penalty", fontsize=14)
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Penalty')
        axs[1].legend()

        # 3) Diversity
        div_vals = np.array(div_hist)
        axs[2].plot(epochs, div_vals, label='Diversity', color='green', lw=2, marker='o',
                    markevery=max(num_epochs // 20, 1), ms=4)
        axs[2].set_title("Diversity", fontsize=14)
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Diversity')
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    return best_sol, best_obj, runtime



########################################
# MULTI-INSTANCE BATCH ANNEALING
########################################
def batch_instance_annealing(
    problem,
    sol_size=100,
    learning_rate=1,
    temp=0,
    min_bg=-2,
    max_bg=0.1,
    curve_rate=2,
    div_param=0,
    num_epochs=int(1e4),
    check_interval=1000,
    device="cpu",
    plot_dynamics=False
):
    """
    Multi-instance annealing (vectorized). No explicit loops over instance.
    * We store losses, penalties, and diversities as arrays of shape (num_epochs, num_instance).
    * Then plot them in lines, limiting to the first 10.
    """
    runtime_start = time()

    x = torch.rand((sol_size, problem.num_instance, problem.max_node), device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([x], lr=learning_rate)

    best_obj = np.full((problem.num_instance,), 1e9, dtype=np.float32)
    # Evaluate best
    with torch.no_grad():
        losses_round = problem.loss_fn(x.round())  # shape (sol_size, num_instance)
        min_vals, min_indices = torch.min(losses_round, dim=0)  # shape (num_instance,)
        best_sol = x.round()[min_indices, torch.arange(x.size(1))]
        best_obj = np.minimum(best_obj, min_vals.cpu().numpy())

    losses_array = []
    penalties_array = []
    div_array = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        bg = min_bg + (max_bg - min_bg) * epoch / num_epochs

        losses = problem.loss_fn(x)      # shape (sol_size, num_instance)
        penalties = torch.sum(1 - (1 - 2 * x) ** curve_rate, dim=2) # shape (sol_size, num_instance)
        div_values = x.std(dim=0).sum(dim=1) # shape (num_instance,)
        total_div_value = div_values.sum()
        div = -total_div_value * sol_size

        total_loss = (torch.sum(losses) + torch.sum(penalties * bg)) * (1 - div_param) + div * div_param
        total_loss.backward()
        optimizer.step()

        # Optional noise
        noise = torch.randn_like(x) * ((2 * learning_rate * temp) ** 0.5)
        with torch.no_grad():
            x.add_(noise).clamp_(0, 1)

        # Evaluate best
        with torch.no_grad():
            losses_round = problem.loss_fn(x.round())  # shape (sol_size, num_instance)
            min_vals, min_indices = torch.min(losses_round, dim=0)  # shape (num_instance,)
            best_sol = x.round()[min_indices, torch.arange(x.size(1))]
            best_obj = np.minimum(best_obj, min_vals.cpu().numpy())

        mean_losses = losses.mean(dim=0).detach().cpu().numpy()
        mean_pen = penalties.mean(dim=0).detach().cpu().numpy()
        div_vals = div_values.detach().cpu().numpy()

        losses_array.append(mean_losses)
        penalties_array.append(mean_pen)
        div_array.append(div_vals)

        if epoch % check_interval == 0 or epoch == num_epochs - 1:
            mean_loss_val = losses.mean().item()
            mean_pen_val = penalties.mean().item()
            print("\n============================== [LOG] ===============================")
            print(f"[ EPOCH {epoch} ]")
            print(f"  Current Best Obj : {best_obj}")
            print(f"  Mean(Loss)       : {mean_loss_val:.4f}")
            print(f"  Mean(Penalty)    : {mean_pen_val:.4f}")
            print(f"  BG               : {bg:.4f}")
            print(f"  div_param        : {div_param:.4f}")
            print("====================================================================")

    runtime = time() - runtime_start
    print("\n============================== [FINAL] ==============================")
    print(f"  BEST LOSS:{best_obj}")
    print(f"  RUN TIME:{runtime:.2f} s")
    print("====================================================================")
    print(f"sol_size:{sol_size}, learning_rate:{learning_rate}, temp:{temp}, "
          f"min_bg:{min_bg}, max_bg:{max_bg}, curve_rate:{curve_rate}, div_param:{div_param}")

    # Plot
    if plot_dynamics:
        epochs = np.arange(num_epochs)
        losses_array = np.stack(losses_array, axis=0)     # shape (num_epochs, num_instance)
        penalties_array = np.stack(penalties_array, axis=0)
        div_array = np.stack(div_array, axis=0)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5), facecolor='white')
        fig.suptitle("Batch Instance Annealing (Vectorized)", fontsize=16, fontweight="bold")

        for ax in axs:
            ax.grid(ls="--", alpha=0.6)

        max_inst = min(problem.num_instance, 10)

        # Loss
        lines_loss = axs[0].plot(epochs, losses_array[:, :max_inst])
        axs[0].set_title("Loss per Instance", fontsize=14)

        # Penalty
        lines_pen = axs[1].plot(epochs, penalties_array[:, :max_inst])
        axs[1].set_title("Penalty per Instance", fontsize=14)

        # Diversity
        lines_div = axs[2].plot(epochs, div_array[:, :max_inst])
        axs[2].set_title("Diversity per Instance", fontsize=14)

        axs[0].legend(lines_loss, [f"Inst {i}" for i in range(max_inst)], loc="upper right")
        axs[1].legend(lines_pen, [f"Inst {i}" for i in range(max_inst)], loc="upper right")
        axs[2].legend(lines_div, [f"Inst {i}" for i in range(max_inst)], loc="upper right")

        plt.tight_layout()
        plt.show()

    return best_sol, best_obj, runtime


def batch_annealing_mis_trajectory(
    problem,
    problem_P1,
    sol_size=100,
    learning_rate=1,
    temp=0,
    min_bg=-2,
    max_bg=0.1,
    curve_rate=2,
    div_param=0,
    num_epochs=int(1e4),
    check_interval=1000,
    mode="mean",
    device="cpu",
    plot_dynamics=False,
    auto_divparam=False,
    div_target=0.3,
    div_param_lr=0.001
):
    """
    Like batch_annealing, but also tracks MIS trajectory for problem_P1.
    * auto_divparam => (div_value / (sol_size * problem.num_nodes)) ~ div_target.
    """
    runtime_start = time()
    x = torch.rand((sol_size, problem.num_nodes), device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([x], lr=learning_rate)

    best_obj = 1e9
    dynamics_memory = {"MIS_DYNAMICS": []}

    losses_hist, losses_std_hist = [], []
    penalties_hist, penalties_std_hist = [], []
    div_hist = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        bg = min_bg + (max_bg - min_bg) * epoch / num_epochs

        losses = problem.loss_fn(x)
        penalties = torch.sum(1 - (1 - 2 * x) ** curve_rate, dim=1)
        if sol_size == 1:
            div_value = 0
            div = 0
        else:
            div_value = x.std(dim=0).sum()
            div = -div_value * sol_size

        total_loss = (torch.sum(losses) + torch.sum(penalties * bg)) * (1 - div_param) + div * div_param
        total_loss.backward()
        optimizer.step()

        noise = torch.randn_like(x) * ((2 * learning_rate * temp) ** 0.5)
        with torch.no_grad():
            x.add_(noise).clamp_(0, 1)

        with torch.no_grad():
            losses_round = problem.loss_fn(x.round())
            best_in_batch, _ = torch.min(losses_round, dim=0)
            if best_in_batch.item() < best_obj:
                best_obj = best_in_batch.item()

        if auto_divparam and sol_size > 1:
            ratio = div_value.item() / (sol_size * problem.num_nodes)
            diff = ratio - div_target
            div_param += div_param_lr * diff
            div_param = max(0.0, min(1.0, div_param))

        # Track MIS
        with torch.no_grad():
            loss_p1 = problem_P1.loss_fn(x.round())
            if mode == 'mean':
                mis_val = -torch.mean(loss_p1).item()
            else:
                mis_val = -torch.min(loss_p1).item()
            dynamics_memory["MIS_DYNAMICS"].append(mis_val)

        arr_losses = losses.detach().cpu().numpy()
        arr_pen = penalties.detach().cpu().numpy()
        losses_hist.append(arr_losses.mean())
        losses_std_hist.append(arr_losses.std())
        penalties_hist.append(arr_pen.mean())
        penalties_std_hist.append(arr_pen.std())
        div_hist.append(div_value if sol_size > 1 else 0)

        if epoch % check_interval == 0 or epoch == num_epochs - 1:
            print("\n============================== [LOG] ===============================")
            print(f"[ EPOCH {epoch} ]")
            print(f"  Best Loss So Far : {best_obj:.4f}")
            print(f"  Mean(Loss)       : {arr_losses.mean():.4f}")
            print(f"  Mean(Penalty)    : {arr_pen.mean():.4f}")
            print(f"  BG               : {bg:.4f}")
            print(f"  DIV Value        : {div_value:.4f}")
            print(f"  div_param        : {div_param:.4f}")
            print("====================================================================")

    runtime = time() - runtime_start
    print("\n============================== [FINAL] ===============================")
    print(f"  BEST LOSS : {best_obj:.4f}")
    print(f"  RUN TIME  : {runtime:.2f} s")
    print("======================================================================")
    print(f"sol_size:{sol_size}, learning_rate:{learning_rate}, temp:{temp}, "
          f"min_bg:{min_bg}, max_bg:{max_bg}, curve_rate:{curve_rate}, div_param:{div_param}")

    # Optional plotting
    if plot_dynamics:
        epochs = np.arange(num_epochs)
        fig, axs = plt.subplots(3, 1, figsize=(7, 10), facecolor='white')
        fig.suptitle("Batch Annealing MIS Trajectory", fontsize=16, fontweight="bold")

        for ax in axs:
            ax.grid(ls="--", alpha=0.6)

        # 1) Loss
        mean_l = np.array(losses_hist)
        std_l = np.array(losses_std_hist)
        axs[0].plot(epochs, mean_l, label='Loss (mean)', color='blue')
        axs[0].fill_between(epochs, mean_l - std_l, mean_l + std_l, alpha=0.3, color='blue')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        # 2) Penalty
        mean_p = np.array(penalties_hist)
        std_p = np.array(penalties_std_hist)
        axs[1].plot(epochs, mean_p, label='Penalty (mean)', color='green')
        axs[1].fill_between(epochs, mean_p - std_p, mean_p + std_p, alpha=0.3, color='green')
        axs[1].set_ylabel('Penalty')
        axs[1].legend()

        # 3) Diversity
        div_vals = np.array(div_hist)
        axs[2].plot(epochs, div_vals, label='div_value', color='red')
        axs[2].axhline(div_target, linestyle='--', color='black', label='div_target')
        axs[2].set_ylabel('Diversity')
        axs[2].legend()

        plt.xlabel('Epoch')
        plt.tight_layout()
        plt.show()

    return best_obj, runtime, dynamics_memory


def batch_annealing_categorical(
    problem,
    sol_size=100,
    learning_rate=1,
    temp=0,
    min_bg=-2,
    max_bg=0.1,
    curve_rate=2,
    div_param=0,
    num_epochs=int(1e4),
    check_interval=1000,
    device="cpu",
    plot_dynamics=False,
    auto_divparam=False,
    div_target=0.3,
    div_param_lr=0.001
):
    """
    Categorical single-instance annealing.
    * auto_divparam => ratio = div_value / (sol_size*problem.num_node) ~ div_target.
    """

    runtime_start = time()
    x = torch.rand((sol_size, problem.num_node, problem.num_category), device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW([x], lr=learning_rate)

    with torch.no_grad():
        max_indices = torch.argmax(x, dim=2)
        x_round = torch.zeros_like(x)
        x_round.scatter_(2, max_indices.unsqueeze(2), 1)
        best_loss, idx_min = torch.min(problem.loss_fn(x_round), dim=0)
        best_string = max_indices[idx_min].detach()

    losses_hist, losses_std_hist = [], []
    penalties_hist, penalties_std_hist = [], []
    div_hist = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        bg = min_bg + (max_bg - min_bg) * epoch / num_epochs

        x_norm = x / (torch.sum(x, dim=2).unsqueeze(2))
        losses = problem.loss_fn(x_norm)
        penalties = torch.sum(
            1 - torch.sum((x.shape[2] * x_norm - 1) ** curve_rate, dim=2)
            / ((x.shape[2] - 1) ** curve_rate + x.shape[2] - 1),
            dim=1
        )

        if sol_size == 0:
            div = 0
        else:
            div = -x_norm.std(dim=0).mean(dim=1).sum() * sol_size

        total_loss = (torch.sum(losses + penalties * bg)) * (1 - div_param) + div * div_param
        total_loss.backward()
        optimizer.step()

        noise = torch.randn_like(x) * ((2 * learning_rate * temp) ** 0.5)
        with torch.no_grad():
            x.add_(noise).clamp_(1e-5, 1)

        with torch.no_grad():
            x += noise
            x.clamp_(1e-5, 1)
            max_indices = torch.argmax(x, dim=2)
            x_round = torch.zeros_like(x)
            x_round.scatter_(2, max_indices.unsqueeze(2), 1)
            losses_round = problem.loss_fn(x_round)
            losses_round_min, idx_min = torch.min(losses_round, dim=0)

            if losses_round_min.item() < best_loss:
                best_loss = losses_round_min.item()
                best_string = max_indices[idx_min].detach()

        if auto_divparam:
            ratio = div.item() / (sol_size * problem.num_node)
            diff = ratio - div_target
            div_param += div_param_lr * diff
            div_param = max(0.0, min(1.0, div_param))

        arr_losses = losses.detach().cpu().numpy()
        arr_pen = penalties.detach().cpu().numpy()
        losses_hist.append(arr_losses.mean())
        losses_std_hist.append(arr_losses.std())
        penalties_hist.append(arr_pen.mean())
        penalties_std_hist.append(arr_pen.std())
        div_hist.append(div.item())

        if epoch % check_interval == 0 or epoch == num_epochs - 1:
            mean_loss_val = arr_losses.mean()
            mean_pen_val = arr_pen.mean()
            print("\n============================== [LOG] ===============================")
            print(f"[ EPOCH {epoch} ]")
            print(f"  Current Best Obj  : {best_loss:.4f}")
            print(f"  Mean(Loss)        : {mean_loss_val:.4f}")
            print(f"  Mean(Penalty)     : {mean_pen_val:.4f}")
            print(f"  BG                : {bg:.4f}")
            print(f"  DIV Value         : {div.item():.4f}")
            print(f"  div_param         : {div_param:.4f}")
            print("====================================================================")

    runtime = time() - runtime_start
    print("\n============================== [FINAL] ===============================")
    print(f"BEST LOSS:{best_loss}, RUN TIME:{runtime:.2f}")
    print("======================================================================")
    print(f"sol_size:{sol_size}, learning_rate:{learning_rate}, temp:{temp}, "
          f"min_bg:{min_bg}, max_bg:{max_bg}, curve_rate:{curve_rate}, div_param:{div_param}")

    if plot_dynamics:
        epochs = np.arange(num_epochs)
        # --- サブプロットを横に 3 つ並べる
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), facecolor="white")
        fig.suptitle("Batch Annealing (Categorical)", fontsize=16, fontweight="bold")

        # 全てのサブプロットで同じグリッドスタイルを設定
        for ax in axs:
            ax.grid(ls="--", alpha=0.6)

        # --- (1) Loss
        mean_l = np.array(losses_hist)
        std_l = np.array(losses_std_hist)
        axs[0].plot(epochs, mean_l, label='Loss (mean)', color='blue', lw=2)
        axs[0].fill_between(epochs, mean_l - std_l, mean_l + std_l, alpha=0.3, color='skyblue')
        axs[0].set_title("Loss", fontsize=14)
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        # --- (2) Penalty
        mean_p = np.array(penalties_hist)
        std_p = np.array(penalties_std_hist)
        axs[1].plot(epochs, mean_p, label='Penalty (mean)', color='green', lw=2)
        axs[1].fill_between(epochs, mean_p - std_p, mean_p + std_p, alpha=0.3, color='lightgreen')
        axs[1].set_title("Penalty", fontsize=14)
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Penalty')
        axs[1].legend()

        # --- (3) Diversity
        div_vals = np.array(div_hist)
        axs[2].plot(
            epochs, div_vals, label='Diversity', color='red', lw=2,
            marker='o', markevery=max(num_epochs // 20, 1), ms=4
        )
        axs[2].axhline(div_target, linestyle='--', color='black', label='div_target')
        axs[2].set_title("Diversity", fontsize=14)
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Diversity')
        axs[2].legend()

        plt.tight_layout()
        plt.show()


    return best_string, best_loss, runtime
