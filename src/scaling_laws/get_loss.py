import requests
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools

DEFAULT_API_KEY = """ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDCj8Lkwc96MtE2JjXLIgF1mLCI97JpoqeUKx9mszDt0xYuQ8HOGqa2D5SgBSeha+dG182xGnAAJ961989Ex8D56FkqMifGhTJSYrJh50G8RV+4MQJyRg9kiOvB7Vdbe/gOmKC1cZDLbJhvbgdmhiJcHslq++PbfNP8a21dcQd35kbAKMaU0kGsI5khuScMQ0VbEapAjxIriwmNVZFgQjixvlqUvawl3Nm4UQisjkPt3M+OobLNcp0Zm/4fjategtlJ0KwfG8EnadBjMIriVW3JNCurB9qH+EkOqKXEj2WM4++QwOmFyiWYJfD26olwj9DJSu534hGZQFrI2nI8+js8xLzqHCL0tXesnB1bSIYIyoDN6uTmmsN/ZpdxPyiv+oBnBXOlJRcb8hYRG5i9DWL0dq2HD+NsN26M94c8vxd+iuORas5SS5FFSLXMx1hj8a7GCsDo2iiw4NBZ68HENZMJ4jUy63cevqbkjklJKPLGVeaJTg0WZM/fJj/XEUW4C8s= ayushalag@Ayushs-MacBook-Pro.local"""

def get_loss(config):
    response = requests.get("http://hyperturing.stanford.edu:8000/loss", config)
    print(response.json())
    return response.json()["loss"]

def aspect_ratio_scaling_law():
    config = {
        "num_heads": 2,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "api_key": DEFAULT_API_KEY,
        "train_flops": int(3e16)
    }

    layer_d_tuples = [(5, 242), (6, 220), (7, 204), (8, 192)]
    for num_layers, d_model in layer_d_tuples:
        config["num_layers"] = num_layers
        config["d_model"] = d_model
        print(config)
        loss = get_loss(config)
        print(f"Num layers: {num_layers}, d_model: {d_model}, loss: {loss}")

def heads_scaling_law():
    config = {
        "num_layers": 5,
        "d_model": 256,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "api_key": DEFAULT_API_KEY,
        "train_flops": int(3e16)
    }

    for num_heads in [2, 4, 8, 16]:
        config["num_heads"] = num_heads
        loss = get_loss(config)
        print(f"Num heads: {num_heads}, loss: {loss}")

def batch_size_lr_scaling_law():
    config = {
        "num_layers": 5,
        "d_model": 256,
        "num_heads": 2,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "api_key": DEFAULT_API_KEY,
        "train_flops": int(3e16)
    }

    for batch_size in [128, 256]:
        for learning_rate in [0.0001, 0.0002, 0.0005, 0.001]:
            config["batch_size"] = batch_size
            config["learning_rate"] = learning_rate
            loss = get_loss(config)
            print(f"Batch size: {batch_size}, learning rate: {learning_rate}, loss: {loss}")

def final_batch_size_lr_scaling_law():
    config = {
        "num_layers": 5,
        "d_model": 256,
        "num_heads": 2,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "api_key": DEFAULT_API_KEY,
        "train_flops": int(6e16)
    }

    for learning_rate in [0.0005, 0.001]:
        config["learning_rate"] = learning_rate
        loss = get_loss(config)
        print(f"Learning rate: {learning_rate}, loss: {loss}")

def final_experiments():
    config = {
        "num_layers": 8,
        "d_model": 260,
        "num_heads": 2,
        "batch_size": 128,
        "learning_rate": 0.001,
        "api_key": DEFAULT_API_KEY,
        "train_flops": int(1e17)
    }

    loss = get_loss(config)
    print(f"Loss: {loss}, config: {config}")

    config2 = {
        "num_layers": 10,
        "d_model": 342,
        "num_heads": 3,
        "batch_size": 128,
        "learning_rate": 0.001,
        "api_key": DEFAULT_API_KEY,
        "train_flops": int(3e17)
    }

    loss = get_loss(config2)
    print(f"Loss: {loss}, config: {config2}")


def params_scaling_law():
    config = {
        "batch_size": 128,
        "learning_rate": 0.0001,
        "api_key": DEFAULT_API_KEY
    }

    results = defaultdict(list)
    flop_budgets = [1e13, 3e13, 6e13, 1e14, 3e14, 6e14, 1e15, 3e15, 6e15, 1e16, 3e16]
    # for train_flops in [1e13, 3e13, 6e13, 1e14, 3e14, 6e14, 1e15]:
    for train_flops in flop_budgets:
        config["train_flops"] = int(train_flops)
        if train_flops > 3e14:
            aspect_ratios = [32, 64]
        elif train_flops < 3e14:
            aspect_ratios = [16, 32]
        else:
            aspect_ratios = [16, 32, 64]

        for aspect_ratio in aspect_ratios:
            # for params in [5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 5e8, 1e9]:
            if train_flops < 1e16:
                layers = [2, 3, 4, 5, 6]
            else:
                layers = [2, 3, 4, 5, 6, 7, 8, 10, 12]
            for num_layers in layers:
                d_model = max(min(aspect_ratio * num_layers, 1024), 64)
                params = 12 * num_layers * (d_model ** 2)
            # for params in [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 5e8, 1e9]:
                # d = int(np.cbrt((params * aspect_ratio) / 12))
                # n = int(d / aspect_ratio)
                config["num_layers"] = num_layers
                config["d_model"] = d_model
                num_heads = min(max(d_model // 128, 2), 16)
                config["num_heads"] = num_heads
                if params < 1e10:
                    config["learning_rate"] = 0.001

                loss = get_loss(config)
                print(f"Train FLOPs: {train_flops}, num_layers: {num_layers}, num_heads: {num_heads}, d_model: {d_model}, params: {params}, loss: {loss}")
                results[train_flops].append((params, loss))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    min_params, min_losses, min_flops = [], [], []

    minima_flops = [3e14, 6e14, 1e15, 3e15, 6e15, 1e16, 3e16]

    for F in flop_budgets:
        pts = np.array(results[F])
        params, losses = pts[:, 0], pts[:, 1]
        c = next(colors)

        ax1.scatter(params, losses, label=f"{F:.0e} FLOPs",
                    s=30, alpha=0.6, color=c)

        x = np.log10(params)
        y = losses
        a, b, cc = np.polyfit(x, y, 2)
        x_fit = np.linspace(x.min(), np.log10(1e9), 200)
        y_fit = np.polyval([a, b, cc], x_fit)
        ax1.plot(10**x_fit, y_fit, lw=2, ls="--", color=c)

        x_min = -b / (2 * a)
        param_min = 10 ** x_min
        loss_min = a * x_min**2 + b * x_min + cc

        if F in minima_flops:
            min_params.append(param_min)
            min_losses.append(loss_min)
            min_flops.append(F)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of parameters")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs. parameters (per FLOP budget)")
    ax1.legend(title="Compute budget", fontsize="small")
    ax1.grid(True, which="both", ls=":")

    ax2.scatter(min_flops, min_params, s=30, color='blue')
    for p, l, F in zip(min_params, min_losses, min_flops):
        ax2.annotate(f"{F:.0e}", (p, l), textcoords="offset points",
                    xytext=(4, 4), fontsize=8)

    log_flops = np.log10(min_flops)
    log_params = np.log10(min_params)
    m, b = np.polyfit(log_flops, log_params, 1)
    print(f"Slope: {m:.2f}, intercept: {b:.2f}")

    x_fit = np.linspace(log_flops.min(), np.log10(1e19), 100)
    y_fit = m * x_fit + b
    C_proj = 1e19
    N_proj = 10**(m * np.log10(C_proj) + b)

    ax2.scatter(C_proj, N_proj,
                marker='x', s=120, linewidths=2,
                color='red', zorder=6,
                label=f'Projected N: {N_proj:,.1e}')

    ymin_plot = ax2.get_ylim()[0]
    xmin_plot = ax2.get_xlim()[0]

    ax2.plot([C_proj, C_proj], [ymin_plot, N_proj],
            ls=':', lw=1.5, color='red', alpha=0.8)
    ax2.plot([xmin_plot, C_proj], [N_proj, N_proj],
            ls=':', lw=1.5, color='red', alpha=0.8)

    ax2.plot(10**x_fit, 10**y_fit, 'b--', label=f'Slope: {m:.2f}')
    ax2.legend()

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("# FLOPs")
    ax2.set_ylabel("Optimal # parameters")
    ax2.set_title("Optimal parameters vs. FLOPs (log–log)")
    ax2.grid(True, which="both", ls=":")

    ax3.scatter(min_flops, min_losses, s=30, color='blue')
    for f, l in zip(min_flops, min_losses):
        ax3.annotate(f"{f:.0e}", (f, l), textcoords="offset points",
                    xytext=(4, 4), fontsize=8)

    log_flops = np.log10(min_flops)
    m_loss, b_loss = np.polyfit(log_flops, min_losses, 1)

    x_fit = np.linspace(log_flops.min(), np.log10(1e19), 100)
    y_fit = m_loss * x_fit + b_loss

    ax3.plot(10**x_fit, y_fit, 'b--', label=f'Slope: {m_loss:.2f}')

    loss_proj = m_loss * np.log10(C_proj) + b_loss
    ymin_plot = ax3.get_ylim()[0]
    xmin_plot = ax3.get_xlim()[0]
    ax3.scatter(C_proj, loss_proj,
                marker='x', s=120, linewidths=2,
                color='red', zorder=6,
                label=f'Projected loss: {loss_proj:,.1e}')
    ax3.plot([C_proj, C_proj], [ymin_plot, loss_proj],
            ls=':', lw=1.5, color='red', alpha=0.8)
    ax3.plot([xmin_plot, C_proj], [loss_proj, loss_proj],
            ls=':', lw=1.5, color='red', alpha=0.8)
    ax3.legend()

    ax3.set_xscale("log")
    ax3.set_xlabel("# FLOPs")
    ax3.set_ylabel("Minimum Loss")
    ax3.set_title("Loss vs. FLOPs")
    ax3.grid(True, which="both", ls=":")

    plt.tight_layout()
    plt.show()


def get_total_flops(config):
    response = requests.get("http://hyperturing.stanford.edu:8000/total_flops_used", config)
    total_flops = response.json()

    print(f"Total FLOPs: {total_flops}")
    print("FLOP Percentage: ", total_flops / 2e18)

if __name__ == "__main__":
    params_scaling_law()
    # final_experiments()
    # aspect_ratio_scaling_law()
    # heads_scaling_law()
    # batch_size_lr_scaling_law()
    # final_batch_size_lr_scaling_law()
    api_key_config = {
        "api_key": DEFAULT_API_KEY
    }
    get_total_flops(api_key_config)