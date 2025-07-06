import json
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def parse_data():
    isoflops_path = '/Users/ayushalag/Documents/StanfordMS/cs336/a3/data/isoflops_curves.json'
    with open(isoflops_path, 'r') as f:
        data = json.load(f)

        compute_budget_to_param_loss = {}
        for d in data:
            compute_budget = d['compute_budget']
            param_loss = d['final_loss']
            parameters = d['parameters']

            if compute_budget not in compute_budget_to_param_loss or param_loss < compute_budget_to_param_loss[compute_budget][1]:
                compute_budget_to_param_loss[compute_budget] = (parameters, param_loss)

    budgets = np.array(sorted(compute_budget_to_param_loss.keys()))
    optimal_params = np.array([compute_budget_to_param_loss[b][0] for b in budgets])
    optimal_data = np.array([b / (6 * compute_budget_to_param_loss[b][0]) for b in budgets])
    return np.log10(budgets), np.log10(optimal_params), np.log10(optimal_data)

# linear fit on a log-log scale
def linear_law(C, k, a):
    return k + a * C

def fit_curve(func, log_x, log_y, init_guesses=None):
    if init_guesses is None:
        init_guesses = [1e-5, 0.5]

    popt, pcov = curve_fit(func, log_x, log_y, p0=init_guesses)
    k_fit, a_fit = popt
    perr = np.sqrt(np.diag(pcov))
    print(f"k_fit: {k_fit}, a_fit: {a_fit}, perr: {perr}")

    for C_test in [1e23, 1e24]:
        N_pred = func(np.log10(C_test), k_fit, a_fit)
        N_pred = 10**N_pred
        print(f"Predicted N_opt at C={C_test:.0e} FLOPs:  N ≈ {N_pred:,.0f}")

    return popt, pcov

def plot_curve(popt, log_x, log_y, xlabel, ylabel):
    _, a_fit = popt

    plt.figure(figsize=(6,4))
    plt.rcParams.update({'font.size': 20})
    plt.loglog(10**log_x, 10**log_y, 'o', label='IsoFLOPs data')
    C_plot = np.logspace(log_x.min(), np.log10(1e24), 200)
    plt.loglog(C_plot, 10**linear_law(np.log10(C_plot), *popt), '-', label=f'Fit: a={a_fit:.3f}')
    plt.scatter([1e23,1e24],
                [10**linear_law(np.log10(1e23),*popt), 10**linear_law(np.log10(1e24),*popt)],
                c='red', zorder=5, label='Extrapolation')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()

PARAM_CURVE = True
if PARAM_CURVE:
    log_budgets, log_optimal_params,_ = parse_data()
    popt, _ = fit_curve(linear_law, log_budgets, log_optimal_params)
    plot_curve(popt, log_budgets, log_optimal_params, "Compute budget C (FLOPs)", "Optimal parameters (Nₒₚₜ)")
else:
    # optimal data = C/6N
    log_budgets, _, log_optimal_data = parse_data()
    popt, _ = fit_curve(linear_law, log_budgets, log_optimal_data)
    plot_curve(popt, log_budgets, log_optimal_data, "Compute budget C (FLOPs)", "Optimal data (Dₒₚₜ)")