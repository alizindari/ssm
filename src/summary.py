import numpy as np

from experiments import _param_count
from plotting import get_transition_epoch


def print_summary_table(main_results, cfg):
    """Print a summary table of main experiment results."""
    hidden_dims = cfg["main_hidden_dims"]
    main_seeds = cfg["main_seeds"]

    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    header = (
        f"{'hidden_dim':>10} | {'n_params':>8} | {'transition_epoch':>20} | "
        f"{'final_train_acc':>15} | {'final_test_acc16':>16} | {'peak_gate_sel':>14}"
    )
    print(header)
    print("-" * len(header))

    for hd in hidden_dims:
        n_params = _param_count(hd, cfg["model"])

        trans_epochs = []
        final_train = []
        final_test = []
        peak_sel = []

        for seed in range(main_seeds):
            data = main_results[hd][seed]
            te = get_transition_epoch(data)
            if te is not None:
                trans_epochs.append(te)
            final_train.append(data["train_acc"][-1])
            final_test.append(data["test_acc"][-1])
            peak_sel.append(max(data["gate_selectivity"]))

        if trans_epochs:
            te_str = f"{np.mean(trans_epochs):.0f} ± {np.std(trans_epochs):.0f}"
        else:
            te_str = "never"

        print(
            f"{hd:>10} | {n_params:>8} | {te_str:>20} | "
            f"{np.mean(final_train):>15.4f} | {np.mean(final_test):>16.4f} | "
            f"{np.mean(peak_sel):>14.4f}"
        )
