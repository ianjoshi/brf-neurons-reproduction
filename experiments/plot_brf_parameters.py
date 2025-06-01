import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Default delta value for divergence boundary calculation
delta = 0.01

def load_model(model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        model = torch.load(model_path, map_location=torch.device("cpu"))
        print(f"Successfully loaded model: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

def divergence_boundary(omega):
    return (-1 + torch.sqrt(1 - (delta * omega) ** 2)) / delta

def extract_brf_parameters(model_state_dict):
    omega = model_state_dict["hidden.omega"].detach().cpu().numpy()
    b_offset = model_state_dict["hidden.b_offset"].detach().cpu().numpy()
    p_w = divergence_boundary(torch.tensor(omega)).numpy()
    b_c = p_w - b_offset

    return {
        "omega": omega,
        "b_offset": b_offset,
        "p_w": p_w,
        "b_c": b_c
    }

def plot_brf_parameters(base_dir, initial_model_filename, optimized_model_filename,
                        output_dir, dataset_name,
                        initial_color="#777777", optimized_color="#EE4B2B"):
    """
    Generate and save a plot of ω vs b_c for BRF models with clean solid dots.

    Parameters:
        base_dir (str): Directory with the model files.
        initial_model_filename (str): Initial model file (.pt).
        optimized_model_filename (str): Optimized model file (.pt).
        output_dir (str): Where to save the plot.
        dataset_name (str): Label for legend and filename.
        initial_color (str): Hex color for initial dots.
        optimized_color (str): Hex color for optimized dots.
    """
    initial_model_path = os.path.join(base_dir, initial_model_filename)
    optimized_model_path = os.path.join(base_dir, optimized_model_filename)

    initial_model = load_model(initial_model_path)
    optimized_model = load_model(optimized_model_path)

    initial_params = extract_brf_parameters(initial_model['model_state_dict'])
    optimized_params = extract_brf_parameters(optimized_model['model_state_dict'])

    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(4, 4))

    # Initial model points
    plt.scatter(initial_params["omega"], initial_params["b_c"],
                label="Initial", color=initial_color, edgecolors='none',
                alpha=0.6, s=20, marker='o', zorder=2)

    # Optimized model points
    plt.scatter(optimized_params["omega"], optimized_params["b_c"],
                label="Optimized", color=optimized_color, edgecolors='none',
                alpha=0.6, s=20, marker='o', zorder=3)

    # Divergence boundary
    omega_range = np.linspace(
        0,
        max(np.max(initial_params["omega"]), np.max(optimized_params["omega"])) + 1,
        300
    )
    with np.errstate(invalid="ignore"):
        p_w_curve = (-1 + np.sqrt(1 - (delta * omega_range) ** 2)) / delta
    p_w_curve = np.where(np.isreal(p_w_curve), p_w_curve, np.nan)
    plt.plot(omega_range, p_w_curve, 'k--', linewidth=1.5, zorder=1)

    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$b_c = p(\omega) - b'$")
    plt.grid(True)
    plt.legend(title=dataset_name, loc="lower right", fontsize=10)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{dataset_name}_brf_param_plot.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Plot saved to {out_path}")

# === Example Usage ===
if __name__ == "__main__":
    base_dir = "experiments/smnist/models"
    initial_model_file = "SMNIST_BRF_init.pt"
    optimized_model_file = "SMNIST_BRF.pt"
    output_dir = "experiments/plots"
    dataset_name = "SMNIST"

    # Clean solid dot colors
    initial_color = "#a8240c"
    optimized_color = "#ff866d"

    plot_brf_parameters(
        base_dir=base_dir,
        initial_model_filename=initial_model_file,
        optimized_model_filename=optimized_model_file,
        output_dir=output_dir,
        dataset_name=dataset_name,
        initial_color=initial_color,
        optimized_color=optimized_color
    )
