import os
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def load_tensorboard_scalars(log_dir, tag="accuracy/test"):
    """
    Load scalar values from a TensorBoard log directory.
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # print("Available tags:")
    # print(ea.Tags()['scalars'])

    if tag not in ea.Tags()['scalars']:
        return None, None
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return np.array(steps), np.array(values)

def smooth_curve(values, window=10):
    """
    Smooth the curve using a moving average with edge padding.
    """
    if len(values) < window:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode='edge')
    smoothed = np.convolve(padded, np.ones(window)/window, mode='valid')
    return smoothed[:len(values)]  # Ensure length consistency

def plot_tensorboard_runs(base_dir, runs, dataset_name, output_dir, epochs=None):
    """
    Generate and save a plot of accuracy vs. epochs for multiple TensorBoard runs.
    
    Parameters:
    - base_dir: directory containing TensorBoard folders
    - runs: list of dicts with 'folder', 'label', 'color', and optionally 'tag'
    - dataset_name: name of the dataset
    - output_dir: where to save the generated plot
    - epochs: optional, number of epochs to limit the x-axis (defaults to all available epochs)
    """
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(6, 6))

    for run in runs:
        log_folder = os.path.join(base_dir, run['folder'])
        tag = run.get('tag', 'Accuracy/test')
        steps, values = load_tensorboard_scalars(log_folder, tag)
        if steps is None or values is None:
            print(f"Skipping {run['label']} — No valid scalar '{tag}' found.")
            continue

        smoothed = smooth_curve(values)
        plt.plot(steps, smoothed, label=run['label'], color=run['color'], linewidth=1.2)
        plt.fill_between(steps, smoothed - 0.3 * np.std(values), smoothed + 0.3 * np.std(values),
                         alpha=0.2, color=run['color'], linewidth=0)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Test accuracy", fontsize=12)
    plt.grid(True)
    plt.legend(title=dataset_name, fontsize=12, loc='lower right')
    plt.ylim(0, 100)
    if epochs is not None:
        plt.xlim(0, epochs)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{dataset_name}_test_accuracy_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    base_dir = "experiments/smnist/runs"
    dataset_name = "SMNIST"
    output_dir = "experiments/plots"
    epochs = 100

    runs = [
        {
            "folder": "May22_15-25-38_DESKTOP-JELMER8630_Adam(0.1),NLL,script-bw,LinLR,LL(False),PERMUTED(False),1,256,10,bs=256,ep=100,BRF_omega15.0_50.0b0.1_1.0,LI20.0_5.0",
            "label": "RF",
            "color": "#4a9600"
        },
        {
            "folder": "May22_17-36-02_DESKTOP-JELMER4705_Adam(0.1),NLL,script-bw,LinLR,LL(False),PERMUTED(False),1,256,10,bs=256,ep=100,BRF_omega15.0_50.0b0.1_1.0,LI20.0_5.0",
            "label": "BRF",
            "color": "#EE4B2B"
        },
        {
            "folder": "May26_08-54-50_DESKTOP-JELMER3047_Adam(0.001),PERMUTED(False),LinearLR,NLL,LL(True),RSNN(1,256,10,bs_256,ep_100,h_o_bias),ALIF(tau_m(20.0,5.0),tau_a(200.0,50.0),linearMask(0.0))LI(tau_m(20.0,5.0))",
            "label": "ALIF",
            "color": "#0017ff"
        }
    ]

    plot_tensorboard_runs(base_dir, runs, dataset_name, output_dir, epochs)

