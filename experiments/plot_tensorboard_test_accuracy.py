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
        tag = run.get('tag', 'accuracy/test')
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
    base_dir = "experiments/ecg/runs"
    dataset_name = "ECG"
    output_dir = "experiments/plots"
    epochs = 400

    runs = [
        {
            "folder": "May24_18-38-50_Inaesh-HP-ZBook7303_Adam(0.1),script-bw,NLL,LinearLR,no_gc,4,36,6,bs=16,ep=400,RF(omega3.0,5.0,b0.1,1.0)LI(20.0,1.0)",
            "label": "RF",
            "color": "#4a9600"
        },
        {
            "folder": "May25_00-24-03_Inaesh-HP-ZBook8977_Adam(0.1),script-bw,NLL,LinearLR,no_gc,4,36,6,bs=16,ep=400,BRF(omega3.0,5.0,b0.1,1.0)LI(20.0,1.0)",
            "label": "BRF",
            "color": "#EE4B2B"
        },
        {
            "folder": "May25_11-35-53_Inaesh-HP-ZBook8709_Adam(0.05),NLL,LinearLR,no_gc,RSNN(4,36,6,sub_seq_10,bs_4,ep_400,h_o_bias(True)),ALIF(tau_m(20.0,0.5),tau_a(7.0,0.2),linMask_0.0)LI(tau_m(20.0,0.5))",
            "label": "ALIF",
            "color": "#0017ff"
        }
    ]

    plot_tensorboard_runs(base_dir, runs, dataset_name, output_dir, epochs)
