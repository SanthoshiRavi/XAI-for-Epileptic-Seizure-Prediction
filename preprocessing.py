import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def read_mat_file(mat_path):
    """Reads EEG .mat file and returns the EEG signal array."""
    mat = scipy.io.loadmat(mat_path)
    signal = None
    for key in mat:
        if not key.startswith("__") and isinstance(mat[key], np.ndarray):
            signal = mat[key].squeeze()
            break
    if signal is None:
        raise ValueError("No valid EEG signal found in .mat file.")
    return signal

def generate_spectrogram(signal, save_path_clean, save_path_detailed, fs=200):
    """Generates both a clean and annotated spectrogram with consistent dimensions."""
    f, t, Sxx = spectrogram(signal, fs=fs)
    Sxx_log = 10 * np.log10(Sxx + 1e-8)

    # Clean spectrogram for overlay (square aspect ratio)
    plt.figure(figsize=(3.2, 3.2), dpi=100)
    plt.axis('off')
    plt.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap='viridis')
    plt.tight_layout(pad=0)
    plt.savefig(save_path_clean, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Detailed spectrogram with axes and colorbar (consistent with clean dimensions)
    plt.figure(figsize=(6, 4), dpi=100)
    ax = plt.gca()
    im = ax.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('EEG Spectrogram')
    cbar = plt.colorbar(im, ax=ax, label='Power [dB]')
    cbar.ax.tick_params(labelsize=8)  # Smaller colorbar font
    plt.tight_layout()
    plt.savefig(save_path_detailed, bbox_inches='tight')
    plt.close()

    # Return spectrogram bounds for proper GradCAM overlay
    return {'xlim': ax.get_xlim(), 'ylim': ax.get_ylim()}