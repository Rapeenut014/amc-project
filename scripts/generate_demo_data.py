"""
Demo Data Generator for AMC Project
Generates synthetic I/Q signals for testing without the full RadioML dataset
"""

import numpy as np
import h5py
import os
from pathlib import Path


# Modulation classes from RadioML 2018
CLASSES = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK',
    '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK',
    '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC',
    'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
]

# SNR values from -20 to +30 dB (step 2)
SNR_VALUES = list(range(-20, 32, 2))


def generate_bpsk(n_samples, length=1024):
    """Generate BPSK signal"""
    symbols = np.random.choice([-1, 1], size=(n_samples, length))
    I = symbols
    Q = np.zeros_like(I)
    return np.stack([I, Q], axis=-1)


def generate_qpsk(n_samples, length=1024):
    """Generate QPSK signal"""
    symbols = np.random.choice([1, -1], size=(n_samples, length, 2)) / np.sqrt(2)
    return symbols


def generate_8psk(n_samples, length=1024):
    """Generate 8PSK signal"""
    phases = np.random.choice(8, size=(n_samples, length))
    angles = phases * (2 * np.pi / 8)
    I = np.cos(angles)
    Q = np.sin(angles)
    return np.stack([I, Q], axis=-1)


def generate_qam(n_samples, length=1024, order=16):
    """Generate QAM signal"""
    m = int(np.sqrt(order))
    levels = np.arange(-m+1, m, 2)
    I = np.random.choice(levels, size=(n_samples, length)) / (m-1)
    Q = np.random.choice(levels, size=(n_samples, length)) / (m-1)
    return np.stack([I, Q], axis=-1)


def generate_fm(n_samples, length=1024):
    """Generate FM signal (simplified)"""
    t = np.linspace(0, 1, length)
    carrier_freq = 10
    mod_freq = 2
    signals = []
    for _ in range(n_samples):
        mod_signal = np.sin(2 * np.pi * mod_freq * t)
        phase = 2 * np.pi * carrier_freq * t + 5 * np.cumsum(mod_signal) / length
        I = np.cos(phase)
        Q = np.sin(phase)
        signals.append(np.stack([I, Q], axis=-1))
    return np.array(signals)


def generate_random_signal(n_samples, length=1024):
    """Generate random signal for other modulation types"""
    return np.random.randn(n_samples, length, 2) * 0.5


def add_noise(signal, snr_db):
    """Add AWGN noise to signal"""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise


def generate_demo_dataset(output_path, samples_per_class=100, length=1024):
    """
    Generate demo dataset with synthetic signals
    
    Args:
        output_path: Path to save HDF5 file
        samples_per_class: Number of samples per (class, SNR) combination
        length: Signal length
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    n_classes = len(CLASSES)
    n_snr = len(SNR_VALUES)
    total_samples = n_classes * n_snr * samples_per_class
    
    print(f"🔧 Generating demo dataset...")
    print(f"   Classes: {n_classes}")
    print(f"   SNR values: {n_snr} ({min(SNR_VALUES)} to {max(SNR_VALUES)} dB)")
    print(f"   Samples per (class, SNR): {samples_per_class}")
    print(f"   Total samples: {total_samples:,}")
    
    # Generators for each modulation type
    generators = {
        'BPSK': lambda n: generate_bpsk(n, length),
        'QPSK': lambda n: generate_qpsk(n, length),
        '8PSK': lambda n: generate_8psk(n, length),
        '16QAM': lambda n: generate_qam(n, length, 16),
        '64QAM': lambda n: generate_qam(n, length, 64),
        'FM': lambda n: generate_fm(n, length),
    }
    
    # Pre-allocate arrays
    X = np.zeros((total_samples, length, 2), dtype=np.float32)
    Y = np.zeros((total_samples, n_classes), dtype=np.float32)  # One-hot
    Z = np.zeros((total_samples, 1), dtype=np.float32)
    
    idx = 0
    for class_idx, class_name in enumerate(CLASSES):
        gen = generators.get(class_name, lambda n: generate_random_signal(n, length))
        
        for snr in SNR_VALUES:
            # Generate clean signal
            signal = gen(samples_per_class)
            
            # Add noise
            signal = add_noise(signal, snr)
            
            # Store
            X[idx:idx+samples_per_class] = signal
            Y[idx:idx+samples_per_class, class_idx] = 1  # One-hot encoding
            Z[idx:idx+samples_per_class, 0] = snr
            
            idx += samples_per_class
        
        print(f"   ✓ {class_name} done")
    
    # Shuffle
    print("   Shuffling...")
    perm = np.random.permutation(total_samples)
    X = X[perm]
    Y = Y[perm]
    Z = Z[perm]
    
    # Save to HDF5
    print(f"   Saving to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('X', data=X, compression='gzip')
        f.create_dataset('Y', data=Y, compression='gzip')
        f.create_dataset('Z', data=Z, compression='gzip')
    
    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"\n✅ Demo dataset created!")
    print(f"   Path: {output_path}")
    print(f"   Size: {file_size:.2f} MB")
    print(f"   Shape: X={X.shape}, Y={Y.shape}, Z={Z.shape}")
    
    return output_path


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate demo data for AMC project')
    parser.add_argument('--output', '-o', type=str, 
                        default='archive/demo_data.hdf5',
                        help='Output HDF5 file path')
    parser.add_argument('--samples', '-n', type=int, default=100,
                        help='Samples per (class, SNR) combination')
    parser.add_argument('--length', '-l', type=int, default=1024,
                        help='Signal length')
    
    args = parser.parse_args()
    
    generate_demo_dataset(args.output, args.samples, args.length)


if __name__ == '__main__':
    main()
