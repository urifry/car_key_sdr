"""Analyze captured car key fob IQ recordings.

Loads a .bin capture, shows the spectrum, detects bursts, demodulates FSK,
estimates baud rate, and extracts bits.

Usage:
    python analyze.py <recording.bin> [--sample-rate 1e6]
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from sdr_common.dsp import (
    extract_channel,
    fm_demodulate,
    detect_baud_rate,
    recover_bits,
)
from sdr_common.detection import detect_bursts, detect_channels


# ---------- Defaults ----------
DEFAULT_SAMPLE_RATE = 1e6
BURST_BW_HZ = 50e3
BURST_THRESHOLD_DB = 8
BURST_MIN_DURATION_S = 0.005


def load_recording(filename):
    """Load complex64 IQ recording from disk."""
    iq = np.fromfile(filename, dtype=np.complex64)
    print(f"Loaded {len(iq)} samples ({len(iq) / DEFAULT_SAMPLE_RATE:.2f}s)")
    return iq


def plot_spectrum(iq, sample_rate, title="Wideband Spectrum"):
    """Plot averaged PSD of the full recording."""
    fft_size = 4096
    num_segments = len(iq) // fft_size
    if num_segments == 0:
        print("Recording too short for spectrum plot.")
        return

    psd = np.zeros(fft_size)
    window = np.hanning(fft_size)
    for i in range(num_segments):
        seg = iq[i * fft_size : (i + 1) * fft_size] * window
        psd += np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
    psd /= num_segments
    psd_db = 10 * np.log10(psd + 1e-12)
    freqs_khz = np.fft.fftshift(
        np.fft.fftfreq(fft_size, 1 / sample_rate)
    ) / 1e3

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(freqs_khz, psd_db)
    ax.set_xlabel("Frequency offset (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


def plot_waterfall(iq, sample_rate, fft_size=1024, title="Waterfall"):
    """Time-frequency waterfall of the full recording."""
    num_rows = len(iq) // fft_size
    if num_rows == 0:
        return
    window = np.hanning(fft_size)
    wf = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        seg = iq[i * fft_size : (i + 1) * fft_size] * window
        wf[i] = 10 * np.log10(
            np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2 + 1e-12
        )

    extent = [
        -sample_rate / 2e3, sample_rate / 2e3,
        num_rows * fft_size / sample_rate, 0,
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wf, aspect="auto", extent=extent, cmap="hot",
              vmin=np.median(wf) - 5, vmax=np.percentile(wf, 99))
    ax.set_xlabel("Frequency offset (kHz)")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    return fig


def analyze_burst(burst_iq, sample_rate, burst_index):
    """Demodulate and analyze a single detected burst."""
    print(f"\n{'='*50}")
    print(f"Burst #{burst_index}")
    print(f"  Samples: {len(burst_iq)}")
    print(f"  Duration: {len(burst_iq) / sample_rate * 1000:.1f} ms")

    # Channel extraction — signal is already near baseband
    # Use a moderate target rate for good time resolution
    target_rate = min(sample_rate, 100e3)
    ch_iq, ch_rate = extract_channel(burst_iq, sample_rate,
                                     channel_offset=0,
                                     target_rate=target_rate)

    # FM demodulate
    demod = fm_demodulate(ch_iq)

    # Detect baud rate
    baud = detect_baud_rate(demod, ch_rate)
    print(f"  Estimated baud rate: {baud:.0f} baud")

    # Recover bits
    bits = recover_bits(demod, ch_rate, baud)
    print(f"  Recovered {len(bits)} bits")
    if len(bits) > 0:
        bit_str = "".join(str(b) for b in bits[:200])
        print(f"  First 200 bits: {bit_str}")
        if len(bits) > 200:
            print(f"  ... ({len(bits) - 200} more)")

    # Plot demodulated signal and bit sampling
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 1. Raw IQ magnitude
    t_ms = np.arange(len(burst_iq)) / sample_rate * 1000
    axes[0].plot(t_ms, np.abs(burst_iq), linewidth=0.5)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Burst #{burst_index} — IQ Magnitude")
    axes[0].grid(True, alpha=0.3)

    # 2. FM demodulated signal
    t_demod_ms = np.arange(len(demod)) / ch_rate * 1000
    axes[1].plot(t_demod_ms, demod, linewidth=0.5)
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_ylabel("FM Demod")
    axes[1].set_title(f"FM Demodulated — baud={baud:.0f}")
    axes[1].grid(True, alpha=0.3)

    # 3. Spectrum of the burst
    fft_size = min(4096, len(burst_iq))
    window = np.hanning(fft_size)
    seg = burst_iq[:fft_size] * window
    spec_db = 10 * np.log10(
        np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2 + 1e-12
    )
    freqs_khz = np.fft.fftshift(
        np.fft.fftfreq(fft_size, 1 / sample_rate)
    ) / 1e3
    axes[2].plot(freqs_khz, spec_db, linewidth=0.5)
    axes[2].set_xlabel("Frequency offset (kHz)")
    axes[2].set_ylabel("Power (dB)")
    axes[2].set_title("Burst Spectrum")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return {"baud_rate": baud, "bits": bits, "demod": demod, "ch_rate": ch_rate}


def main():
    parser = argparse.ArgumentParser(description="Analyze car key IQ recording")
    parser.add_argument("recording", help="Path to .bin IQ recording")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE,
                        help=f"Sample rate in Hz (default: {DEFAULT_SAMPLE_RATE:.0f})")
    args = parser.parse_args()

    sample_rate = args.sample_rate
    iq = load_recording(args.recording)

    # Wideband overview
    print("\n--- Wideband Spectrum ---")
    plot_spectrum(iq, sample_rate, title="Full Recording Spectrum")

    print("\n--- Waterfall ---")
    plot_waterfall(iq, sample_rate, title="Full Recording Waterfall")

    # Detect active channels
    channels = detect_channels(iq, sample_rate, threshold_db=8,
                               min_spacing_hz=10e3)
    if channels:
        print(f"\nDetected channels (offset from center):")
        for ch in channels:
            print(f"  {ch / 1e3:+.1f} kHz")
    else:
        print("\nNo strong channels detected — signal may be at center freq.")

    # Detect bursts at baseband (signal expected near center)
    channel_offset = channels[0] if channels else 0
    print(f"\nSearching for bursts at offset {channel_offset / 1e3:+.1f} kHz...")
    bursts = detect_bursts(
        iq, sample_rate,
        channel_offset=channel_offset,
        bw=BURST_BW_HZ,
        threshold_db=BURST_THRESHOLD_DB,
        min_duration_s=BURST_MIN_DURATION_S,
    )

    print(f"Found {len(bursts)} burst(s)")

    results = []
    for i, (start, end) in enumerate(bursts):
        # Add some margin around the burst
        margin = int(sample_rate * 0.005)  # 5 ms margin
        s = max(0, start - margin)
        e = min(len(iq), end + margin)
        burst_iq = iq[s:e]
        result = analyze_burst(burst_iq, sample_rate, burst_index=i)
        results.append(result)

    # Compare bursts if multiple found
    if len(results) >= 2:
        print(f"\n{'='*50}")
        print("Burst Comparison")
        print(f"{'='*50}")
        for i, r in enumerate(results):
            print(f"  Burst #{i}: baud={r['baud_rate']:.0f}, bits={len(r['bits'])}")

        # Check if bit patterns repeat (same button press)
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                bi = results[i]["bits"]
                bj = results[j]["bits"]
                min_len = min(len(bi), len(bj))
                if min_len > 0:
                    match = np.sum(bi[:min_len] == bj[:min_len]) / min_len
                    print(f"  Burst #{i} vs #{j}: {match * 100:.1f}% bit match "
                          f"(over {min_len} bits)")

    plt.show()


if __name__ == "__main__":
    main()
