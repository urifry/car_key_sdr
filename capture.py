"""Capture car key fob signal at 433.9 MHz using RTL-SDR.

Records wideband IQ around the ISM band and detects short bursts
from key fob transmissions. Press Ctrl+C to stop recording.

Usage:
    python capture.py [--duration SECONDS] [--gain GAIN_DB]
"""

import argparse
import datetime
import sys

import numpy as np
from rtlsdr import RtlSdr

from sdr_common.detection import detect_bursts

# ---------- Radio parameters ----------
CENTER_FREQ_HZ = 433.9e6
SAMPLE_RATE = 1e6       # 1 MHz — covers ~500 kHz each side of center
GAIN_DB = 40             # reasonable starting gain for 433 MHz fob signals
CHUNK_SAMPLES = 262144   # ~262 ms per chunk at 1 MHz

# ---------- Burst detection ----------
BURST_BW_HZ = 50e3          # expected signal bandwidth for key fobs
BURST_THRESHOLD_DB = 8       # energy above noise floor
BURST_MIN_DURATION_S = 0.005 # key fob bursts can be very short (~5 ms+)


def make_filename(prefix="iq"):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{CENTER_FREQ_HZ / 1e6:.1f}MHz_{SAMPLE_RATE / 1e6:.1f}Msps.bin"


def configure_sdr(gain_db=GAIN_DB):
    """Open and configure the RTL-SDR dongle."""
    sdr = RtlSdr()
    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = CENTER_FREQ_HZ
    if gain_db == 0:
        sdr.gain = "auto"
        print(f"Gain: auto (AGC)")
    else:
        sdr.gain = gain_db
        print(f"Gain: {sdr.gain} dB")
    print(f"Center freq: {sdr.center_freq / 1e6:.3f} MHz")
    print(f"Sample rate: {sdr.sample_rate / 1e6:.1f} Msps")
    return sdr


def capture_continuous(sdr, duration_s, outfile):
    """Stream IQ to disk for a fixed duration and report detected bursts."""
    total_samples = int(duration_s * SAMPLE_RATE)
    collected = 0
    all_bursts = []
    chunk_index = 0

    print(f"\nRecording {duration_s}s to {outfile} — press key fob now...")
    print("-" * 50)

    with open(outfile, "wb") as f:
        while collected < total_samples:
            n = min(CHUNK_SAMPLES, total_samples - collected)
            iq = sdr.read_samples(n)
            iq = iq.astype(np.complex64)
            iq.tofile(f)

            # Check for bursts in this chunk
            bursts = detect_bursts(
                iq, SAMPLE_RATE,
                channel_offset=0,
                bw=BURST_BW_HZ,
                threshold_db=BURST_THRESHOLD_DB,
                min_duration_s=BURST_MIN_DURATION_S,
            )
            if bursts:
                for start, end in bursts:
                    abs_start = collected + start
                    abs_end = collected + end
                    dur_ms = (end - start) / SAMPLE_RATE * 1000
                    t_sec = abs_start / SAMPLE_RATE
                    print(f"  BURST @ {t_sec:.3f}s  duration={dur_ms:.1f}ms")
                    all_bursts.append((abs_start, abs_end))

            collected += len(iq)
            elapsed = collected / SAMPLE_RATE
            sys.stdout.write(f"\r  {elapsed:.1f}s / {duration_s:.1f}s")
            sys.stdout.flush()
            chunk_index += 1

    print(f"\n\nDone. {len(all_bursts)} burst(s) detected.")
    print(f"Saved: {outfile}")
    return all_bursts


def main():
    parser = argparse.ArgumentParser(description="Capture car key fob signal")
    parser.add_argument("--duration", type=float, default=10,
                        help="Recording duration in seconds (default: 10)")
    parser.add_argument("--gain", type=float, default=GAIN_DB,
                        help=f"Gain in dB, 0 for AGC (default: {GAIN_DB})")
    args = parser.parse_args()

    sdr = configure_sdr(gain_db=args.gain)
    outfile = make_filename()

    try:
        capture_continuous(sdr, args.duration, outfile)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        sdr.close()


if __name__ == "__main__":
    main()
