"""Sound for the promo video, made from scratch so there is nothing to license.

Quiet and airy, like the site: a soft pad in D major that moves through four
chords, a shimmer under the splash and a small bell as its logo lands, a soft
tap on every click, a rising glide while the speed slider moves, sparse plucked
notes while the network trains (busier as it speeds up, climbing as it learns),
and a bell chord on the closing card.

    python scripts/promo_audio.py events.json out.wav

events.json: {"duration": 15, "splash": 0.2, "logo": 1.9, "taps": [4.5, 6.1],
              "drag": [6.4, 7.6], "train": [6.1, 13.0], "card": 13.0}
"""

import json
import sys

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 48000
rng = np.random.default_rng(7)


def note(name: str) -> float:
    """'D5' -> Hz."""
    pcs = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    return 440.0 * 2 ** ((pcs[name[:-1]] + 12 * (int(name[-1]) + 1) - 69) / 12)


def lowpass(x, hz, order=2):
    return sosfilt(butter(order, hz, "low", fs=SR, output="sos"), x)


def highpass(x, hz, order=2):
    return sosfilt(butter(order, hz, "high", fs=SR, output="sos"), x)


def add(buf, at, sig, pan=0.0, gain=1.0):
    """Mix a mono signal into the stereo buffer at `at` seconds, panned -1..1."""
    i = int(at * SR)
    if i >= buf.shape[1] or i + len(sig) <= 0:
        return
    s = sig[max(0, -i): buf.shape[1] - i] * gain
    i = max(0, i)
    buf[0, i:i + len(s)] += s * np.sqrt((1 - pan) / 2)
    buf[1, i:i + len(s)] += s * np.sqrt((1 + pan) / 2)


def env(n, attack, release, total=None):
    t = np.arange(n) / SR
    total = total or n / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    r = np.clip((total - t) / max(release, 1e-4), 0, 1)
    return a * r


def pad(chord, dur):
    """Soft detuned sine and triangle voices, slowly breathing."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for k, name in enumerate(chord):
        f = note(name)
        for det in (-0.03, 0.03):
            ph = rng.uniform(0, 2 * np.pi)
            w = 2 * np.pi * f * (1 + det / 100) * t + ph
            tri = 2 / np.pi * np.arcsin(np.sin(w))
            out += (0.8 * np.sin(w) + 0.2 * tri) * (1 + 0.08 * np.sin(2 * np.pi * (0.11 + 0.03 * k) * t))
    return highpass(lowpass(out / (len(chord) * 2), 1600), 140) * env(n, 1.4, 1.6)


def bell(f, dur=2.4):
    """Sine partials with their own decays: a soft glass bell."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    parts = [(1, 1.0, 1.6), (2.0, 0.35, 0.9), (3.01, 0.15, 0.5), (4.2, 0.07, 0.3)]
    s = sum(a * np.sin(2 * np.pi * f * m * t) * np.exp(-t / d) for m, a, d in parts)
    return s * env(n, 0.004, 0.3)


def pluck(f, dur=0.9):
    """Karplus-Strong string: a soft, short pluck."""
    n = int(dur * SR)
    period = int(SR / f)
    buf = lowpass(rng.uniform(-1, 1, period), 3000)
    out = np.zeros(n)
    for i in range(n):
        out[i] = buf[i % period]
        buf[i % period] = 0.497 * (buf[i % period] + buf[(i + 1) % period])
    return lowpass(out, 4000) * env(n, 0.002, 0.25)


def tap():
    """A soft click: a short high blip over a tiny low thump."""
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    blip = np.sin(2 * np.pi * 1850 * t) * np.exp(-t / 0.012)
    thump = np.sin(2 * np.pi * 180 * t) * np.exp(-t / 0.035)
    return 0.55 * blip + 0.45 * thump


def shimmer(dur):
    """Airy filtered noise that swells and fades."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = highpass(lowpass(rng.normal(0, 1, n), 7000), 2500)
    return x * np.sin(np.pi * t / dur) ** 2 * 0.25


def glide(f0, f1, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 * (f1 / f0) ** (t / dur)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.08, 0.15) * 0.5


def reverb(x, seconds=2.2, wet=0.32):
    """Convolution with a decaying stereo noise tail: a soft room."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros_like(x)
    for c in range(2):
        ir = rng.normal(0, 1, n) * np.exp(-t / (seconds / 5))
        ir = lowpass(ir, 6000)
        ir /= np.sqrt(np.sum(ir ** 2))
        out[c] = fftconvolve(x[c], ir)[: x.shape[1]]
    return (1 - wet) * x + wet * out


def main(events_path, out_path):
    ev = json.load(open(events_path))
    dur = ev["duration"]
    mix = np.zeros((2, int(dur * SR)))
    music = np.zeros_like(mix)

    # the pad: D major, four chords, the last resolving under the card
    card = ev["card"]
    chords = [(0.0, ["A3", "D4", "E4", "F#4"]), (card * 0.36, ["F#3", "B3", "D4", "A4"]),
              (card * 0.68, ["G3", "B3", "D4", "F#4"]), (card - 0.4, ["A3", "D4", "F#4", "E5"])]
    for k, (start, chord) in enumerate(chords):
        end = chords[k + 1][0] + 1.2 if k + 1 < len(chords) else dur
        add(music, start, pad(chord, end - start), gain=0.075)

    # the splash: a shimmer, and a bell as the logo lands
    add(music, ev["splash"], shimmer(2.6), gain=0.5)
    add(music, ev["logo"], bell(note("A5")), pan=-0.2, gain=0.16)
    add(music, ev["logo"] + 0.09, bell(note("D6")), pan=0.2, gain=0.11)

    # every tap
    for t in ev["taps"]:
        add(mix, t, tap(), gain=0.32)

    # the speed slider: a quiet rising glide while it moves
    d0, d1 = ev["drag"]
    add(music, d0, glide(520, 880, d1 - d0 + 0.1), gain=0.06)

    # training: plucked notes, sparse at first, busier and higher as it learns
    scale = ["D", "E", "F#", "A", "B"]
    t0, t1 = ev["train"]
    t = t0 + 0.15
    while t < t1 - 0.2:
        progress = (t - t0) / (t1 - t0)
        speed = 3.0 if t < d1 else 6.5           # notes per second, before and after the speed-up
        octave = 4 + int(progress * 1.6 + rng.uniform(0, 1.2))
        name = scale[rng.integers(0, len(scale))] + str(min(octave, 6))
        add(music, t, pluck(note(name)), pan=rng.uniform(-0.6, 0.6), gain=0.16 + 0.08 * progress)
        t += rng.exponential(1 / speed) + 0.05

    # the card: a bell chord
    for k, name in enumerate(["D5", "F#5", "A5", "D6"]):
        add(music, card + 0.05 + 0.06 * k, bell(note(name), 3.0), pan=(k - 1.5) / 2.5, gain=0.12)

    out = reverb(music) + reverb(mix, 0.8, 0.15)
    # in the background, not silent: peaks at -3 dB (about -18 LUFS), with short fades at both ends
    out *= 10 ** (-3 / 20) / np.max(np.abs(out))
    n = out.shape[1]
    fade = np.ones(n)
    fade[: int(0.25 * SR)] = np.linspace(0, 1, int(0.25 * SR))
    fade[-int(0.8 * SR):] = np.linspace(1, 0, int(0.8 * SR))
    out *= fade

    pcm = (np.clip(out.T, -1, 1) * 32767).astype("<i2")
    import wave
    with wave.open(out_path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
