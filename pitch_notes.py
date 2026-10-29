"""Conservative singing segmentation: stabilize pitch before deciding note onsets."""
import numpy as np
from scipy.ndimage import median_filter
from music import Note


def runs(values):
    boundaries = np.flatnonzero(np.r_[True, np.diff(values) != 0, True])
    return [(int(a), int(b), int(values[a])) for a, b in zip(boundaries, boundaries[1:])]


def stabilize_vibrato(midi, valid, rms, dt):
    """Collapse only sustained smooth periodic oscillation around a stable pitch.

    Requires multiple cycles, small trend and no deep energy rearticulation.
    Trills and expressive pitch motion remain ambiguous; preserve source frames.
    """
    result=midi.copy()
    width=max(9,round(.65/dt))
    sums=np.zeros(len(midi)); weights=np.zeros(len(midi))
    for a,b,voice in runs(valid.astype(int)):
        if not voice or b-a<width:
            continue
        offsets=list(range(a,b-width+1,max(1,round(.1/dt))))
        if offsets[-1]!=b-width:
            offsets.append(b-width)
        t=np.arange(width)*dt
        for left in offsets:
            values=midi[left:left+width]
            energy=rms[left:left+width]
            if np.min(energy)<.5*np.median(energy):
                continue
            best=None
            for frequency in np.arange(4,9.1,.5):
                design=np.column_stack([np.ones(width),t-t.mean(),np.sin(2*np.pi*frequency*t),np.cos(2*np.pi*frequency*t)])
                coef=np.linalg.lstsq(design,values,rcond=None)[0]
                error=float(np.mean((design@coef-values)**2))
                if best is None or error<best[0]:
                    best=(error,coef)
            error,coef=best
            amplitude=float(np.hypot(coef[2],coef[3]))
            if .45<amplitude<1.6 and abs(coef[1])<.45 and error<.025 and error<amplitude**2*.07:
                sums[left:left+width]+=coef[0]
                weights[left:left+width]+=1
    mask=weights>0
    result[mask]=sums[mask]/weights[mask]
    return result


def segment_pitch(midi, probability, rms, dt, duration):
    midi = np.asarray(midi, dtype=float)
    probability = np.nan_to_num(np.asarray(probability, dtype=float))
    rms = np.asarray(rms, dtype=float)
    if not (len(midi) == len(probability) == len(rms)):
        raise ValueError('Pitch, confidence and energy frames must align')
    if not len(midi):
        return []
    floor = max(1e-5, float(np.percentile(rms, 95)) * 0.008)
    valid = np.isfinite(midi) & (probability >= 0.12) & (rms >= floor)
    midi = stabilize_vibrato(midi, valid, rms, dt)
    labels = np.full(len(midi), -1, dtype=int)
    window = max(3, int(.085 / dt) | 1)
    # Short median smoothing followed by hysteresis avoids semitone chatter.
    for a, b, voice in runs(valid.astype(int)):
        if not voice:
            continue
        smooth = median_filter(midi[a:b], size=window, mode='nearest')
        pitch = round(float(np.median(smooth[:min(len(smooth), window)])))
        for i, value in enumerate(smooth):
            if abs(value - pitch) > .72:
                pitch = round(float(value))
            labels[a + i] = pitch
    # Rejoin brief excursions/dropouts only when both sides agree. Genuine
    # silences and persistent new pitches remain boundaries.
    for _ in range(2):
        spans = runs(labels)
        for index in range(1, len(spans) - 1):
            a, b, pitch = spans[index]
            left, right = spans[index - 1], spans[index + 1]
            if left[2] < 0 or left[2] != right[2]:
                continue
            if pitch == -1:
                bridge = (b - a) * dt <= .055 and float(np.min(rms[a:b])) > floor
            else:
                bridge = (b - a) * dt <= .11 and (abs(pitch - left[2]) <= 2 or abs(pitch - left[2]) == 12)
            if bridge:
                labels[a:b] = left[2]
    energy = median_filter(rms, size=3, mode='nearest')
    result = []
    for a, b, pitch in runs(labels):
        if pitch < 0 or (b - a) * dt < .09:
            continue
        cuts = [a]
        # A spectral onset alone is not a new sung note. For repeated notes,
        # require a deep energy valley and recovery, away from either edge.
        margin = max(1, round(.13 / dt))
        side = max(2, round(.09 / dt))
        for i in range(a + margin, b - margin):
            if i - cuts[-1] < margin or energy[i] > energy[i - 1] or energy[i] >= energy[i + 1]:
                continue
            left = float(np.max(energy[max(a, i - side):i]))
            right = float(np.max(energy[i + 1:min(b, i + side + 1)]))
            if min(left, right) > floor * 2 and energy[i] < min(left, right) * .5:
                cuts.append(i)
        cuts.append(b)
        for begin, end in zip(cuts, cuts[1:]):
            start = begin * dt
            length = min(duration, end * dt) - start
            if length < .09:
                continue
            confidence = float(np.mean(probability[begin:end]))
            result.append(Note(pitch, 0, 1, start, length, confidence))
    return result


def track_melody(y, sr, cancel_check, progress):
    import librosa
    hop = 256
    dt = hop / sr
    count = 1 + len(y) // hop
    f0 = np.full(count, np.nan)
    probability = np.zeros(count)
    chunk_frames = round(8 / dt)
    context = round(.25 / dt)
    for start in range(0, count, chunk_frames):
        cancel_check()
        end = min(count, start + chunk_frames)
        left, right = max(0, start - context), min(count, end + context)
        segment = y[left * hop:min(len(y), right * hop)]
        if len(segment) < 2048:
            segment = np.pad(segment, (0, 2048 - len(segment)))
        values, voiced, prob = librosa.pyin(segment, sr=sr, fmin=librosa.note_to_hz('C2'),
                                         fmax=librosa.note_to_hz('C7'), frame_length=2048,
                                         hop_length=hop, fill_na=np.nan)
        size = min(end - start, len(values) - (start - left))
        offset = start - left
        f0[start:start + size] = values[offset:offset + size]
        probability[start:start + size] = prob[offset:offset + size]
        progress(58 + 32 * end / count, f'已分析 {min(len(y) / sr, end * dt):.0f} / {len(y) / sr:.0f} 秒')
    cancel_check()
    midi = librosa.hz_to_midi(f0)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0][:count]
    return segment_pitch(midi, probability, rms, dt, len(y) / sr)


def extract_vocal_notes(vocal_path, bpm, beat_origin):
    """Extract notes from an already separated vocal track without PANNs vocal gate."""
    import math
    import librosa
    from music import performance_timing

    bpm = float(bpm)
    beat_origin = float(beat_origin)
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError('bpm must be finite and positive')
    if not math.isfinite(beat_origin):
        raise ValueError('beat_origin must be finite')
    y, sr = librosa.load(str(vocal_path), sr=22050, mono=True)
    if not np.all(np.isfinite(y)):
        raise ValueError('Audio contains non-finite samples')
    if len(y) == 0 or float(np.max(np.abs(y))) == 0.0:
        return []
    raw_notes = track_melody(y, sr, cancel_check=lambda: None, progress=lambda p, s: None)
    return performance_timing(raw_notes, bpm, beat_origin)
