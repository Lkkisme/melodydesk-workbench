"""High-Fidelity Acoustic Guitar Sound Simulator (SF2 + 6-String Extended Karplus-Strong Physical Waveguide + Wood Body Formant Filter) & Real-Vocal Unplugged Mixer."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import math
import soundfile as sf
from scipy.signal import lfilter, iirpeak, butter, sosfilt, resample_poly
from .chord_voicing import _integer


def _sample_rate(sr):
    _integer(sr,8001,float('inf'),'sample rate')


def _pluck(pitch, string, duration_samples, sr, velocity, technique, seed):
    frequency = 440.0*2.0**((pitch-69)/12.0)
    omega = 2*np.pi*frequency/sr
    alpha = 0.18+0.30*(string-1)/5
    dispersion_a = 0.95-0.25*(string-1)/5
    z = np.exp(-1j*omega)
    response = ((1-alpha)+alpha*z)*(dispersion_a+z)/(1+dispersion_a*z)
    remaining = sr/frequency-(-np.angle(response)/omega)
    delay_length = int(np.floor(remaining))
    if delay_length < 1:
        raise ValueError('Pitch is too high for the sample rate')
    fractional = remaining-delay_length
    fractional_a = None
    if fractional > 1e-12:
        ratio = np.tan(omega*fractional/2)/np.tan(omega/2)
        fractional_a = (1-ratio)/(1+ratio)
    rng = np.random.default_rng(seed)
    phase = np.arange(delay_length,dtype=np.float64)/delay_length
    excitation = np.zeros(delay_length,dtype=np.float64)
    for harmonic in range(1,min(delay_length//2, int(sr/(2*frequency)))+1):
        excitation += np.sin(np.pi*harmonic*0.18)/harmonic**2 * np.cos(2*np.pi*harmonic*phase)
    if technique not in {'H','P'}:
        noise = rng.standard_normal(delay_length)
        excitation += 0.035*(noise-np.roll(noise,round(0.18*delay_length)))
    excitation -= excitation.mean()
    rms = np.sqrt(np.mean(excitation**2))
    if rms > 0:
        excitation /= rms
    output = np.zeros(duration_samples,dtype=np.float64)
    initial = min(delay_length,duration_samples)
    output[:initial] = excitation[:initial]
    fir_state = np.zeros(1)
    dispersion_state = np.zeros(1)
    fractional_state = np.zeros(1)
    gain = 10**(-3/(frequency*(2.8+0.3*(string-1))))
    for start in range(delay_length,duration_samples,delay_length):
        end = min(start+delay_length,duration_samples)
        block = output[start-delay_length:end-delay_length]
        block, fir_state = lfilter([1-alpha,alpha],[1],block,zi=fir_state)
        block, dispersion_state = lfilter([dispersion_a,1],[1,dispersion_a],block,zi=dispersion_state)
        if fractional_a is not None:
            block, fractional_state = lfilter([fractional_a,1],[1,fractional_a],block,zi=fractional_state)
        output[start:end] = gain*block
    t = np.arange(duration_samples,dtype=np.float64)/sr
    output *= np.exp(-t*(16 if technique == 'PM' else 16/3.5))
    if technique == 'PM':
        output = sosfilt(butter(2,700,fs=sr,output='sos'),output)
    attack = min(duration_samples,max(1,round(sr*(0.015 if technique in {'H','P'} else 0.002))))
    output[:attack] *= np.sin(np.linspace(0,np.pi/2,attack))**2
    release = min(duration_samples,max(1,round(0.005*sr)))
    output[-release:] *= np.linspace(1,0,release)**2
    return output*(velocity/127.0)


_SLAP_BODY_GAIN = 0.45
_SLAP_BRUSH_GAIN = 0.30


def _slap(sr,velocity,seed):
    t = np.arange(round(0.2*sr),dtype=np.float64)/sr
    noise = np.random.default_rng(seed).standard_normal(len(t))
    brush = sosfilt(butter(2,[2000,4000],btype='bandpass',fs=sr,output='sos'),noise)
    body = np.sin(2*np.pi*105*t)+0.6*np.sin(2*np.pi*215*t)
    result = (_SLAP_BODY_GAIN*body+_SLAP_BRUSH_GAIN*brush)*np.exp(-t/0.045)
    attack = min(len(t),max(1,round(0.001*sr)))
    result[:attack] *= np.sin(np.linspace(0,np.pi/2,attack))**2
    result[-max(1,round(0.005*sr)):] *= np.linspace(1,0,max(1,round(0.005*sr)))
    return result*velocity/127


def _schedule(events,bpm,sr):
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError('BPM must be positive and finite')
    groups, scheduled = {}, []
    for event in events:
        for field in ('start','duration','velocity'):
            if not math.isfinite(event[field]):
                raise ValueError(f'Nonfinite event {field}')
        if event['start'] < 0 or event['duration'] <= 0 or not 0 <= event['velocity'] <= 127:
            raise ValueError('Invalid event timing or velocity')
        _integer(event['string'],1,6,'string')
        _integer(event['pitch'],0,127,'pitch')
        _integer(event['fret'],0,24,'fret')
        if event['technique'] not in {'X','CM','H','P','PB'}:
            groups.setdefault(event['start'],[]).append(event['string'])
    for index,event in enumerate(events):
        rake = 0.0
        if event['technique'] not in {'X','CM','H','P','PB'}:
            played = groups[event['start']]
            if len(set(played)) >= 2:
                rake = ((max(played)-event['string']) if float(event['start']).is_integer()
                        else (event['string']-min(played)))*0.014
        onset = round((event['start']*60/bpm+rake)*sr)
        duration = max(1,round(event['duration']*60/bpm*sr))
        scheduled.append(dict(event=event,onset=onset,duration=duration,index=index))
    pitched = sorted((s for s in scheduled if s['event']['technique'] != 'X'),key=lambda s:s['onset'])
    last = {}
    for item in pitched:
        string = item['event']['string']
        if string in last:
            previous = last[string]
            previous['duration'] = min(previous['duration'],max(0,item['onset']-previous['onset']))
        last[string] = item
    return scheduled


def _room(mono,left,right,left_gain,right_gain,sr):
    delays = [round(left*sr),round(right*sr)]
    stereo = np.zeros((len(mono)+max(delays),2),dtype=np.float64)
    stereo[:len(mono),:] = mono[:,None]
    for channel,delay,gain in ((0,delays[0],left_gain),(1,delays[1],right_gain)):
        stereo[delay:delay+len(mono),channel] += gain*mono
    return stereo


def _sf2_attacks(path,scheduled,sr,length):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    import fluidsynth
    synth = fluidsynth.Synth(samplerate=float(sr))
    result = np.zeros((length,2),dtype=np.float64)
    try:
        soundfont = synth.sfload(str(path))
        if soundfont < 0 or synth.program_select(0,soundfont,0,25) < 0:
            raise RuntimeError('SoundFont load or guitar program selection failed')
        for item in scheduled:
            event = item['event']
            if event['technique'] in {'X','H','P'} or item['duration'] <= 0:
                continue
            synth.all_sounds_off(0)
            synth.noteon(0,event['pitch'],int(event['velocity']))
            size = min(round(0.08*sr),item['duration'],length-item['onset'])
            attack = np.asarray(synth.get_samples(size),dtype=np.float64).reshape(-1,2)/32768
            synth.noteoff(0,event['pitch'])
            attack *= np.linspace(1,0,len(attack))[:,None]
            result[item['onset']:item['onset']+size] += 0.12*attack
    finally:
        synth.delete()
    return result


def synthesize_realistic_guitar(
    project,
    arrangement: dict,
    out_wav_path: str | Path | None = None,
    sr: int = 44100,
    sf2_path: str | Path | None = None,
) -> np.ndarray:
    """Synthesize stereo (N, 2) float32 realistic acoustic guitar audio at 44.1 kHz.

    Models:
    1. 6-String Extended Karplus-Strong Plucked-String Waveguide (+ optional SF2 attack layer)
       with string-gauge dispersion and pick-position comb filtering.
    2. Acoustic Guitar Wood Body Resonator (98 Hz Helmholtz air mode, 205 Hz top plate, 2.4 kHz brilliance).
    3. Articulation physics:
       - Strumming Rake Delay (14 ms/string in both stroke directions when multiple strings share a beat)
       - 'X' Dead Slap (wood body knock + muted fretwire metallic brush on beats with technique=='X')
       - 'P.M.' Palm Mute (lowered loop cutoff + 3.5x faster bridge damping when technique=='PM')
       - 'H' / 'P' Hammer-on & Pull-off (suppressed pick transient, smooth legato excitation)
    """
    _sample_rate(sr)
    scheduled = _schedule(arrangement['events'],project.bpm,sr)
    length = max((s['onset']+(round(0.2*sr) if s['event']['technique'] == 'X' else s['duration']) for s in scheduled),default=1)
    dry = np.zeros(length,dtype=np.float64)
    for item in scheduled:
        event = item['event']
        seed = item['index']*100003+event['pitch']*101+event['string']
        signal = (_slap(sr,event['velocity'],seed) if event['technique'] == 'X' else
                  _pluck(event['pitch'],event['string'],item['duration'],sr,event['velocity'],event['technique'],seed)
                  if item['duration'] > 0 else np.zeros(0))
        dry[item['onset']:item['onset']+len(signal)] += signal
    body = dry.copy()
    for frequency,q,contribution in ((98,5,0.20),(205,7,0.15),(2400,1.5,0.10)):
        b,a = iirpeak(frequency,q,fs=sr)
        body += contribution*lfilter(b,a,dry)
    stereo = _room(body,0.009,0.014,0.12,0.12,sr)
    tail = round(0.037*sr)
    stereo = np.pad(stereo,((0,max(0,len(body)+tail-len(stereo))),(0,0)))
    for channel,delay in ((0,round(0.031*sr)),(1,tail)):
        stereo[delay:delay+len(body),channel] += 0.06*body
    if sf2_path is not None:
        stereo += _sf2_attacks(sf2_path,scheduled,sr,len(stereo))
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo *= 0.85/peak
    if not np.isfinite(stereo).all():
        raise ValueError('Nonfinite synthesized audio')
    stereo = stereo.astype(np.float32)
    if out_wav_path is not None:
        sf.write(str(out_wav_path),stereo,sr,format='WAV',subtype='PCM_16')
    return stereo


def mix_real_vocal_and_guitar(
    vocal_wav_path: str | Path,
    project,
    arrangement: dict,
    out_wav_path: str | Path,
    sr: int = 44100,
    vocal_gain: float = 1.0,
    guitar_gain: float = 0.85,
) -> dict:
    """Time-align the simulated acoustic guitar (`beat_origin` + `bpm` timeline) with the real singer's `vocals.wav`, apply subtle room ambience, write stereo WAV, and return objective acoustic evaluation metrics."""
    _sample_rate(sr)
    if not all(math.isfinite(v) for v in (vocal_gain,guitar_gain,project.beat_origin)):
        raise ValueError('Nonfinite gain or recording origin')
    vocal_channels, source_sr = sf.read(str(vocal_wav_path),dtype='float64',always_2d=True)
    if not np.isfinite(vocal_channels).all():
        raise ValueError('Nonfinite vocal samples')
    vocal = vocal_channels.mean(axis=1)
    if source_sr != sr:
        divisor = math.gcd(source_sr,sr)
        vocal = resample_poly(vocal,sr//divisor,source_sr//divisor)
    guitar = synthesize_realistic_guitar(project,arrangement,sr=sr).astype(np.float64)
    offset = int(round(getattr(project,'beat_origin',0.0)*sr))
    guitar = np.pad(guitar,((offset,0),(0,0))) if offset >= 0 else guitar[min(-offset,len(guitar)):]
    vocal_room = _room(vocal,0.023,0.039,0.04,0.03,sr)
    length = max(len(vocal_room),len(guitar))
    padded_guitar = np.pad(guitar,((0,length-len(guitar)),(0,0)))
    padded_vocal = np.pad(vocal,(0,length-len(vocal)))
    mix = vocal_gain*np.pad(vocal_room,((0,length-len(vocal_room)),(0,0)))+guitar_gain*padded_guitar
    peak = np.max(np.abs(mix)) if mix.size else 0.0
    if peak > 0:
        mix *= 10**(-1/20)/peak
    sf.write(str(out_wav_path),mix,sr,format='WAV',subtype='PCM_16')
    return evaluate_vocal_guitar_mix(padded_vocal,padded_guitar.mean(axis=1),sr=sr)


def evaluate_vocal_guitar_mix(
    vocal_audio: np.ndarray,
    guitar_audio: np.ndarray,
    sr: int = 44100,
) -> dict:
    """Compute objective audio-domain metrics verifying harmonic consonance and rhythmic lock between the real vocal and simulated guitar."""
    _sample_rate(sr)
    def mono(audio):
        array = np.asarray(audio,dtype=np.float64)
        if array.ndim == 2 and array.shape[1] in (1, 2):
            array = array.mean(axis=1)
        elif array.ndim != 1:
            raise ValueError('Expected mono or stereo audio')
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite audio samples')
        return array
    vocal, guitar = mono(vocal_audio),mono(guitar_audio)
    length = max(len(vocal),len(guitar))
    vocal = np.pad(vocal,(0,length-len(vocal)))
    guitar = np.pad(guitar,(0,length-len(guitar)))
    frequencies = np.fft.rfftfreq(4096,1/sr)
    mask = (frequencies >= 50) & (frequencies <= 5000)
    pcs = np.rint(69+12*np.log2(frequencies[mask]/440)).astype(np.int64)%12
    starts = range(0,max(1,length),1024)
    window = np.hanning(4096)
    def analyze(audio):
        chroma, rms = [], []
        for start in starts:
            frame = np.pad(audio[start:start+4096],(0,max(0,start+4096-len(audio))))
            rms.append(np.sqrt(np.mean(frame**2)))
            power = np.abs(np.fft.rfft(frame*window))**2
            vector = np.bincount(pcs,weights=power[mask],minlength=12)
            total = vector.sum()
            chroma.append(vector/total if total > 0 else vector)
        return np.asarray(chroma),np.asarray(rms)
    vc,vr = analyze(vocal)
    gc,gr = analyze(guitar)
    intervals = (np.arange(12)[:,None]-np.arange(12)[None,:])%12
    kernel = np.isin(intervals,[0,3,4,5,7,8,9]).astype(float)
    clash_kernel = np.isin(intervals,[1,6,11]).astype(float)
    consonance = np.einsum('bi,ij,bj->b',vc,kernel,gc)
    clashes = np.einsum('bi,ij,bj->b',vc,clash_kernel,gc)
    overlap = (vr > 0) & (gr > 0)
    weights = vr[overlap]*gr[overlap]
    active_count = int(overlap.sum())
    spectral = float(np.average(consonance[overlap],weights=weights)) if active_count else 0.0
    clash_ratio = float(np.mean(clashes[overlap] > consonance[overlap])) if active_count else 0.0
    vo = np.maximum(0,np.diff(vr))
    go = np.maximum(0,np.diff(gr))
    correlation = float(np.corrcoef(vo,go)[0,1]) if len(vo) > 1 and np.std(vo) > 0 and np.std(go) > 0 else 0.0
    summed = vocal+guitar
    peak = float(np.max(np.abs(summed))) if length else 0.0
    rms = float(np.sqrt(np.mean(summed**2))) if length else 0.0
    crest = float(20*np.log10(peak/rms)) if rms > 0 else 0.0
    return dict(spectral_consonance=spectral,clash_frame_ratio=clash_ratio,beat_sync_corr=correlation,
                crest_factor_db=crest,active_overlap_frames=active_count)
