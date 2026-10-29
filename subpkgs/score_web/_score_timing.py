"""Shared, non-mutating score timing in 48 ticks per quarter note."""
from fractions import Fraction
import math

TICKS = 48


def duration_parts(ticks):
    if not isinstance(ticks, int) or ticks <= 0:
        raise ValueError("Duration must be a positive integer tick count")
    vocabulary = {}
    for denominator in (1, 2, 4, 8, 16, 32, 64, 128):
        for dots in (0, 1, 2):
            for tuplet in (None, (3, 2)):
                value = Fraction(4 * TICKS, denominator) * (2 - Fraction(1, 2 ** dots))
                if tuplet:
                    value *= Fraction(tuplet[1], tuplet[0])
                if value.denominator == 1:
                    vocabulary.setdefault(int(value), dict(ticks=int(value), denominator=denominator,
                                                           dots=dots, tuplet=tuplet))
    result = []
    for size in sorted(vocabulary, reverse=True):
        while ticks >= size:
            result.append(vocabulary[size].copy())
            ticks -= size
    assert ticks == 0
    return result


def build_score_timeline(project, arrangement):
    numerator, denominator = map(int, project.meter.split('/'))
    if numerator <= 0 or denominator <= 0:
        raise ValueError("Meter must be positive")
    measure = numerator * 4 / denominator
    measure_ticks = round(measure * TICKS)
    if measure_ticks <= 0 or not math.isclose(measure_ticks, measure * TICKS):
        raise ValueError("Meter cannot be represented on the score grid")
    intervals = []
    for index, note in sorted(enumerate(project.notes), key=lambda pair: pair[1].start):
        if not math.isfinite(note.start) or not math.isfinite(note.duration) or note.start < 0 or note.duration <= 0:
            raise ValueError("Invalid melody timing")
        start, end = round(note.start * TICKS), round((note.start + note.duration) * TICKS)
        if end <= start or (intervals and intervals[-1][1] > start):
            raise ValueError("Melody collapses or overlaps on the notation grid")
        intervals.append((start, end, index, int(note.pitch)))
    last = max((end for _, end, _, _ in intervals), default=0)
    for event in arrangement['events']:
        start, duration = event['start'], event['duration']
        if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or duration <= 0:
            raise ValueError("Invalid guitar event timing")
        last = max(last, round((start + duration) * TICKS))
    count = max(arrangement['bar_count'], len(arrangement['bars']), math.ceil(last / measure_ticks))
    bars = []
    for bar in range(count):
        left, right = bar * measure_ticks, (bar + 1) * measure_ticks
        spans, cursor = [], left
        for start, end, index, pitch in intervals:
            a, b = max(start, left), min(end, right)
            if b <= a:
                continue
            if a > cursor:
                spans.append(dict(start_tick=cursor-left, duration_ticks=a-cursor, pitch=None,
                                  note_index=None, continuation=False))
            spans.append(dict(start_tick=a-left, duration_ticks=b-a, pitch=pitch,
                              note_index=index, continuation=a > start))
            cursor = b
        if cursor < right:
            spans.append(dict(start_tick=cursor-left, duration_ticks=right-cursor, pitch=None,
                              note_index=None, continuation=False))
        bars.append(spans)
    return dict(measure_ticks=measure_ticks, bars=bars)
