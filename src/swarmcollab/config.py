"""Paths, periods and the data-split guard."""
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
VILLAGE = DATA / 'ai-village' / 'main-village'
TURNS = VILLAGE / 'computer_use_turns.jsonl.gz'
SESSIONS = VILLAGE / 'computer_use_sessions.jsonl.gz'
AGENTS = VILLAGE / 'agents.jsonl.gz'
LABELS = DATA / 'labels' / 'ai-village-chart-props.json'
OUT = ROOT / 'outputs'
STEPS = OUT / 'steps'

# Main period: from the start of the individual-goals phase to the last request record.
MAIN_START = dt.date(2026, 7, 6)
MAIN_END = dt.date(2026, 9, 15)
# First slice: all tuning happens here. Train on 6-31 July, validate on 3-14 August.
DEV_START, DEV_END = MAIN_START, dt.date(2026, 8, 14)
TRAIN_END = dt.date(2026, 7, 31)
VALID_START = dt.date(2026, 8, 3)
# Second slice: sealed until the final evaluation.
TEST_START, TEST_END = dt.date(2026, 8, 17), MAIN_END


class SealedSliceError(RuntimeError):
    pass


def as_date(d):
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    return dt.date.fromisoformat(str(d)[:10])


def check_unsealed(days, final=False):
    """Refuse to score any day of the sealed slice unless this is the final run."""
    if final:
        return
    bad = sorted({as_date(d) for d in days if as_date(d) >= TEST_START})
    if bad:
        raise SealedSliceError(f'{len(bad)} sealed days requested ({bad[0]} .. {bad[-1]}); pass final=True only for the final evaluation')


def dev_days(days):
    """Keep only first-slice days."""
    return [d for d in days if DEV_START <= as_date(d) <= DEV_END]


def parse_utc(stamp):
    """'2026-07-06 16:08:50.69202' (UTC, fraction of any length) -> aware datetime."""
    if '.' in stamp:
        head, frac = stamp.split('.', 1)
        stamp = head + '.' + (frac + '000000')[:6]
    return dt.datetime.fromisoformat(stamp).replace(tzinfo=dt.timezone.utc)
