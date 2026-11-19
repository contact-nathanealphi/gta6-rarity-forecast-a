#!/usr/bin/env python3
"""GTA VI trophy rarity forecast (builder A: hierarchical logit curve).

Forecasts each trophy's Exophase earn rate at the case target as five
quantiles, from the case history alone.  Standard library only, no network,
no randomness: two runs on the same case are byte-identical.

How the forecast is built
--------------------------
Exophase does not move its counts smoothly.  Between refreshes the tracked
player count is frozen and every rate sits still; at a refresh the count
jumps and each rate is pulled toward the rate among the newly tracked
players, who are on average earlier in the game and so hold fewer trophies.
The forecast therefore splits the rate path into two pieces that are
estimated separately and multiplied back together at the end:

1. Progression.  Each read's rate is first divided by the dilution the
   trophy has already suffered from every refresh seen so far in the case
   history, giving a "dilution-free" series.  A straight line is fitted by
   least squares to the logit of that series against ln(1 + hours since
   2026-11-19T00:00Z), per trophy, and read at the target.  The line is
   shrunk toward the tier mean line (platinum / gold / silver / bronze)
   and the tier line toward the global line, because launch-night rates rest
   on only a handful of players and the gaps between two trophies of the
   same tier are mostly counting noise.  The line's slope comes from the
   dilution-free series, but its level is anchored on the trophy's last
   OBSERVED rate (which already carries all past dilution), so the
   forecast only adds the dilution still to come.

2. Dilution still to come.  Refreshes in the history give, per event, the
   relative growth of the tracked population and, per trophy, the implied
   rate among the newly tracked players (the dilution arithmetic: the new
   cohort's rate is what makes the before/after counts add up).  Future
   refreshes are simulated forward: how many are expected before the
   target (from the observed refresh cadence, or a prior of about one
   refresh every five hours for the first one and about twelve hours
   between later ones when the history has seen none yet), how much
   the population grows at each (decaying as the base grows), and how far
   each refresh pulls each trophy's rate toward its new-cohort rate.
   When the history shows no refresh at all, the launch-night refresh
   known from the contest (638 -> 3,868 tracked players, every rate but
   the 100% one falling) supplies the prior: the first refresh grows the
   population about six-fold and new players hold trophies at a small
   fraction of the launch-night rates (tier medians measured on that
   refresh: platinum 0.00, gold 0.03, silver 0.00, bronze 0.04), least
   for the platinum and most for the first trophy everyone earns: its
   new cohort holds it at 100% (the contest notes it never fell, and the
   live page still shows 100% hours later), so its forecast stays at
   the top.

Spread.  The logit spread around the median combines four independent
uncertainties (added as variances): a random-walk term that grows with the
square root of the horizon in days; the residual scatter of the trophy's
own line fit; the uncertainty of the coming dilution (it grows with the
size of the expected dilution shift and with the number of refreshes);
and the counting noise of the last read (a rate resting on four players
is worth less than one resting on four thousand).  Quantiles use the
standard normal z values.

Edge cases.  A trophy stuck at 100% can only fall, so its new-cohort rate
is 100% and its forecast stays high.  Null rates in a read are skipped,
never treated as zero.  A trophy with too few informative reads falls back
to its tier's line.  Every quantile is clamped to [0.01, 99.99], rounded to
four decimals, and forced non-decreasing.
"""

import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------- constants
# Anchor for the log-time axis: the release day, 2026-11-19T00:00:00Z.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# Standard-normal quantiles for the five requested levels.
Z_VALUES = (-1.645, -0.674, 0.0, 0.674, 1.645)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Q_MIN, Q_MAX = 0.01, 99.99

# Priors used only when the case history has not shown a refresh yet.
# They come from the launch-night refresh the contest describes, now
# measured exactly: the page sat unchanged at 638 tracked players, then
# refreshed to 3,868, and every rate but the 100% one fell.  The live
# page at 11:37Z still shows that same refreshed state (its 55 rates are
# consistent with integer holder counts out of 3,868 summing to 4,276),
# so the first refresh's new-cohort rates are known per trophy.  As a
# fraction of each trophy's own rate, the new cohort holds (tier medians):
# platinum 0.00, gold 0.03, silver 0.00, bronze 0.04 (overall median
# 0.04; the few high ones are early easy activities like fishing at 0.49).
# The floors of 0.01 keep the "new cohort holds literally nothing" case
# from being stated as a certainty.
GAP_FIRST_PRIOR_HOURS = 5.0   # hours to the first refresh (it came ~4.5h
                              # after release, after 4.2h of a flat page)
GAP_LATER_PRIOR_HOURS = 12.0  # hours between later refreshes: no second
                              # refresh had happened by 11:37Z, 7h after
                              # the first, so later gaps are well above 5h
GROWTH_FIRST_PRIOR = 5.06    # 3868 / 638 - 1: first-refresh population growth
GROWTH_DECAY = 0.45           # each later refresh grows the base less
# New-cohort rate as a fraction of the trophy's own rate, by tier, for the
# first refresh (new players are earlier in the game; the platinum's new
# cohort holds almost nothing, the first trophy's holds everything).
NEW_COHORT_PRIOR = {
    "platinum": 0.01,
    "gold": 0.03,
    "silver": 0.01,
    "bronze": 0.04,
}
NEW_COHORT_FIRST_TROPHY = 1.0

# Spread calibration (logit units).  The random-walk scale matches the
# house baseline's 0.5; the refresh terms were sized on the launch-night
# refresh and the live-snapshot pseudo-case backtest.
RW_SCALE = 0.5
FIT_RESID_FLOOR = 0.15
FIT_RESID_CAP = 0.80
REFRESH_SHIFT_SHARE = 0.35   # fraction of the expected dilution shift
REFRESH_COUNT_SCALE = 0.25   # per expected refresh, in quadrature
SPREAD_CAP = 1.5


# ------------------------------------------------------------------ helpers
def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(rate):
    """Clamp a percent rate into the range the logit can take."""
    if rate is None:
        return None
    return min(max(float(rate), Q_MIN), Q_MAX)


def logit(pct):
    """Logit of a percent rate (clamped first)."""
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    """Inverse logit, returned as a percent rate."""
    if value >= 0:
        odds = math.exp(-value)
        return 100.0 / (1.0 + odds)
    odds = math.exp(value)
    return 100.0 * odds / (1.0 + odds)


def log_hours(moment):
    """ln(1 + hours from the release-day anchor to `moment`)."""
    hours = (moment - EPOCH).total_seconds() / 3600.0
    return math.log1p(max(hours, -0.99))


def median(values):
    """Median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def fit_line(points):
    """Weighted least-squares line through (x, y, weight) points.

    Returns (slope, intercept, residual_rms).  With fewer than two distinct
    x values the slope is 0 and the intercept is the weighted mean of y.
    """
    if not points:
        return 0.0, 0.0, 0.0
    total_w = sum(w for _, _, w in points)
    if total_w <= 0:
        return 0.0, 0.0, 0.0
    mean_x = sum(w * x for x, _, w in points) / total_w
    mean_y = sum(w * y for _, y, w in points) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for x, _, w in points)
    if sxx <= 1e-12:
        resid = [y - mean_y for _, y, _ in points]
        rms = math.sqrt(sum(r * r for r in resid) / len(resid)) if resid else 0.0
        return 0.0, mean_y, rms
    sxy = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in points)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid = [y - (intercept + slope * x) for x, y, _ in points]
    rms = math.sqrt(sum(r * r for r in resid) / len(resid))
    return slope, intercept, rms


def shrink(value, toward, weight):
    """Pull `value` toward `toward`; weight 1 keeps value, 0 keeps toward."""
    return weight * value + (1.0 - weight) * toward


# ------------------------------------------------------------- case reading
def read_case(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_refreshes(case):
    """Find refresh events in the history.

    A refresh is a consecutive pair of reads where the tracked-player count
    jumps by more than 2%.  Returns a list of events, each with the growth
    factor, the two read indexes, and the per-trophy rate ratios.
    """
    history = case.get("history") or []
    events = []
    for j in range(1, len(history)):
        prev, cur = history[j - 1], history[j]
        n_prev, n_cur = prev.get("players_tracked"), cur.get("players_tracked")
        if not n_prev or not n_cur or n_cur <= n_prev * 1.02:
            continue
        growth = n_cur / n_prev - 1.0
        ratios = []
        rates_prev = prev.get("rates") or []
        rates_cur = cur.get("rates") or []
        for a, b in zip(rates_prev, rates_cur):
            if a is None or b is None or not a:
                ratios.append(None)
            else:
                ratios.append(b / a)
        events.append({
            "growth": growth,
            "at": cur.get("at"),
            "index": j,
            "ratios": ratios,
        })
    return events


def new_cohort_fraction(ratio, growth):
    """Implied new-cohort rate as a fraction of the old rate.

    From the dilution arithmetic r_after = (r_before * N + r_new * dN)
    / (N + dN): r_new / r_before = (ratio * (1 + g) - 1) / g.
    """
    if growth <= 0:
        return None
    value = (ratio * (1.0 + growth) - 1.0) / growth
    return min(max(value, 0.0), 1.0)


# ------------------------------------------------------------------- model
def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_hours = case.get("horizon_minutes", 0) / 60.0
    horizon_days = horizon_hours / 24.0
    x_target = log_hours(target)

    n_trophies = len(trophies)
    tiers = [t.get("tier", "bronze") for t in trophies]

    # -- refreshes seen in the history -------------------------------------
    # Two cadences: the FIRST refresh comes quickly (~5h after release),
    # later ones much slower (no second refresh 7h after the first).  Once
    # refreshes are observed, the most recent gap is the best predictor of
    # the next one; with a single observed refresh the later gap is still
    # the prior.
    events = find_refreshes(case)
    if events:
        if len(events) >= 2:
            gap_later = (parse_time(events[-1]["at"])
                         - parse_time(events[-2]["at"])).total_seconds() / 3600.0
        else:
            gap_later = GAP_LATER_PRIOR_HOURS
        gap_first = min(max(gap_later, 1.0), 48.0)
        gap_later = min(max(gap_later, 1.0), 48.0)
        # The next refresh grows the base about as much as the latest one
        # did: population growth per refresh decays as the base grows, so
        # the most recent refresh is the closest regime to the future,
        # not the median (which would keep re-applying the launch-night
        # six-fold jump long after the base is in the thousands).
        growth_next = events[-1]["growth"]
        growth_next = min(max(growth_next, 0.02), 20.0)
        last_event = events[-1]
        last_refresh_at = parse_time(last_event["at"])
    else:
        gap_first = GAP_FIRST_PRIOR_HOURS
        gap_later = GAP_LATER_PRIOR_HOURS
        growth_next = GROWTH_FIRST_PRIOR
        last_refresh_at = None

    if last_refresh_at is not None:
        hours_since_refresh = (checkpoint - last_refresh_at).total_seconds() / 3600.0
    elif history:
        first_at = parse_time(history[0]["at"])
        hours_since_refresh = (checkpoint - first_at).total_seconds() / 3600.0
    else:
        hours_since_refresh = 0.0
    hours_since_refresh = max(hours_since_refresh, 0.0)

    # Expected number of refreshes between checkpoint and target.  Refreshes
    # run roughly on a cadence, so the next one is due when the gap since
    # the last one runs out.  After that first one, further refreshes run
    # on the LATER cadence, and one only counts once the horizon is at
    # least halfway to it: the cadence is regular enough that counting a
    # full extra refresh for a fraction of a gap would over-dilute (the
    # simulation backtest caught this).
    time_to_next = max(gap_first - hours_since_refresh, 0.0)
    if horizon_hours <= time_to_next:
        expected_refreshes = 0.5 * horizon_hours / max(time_to_next, 0.5)
    else:
        expected_refreshes = 1.0 + max(
            0.0, (horizon_hours - time_to_next) / gap_later - 0.5
        )

    # -- per-trophy new-cohort fraction ------------------------------------
    # From observed refreshes when there are any: the median over the most
    # recent three events, shrunk toward the tier median.  Recency matters
    # because new cohorts get more progressed as the game ages, so the
    # first refresh's new-cohort rates (near zero for most trophies) are
    # not what later refreshes bring.  With no observed refresh, the tier
    # prior (measured on the launch-night refresh) is used.
    rho_observed = [[] for _ in range(n_trophies)]
    for event in events[-3:]:
        for i, ratio in enumerate(event["ratios"]):
            if ratio is None:
                continue
            frac = new_cohort_fraction(ratio, event["growth"])
            if frac is not None:
                rho_observed[i].append(frac)
    rho_tier = {}
    for tier in set(tiers):
        vals = [median(rho_observed[i]) for i in range(n_trophies)
                if tiers[i] == tier and rho_observed[i]]
        if vals:
            rho_tier[tier] = median(vals)
    rho_global = median(list(rho_tier.values())) if rho_tier else 0.04
    rho = []
    for i in range(n_trophies):
        tier = tiers[i]
        last_rate = last_seen_rate(history, i)
        if last_rate is not None and last_rate >= 99.9:
            rho.append(NEW_COHORT_FIRST_TROPHY)
            continue
        if rho_observed[i]:
            own = median(rho_observed[i])
            w = len(rho_observed[i]) / (len(rho_observed[i]) + 2.0)
            base = rho_tier.get(tier, rho_global)
            rho.append(shrink(own, base, w))
        else:
            rho.append(NEW_COHORT_PRIOR.get(tier, 0.04))

    # -- dilution already suffered, per read --------------------------------
    # cumulative[i][j] = product of observed refresh ratios for trophy i up
    # to and including read j (1.0 before the first refresh).
    cumulative = [[1.0] * len(history) for _ in range(n_trophies)]
    for event in events:
        j = event["index"]
        for i, ratio in enumerate(event["ratios"]):
            if ratio is None:
                continue
            for k in range(j, len(history)):
                cumulative[i][k] *= ratio

    # -- progression lines on the dilution-free series ----------------------
    # Weight a read by how many players hold the trophy: later reads with
    # thousands of tracked players pin the rate down far better than the
    # launch-night reads resting on four players.
    points_per_trophy = [[] for _ in range(n_trophies)]
    for j, read in enumerate(history):
        x = log_hours(parse_time(read["at"]))
        n_players = read.get("players_tracked") or 0
        rates = read.get("rates") or []
        for i in range(n_trophies):
            if i >= len(rates) or rates[i] is None:
                continue
            adjusted = rates[i] / cumulative[i][j] if cumulative[i][j] > 0 else rates[i]
            holders = max(adjusted / 100.0 * max(n_players, 1), 1.0)
            points_per_trophy[i].append((x, logit(adjusted), holders))

    own_fit = [fit_line(pts) for pts in points_per_trophy]

    # Tier lines: pool every trophy of the tier; the global line pools all.
    tier_fit = {}
    for tier in set(tiers):
        pooled = []
        for i in range(n_trophies):
            if tiers[i] == tier:
                pooled.extend(points_per_trophy[i])
        tier_fit[tier] = fit_line(pooled)
    global_fit = fit_line([p for pts in points_per_trophy for p in pts])

    # Shrink each tier line toward the global line by how many trophies the
    # tier holds (a one-trophy tier is mostly the global line).
    tier_slope = {}
    tier_resid = {}
    for tier, (slope, _, resid) in tier_fit.items():
        count = sum(1 for t in tiers if t == tier)
        w = count / (count + 6.0)
        tier_slope[tier] = shrink(slope, global_fit[0], w)
        tier_resid[tier] = min(max(resid, FIT_RESID_FLOOR), FIT_RESID_CAP)

    global_resid = min(max(global_fit[2], FIT_RESID_FLOOR), FIT_RESID_CAP)

    # -- per-trophy forecast -------------------------------------------------
    rows = []
    for i, trophy in enumerate(trophies):
        tier = tiers[i]
        pts = points_per_trophy[i]
        slope_i, _, resid_i = own_fit[i]
        # How much the trophy's own slope can be trusted: more reads and a
        # wider log-time span mean more of its own slope, else the tier's.
        if pts:
            xs = [p[0] for p in pts]
            x_span = max(xs) - min(xs)
        else:
            x_span = 0.0
        w_own = (len(pts) / (len(pts) + 6.0)) * (x_span / (x_span + 1.0))
        slope = shrink(slope_i, tier_slope[tier], w_own)

        last_rate = last_seen_rate(history, i)
        if last_rate is None:
            # No read ever listed this trophy: use the tier's last median.
            last_rate = tier_last_rate(history, tiers, tier, i)
        # Anchor the level on the last OBSERVED rate: it already carries
        # every past refresh's dilution, so only future dilution is added
        # below.  (Anchoring on the dilution-free rate instead would undo
        # past dilution and never put it back.)
        x_last = log_hours(parse_time(history[-1]["at"])) if history else log_hours(checkpoint)
        anchor = logit(last_rate) - slope * x_last
        prog_logit = anchor + slope * x_target

        # Future dilution: simulate the expected refreshes forward.  Each
        # refresh pulls the rate toward the new cohort's rate; later
        # refreshes grow the population less, so their pull is weaker.
        # A fractional expected count dilutes only that share of the next
        # refresh, in log space.  When the history shows refreshes, the
        # growth decay continues from the latest one: the next refresh
        # grows the base by about the decay factor times the latest
        # growth (the population is already bigger).  With no refresh
        # seen yet, the first future refresh is the launch-night one,
        # growth 5.06.
        #
        # The pull is applied to the RATE, not to its logit: near 100% a
        # small percent dilution is a huge logit move (a trophy at 100%
        # whose new cohort holds it at ~100% stays near the top, which is
        # four logit units down from 99.99, not the 0.01 the logit shift
        # would give), and the same holds in reverse near 0%.
        dilution = 1.0
        whole = int(expected_refreshes)
        frac = expected_refreshes - whole
        growth = growth_next * (GROWTH_DECAY if events else 1.0)
        for _ in range(whole):
            ratio = (1.0 + rho[i] * growth) / (1.0 + growth)
            dilution *= max(ratio, 1e-6)
            growth *= GROWTH_DECAY
        if frac > 0:
            ratio = (1.0 + rho[i] * growth) / (1.0 + growth)
            dilution *= max(ratio, 1e-6) ** frac
        median_rate = inv_logit(prog_logit) * dilution
        median_logit = logit(median_rate)
        log_dilution = math.log(max(dilution, 1e-6))

        # Spread: random walk + fit residual + refresh uncertainty +
        # counting noise of the last read, added as variances.
        s_rw = RW_SCALE * math.sqrt(max(horizon_days, 0.0))
        if pts:
            s_fit = min(max(resid_i, FIT_RESID_FLOOR), FIT_RESID_CAP)
            s_fit = shrink(s_fit, tier_resid[tier], w_own)
        else:
            s_fit = tier_resid[tier]
        s_ref = (REFRESH_SHIFT_SHARE * abs(log_dilution)
                 + REFRESH_COUNT_SCALE * math.sqrt(max(expected_refreshes, 0.0)))
        last_n = history[-1].get("players_tracked") or 1 if history else 1
        holders = max((last_rate or 0.0) / 100.0 * last_n, 1.0)
        s_count = min(0.5 / math.sqrt(holders), 0.4)
        spread = math.sqrt(s_rw ** 2 + s_fit ** 2 + s_ref ** 2 + s_count ** 2)
        spread = min(spread, SPREAD_CAP)

        quantiles = []
        for z in Z_VALUES:
            q = inv_logit(median_logit + z * spread)
            quantiles.append(round(min(max(q, Q_MIN), Q_MAX), 4))
        # Rounding can only break monotonicity by a hair; enforce it.
        for k in range(1, 5):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})

    return {"trophies": rows}


def last_seen_rate(history, index):
    """The trophy's rate in the latest read that lists it, else None."""
    for read in reversed(history):
        rates = read.get("rates") or []
        if index < len(rates) and rates[index] is not None:
            return rates[index]
    return None


def tier_last_rate(history, tiers, tier, index):
    """Fallback last rate: median of the tier's last-seen rates."""
    vals = []
    for i, t in enumerate(tiers):
        if t == tier:
            rate = last_seen_rate(history, i)
            if rate is not None:
                vals.append(rate)
    return median(vals) if vals else 1.0


# ---------------------------------------------------------------------- main
def main(argv):
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input/")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output/")
    os.makedirs(output_dir, exist_ok=True)
    case_paths = sorted(glob.glob(os.path.join(input_dir, "*.in")))
    for path in case_paths:
        case = read_case(path)
        forecast = forecast_case(case)
        out_path = os.path.join(output_dir, case["id"] + ".out")
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(forecast, separators=(",", ":")))
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
