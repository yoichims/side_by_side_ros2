#!/usr/bin/env python3
"""Summarise a formation_metrics CSV.

    python3 tools/summarise_metrics.py /tmp/run.csv [--tail 30]

Reports the same statistics the offline closed-loop sim reports, over the same
window, so a real run and a simulated one can be compared line for line rather
than by impression. numpy only -- no ROS, no pandas.

`--tail` is the trailing window in seconds. Transients at the start are the
vehicles accelerating from rest onto their lanes and say nothing about whether
the formation holds; the tail is where a steady state either exists or does
not.
"""

import sys

import numpy as np


def load(path):
    """Rows, plus the '#' parameter header the metrics node writes above them."""
    header, names = [], None
    with open(path) as handle:
        for line in handle:
            if line.startswith('#'):
                header.append(line[1:].strip())
                continue
            names = line.strip().split(',')
            break
    data = np.genfromtxt(path, delimiter=',', comments='#',
                         skip_header=len(header) + 1)
    if data.ndim == 1:
        data = data[None, :]
    return {name: data[:, i] for i, name in enumerate(names)}, header


def weight_of(header, name, default=None):
    """A weight from the '#' parameter header, or `default` if absent."""
    tag = f' {name}='
    for line in header:
        if tag in line:
            try:
                return float(line.split(tag, 1)[1].split()[0])
            except ValueError:
                return default
    return default


def band(label, values, unit='m', fmt='%7.2f'):
    if values is None:
        print('  %-26s (not in this log)' % label)
        return
    ok = np.isfinite(values)
    if not ok.any():
        print('  %-26s (no data)' % label)
        return
    v = values[ok]
    print(('  %-26s ' + fmt + ' ' + fmt + ' ' + fmt + '   %s') % (
        label, np.nanmean(v), np.nanpercentile(v, 50), np.nanmax(np.abs(v)), unit))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    tail_s = 30.0
    if '--tail' in sys.argv:
        tail_s = float(sys.argv[sys.argv.index('--tail') + 1])
    if not args:
        raise SystemExit(__doc__)

    d, header = load(args[0])
    t = d['t'] - d['t'][0]
    print('%s: %d rows, %.1f s' % (args[0], len(t), t[-1]))
    if header:
        # Only the parameters that shape the behaviour under investigation;
        # a full dump buries the one line that explains the run.
        keep = [h for h in header
                if ' w_' in h or 'concept' in h or 'n_v' in h or 'n_w' in h]
        print('run parameters (%d recorded, showing %d):'
              % (len(header), len(keep or header[:12])))
        for line in keep or header[:12]:
            print('  ' + line)
    if t[-1] <= 0:
        raise SystemExit('no elapsed time in this log')

    tail = t >= max(0.0, t[-1] - tail_s)
    print('\nlast %.0f s (%d rows)            mean  median     max' % (
        min(tail_s, t[-1]), int(tail.sum())))

    sep = d['sep'][tail]
    band('separation', sep)
    # `sep_target` drives `rel_dist_end` and nothing else, so with that weight
    # at zero it is not a target the pair is tracking, and printing
    # sep - sep_target as a "separation error" invented a tracking error
    # against a quantity nothing was asking for. What sets the separation now
    # is cross_track holding each vehicle on its lane, so the lateral offset
    # is the quantity to read.
    rel_dist_end = weight_of(header, 'w_rel_dist_end')
    if rel_dist_end:
        band('separation error', sep - d['sep_target'][tail])
    else:
        print('  %-26s (w_rel_dist_end=0: separation follows the lanes, '
              'not sep_target)' % 'separation error')
    band('lateral offset |v1|', np.abs(d['v1_lateral'][tail]))
    band('along-track offset |v1|', np.abs(d['v1_along'][tail]))

    # The point of the formation: the passengers hold a conversation. What
    # that needs is not a separation but an *angle* -- how far off straight
    # ahead each passenger must look to face the other, which is what a
    # stagger creates. At 1 m of lateral spacing, 0.2 m of stagger costs 11
    # degrees and 0.9 m costs 42. Nothing was measuring this, so the quantity
    # the system exists to produce was not in its own log.
    turn = np.degrees(np.arctan2(np.abs(d['v1_along'][tail]),
                                 np.abs(d['v1_lateral'][tail])))
    band('head turn to face partner', turn, 'deg')
    comfortable = (turn < 25.0) & (sep < 2.0)
    print('  %-26s %6.0f%% of the window' % (
        'conversational geometry', 100.0 * np.nanmean(comfortable)))
    band('lane error v1 (outer)', d['v1_lane_err'][tail])
    band('lane error v2 (inner)', d['v2_lane_err'][tail])
    band('relative yaw', np.degrees(d['rel_yaw'][tail]), 'deg')
    # Centre-based first, body-based second, and the body one is the
    # constraint's own quantity: with a 0.275 m half-width, 0.29 m of centre
    # clearance was a body touching the wall. Reading only the centre column
    # is what hid the collisions.
    band('wall clearance v1 (centre)', d['v1_wall'][tail])
    band('wall clearance v2 (centre)', d['v2_wall'][tail])
    for name in ('v1_wall_body', 'v2_wall_body'):
        label = 'wall clearance %s (body)' % name[:2]
        band(label, d[name][tail] if name in d else None)
    band('speed v1', d['v1_v'][tail], 'm/s')
    band('speed v2', d['v2_v'][tail], 'm/s')
    band('speed error v1', d['v1_v'][tail] - d['v1_v_star'][tail], 'm/s')
    band('speed error v2', d['v2_v'][tail] - d['v2_v_star'][tail], 'm/s')
    band('yaw rate v1', d['v1_w'][tail], 'rad/s')
    band('yaw rate v2', d['v2_w'][tail], 'rad/s')

    # Route-relative: signed, so a side swap is a sign change rather than a
    # distance that happens to come out right.
    def maybe(name):
        return d[name][tail] if name in d else None

    for ns in ('v1', 'v2'):
        cross, target = maybe(f'{ns}_cross'), maybe(f'{ns}_cross_target')
        if cross is None:
            band(f'cross-track {ns}', None)
            continue
        band(f'cross-track {ns}', cross)
        band(f'cross-track error {ns}', cross - target)
        band(f'heading vs lane {ns}',
             np.degrees(maybe(f'{ns}_tangent')), 'deg')

    print('\nwhole run')
    # Spinning: turning hard while going nowhere. This is the failure that
    # looks dramatic in RViz and is invisible in a mean.
    for ns in ('v1', 'v2'):
        spin = (np.abs(d[f'{ns}_v']) < 0.15) & (np.abs(d[f'{ns}_w']) > 0.25)
        print('  %s spinning in place        %5.1f%% of ticks' % (
            ns, 100.0 * spin.mean()))
    # Rigid co-rotation: every formation term is invariant to it, so the pair
    # can satisfy the whole social cost while going anywhere at all. Turning
    # together is also what a healthy pair does, so this only means anything
    # alongside the arc length below: identical yaw rates while advancing is
    # the formation working, and while stalled it is the degeneracy.
    if 'v1_cross' in d and 'v2_cross' in d:
        # On its own side of the centreline means the sign matches the target.
        for ns in ('v1', 'v2'):
            ok = np.sign(d[f'{ns}_cross']) == np.sign(d[f'{ns}_cross_target'])
            print('  %s on its own side of the route %5.1f%% of ticks' % (
                ns, 100.0 * np.mean(ok)))
        together = (np.sign(d['v1_cross']) == np.sign(d['v2_cross']))
        print('  both on the SAME side of the route   %5.1f%% of ticks' % (
            100.0 * np.mean(together)))
    rigid = np.abs(d['v1_w'] - d['v2_w']) < 0.02
    crawling = (np.abs(d['v1_v']) < 0.30) & (np.abs(d['v2_v']) < 0.30)
    print('  identical yaw rates         %5.1f%% of ticks '
          '(%.1f%% of those while crawling)' % (
              100.0 * rigid.mean(),
              100.0 * crawling[rigid].mean() if rigid.any() else 0.0))
    near = np.minimum(d['v1_wall'], d['v2_wall'])
    ok = np.isfinite(near)
    if ok.any():
        print('  either vehicle within 0.30 m of a wall  %.1f%% of ticks' % (
            100.0 * (near[ok] < 0.30).mean()))
    # Did they actually get anywhere, or loop?
    for ns, side in (('v1', 'outer'), ('v2', 'inner')):
        s = d[f'{ns}_s']
        advance = np.diff(s)
        # unwrap the lap boundary the same way Reference.progress does
        span = np.nanmax(s) - np.nanmin(s)
        advance = advance[np.abs(advance) < span / 2.0]
        print('  %s advanced %6.1f m along the %s lane' % (
            ns, np.nansum(advance), side))


if __name__ == '__main__':
    main()
