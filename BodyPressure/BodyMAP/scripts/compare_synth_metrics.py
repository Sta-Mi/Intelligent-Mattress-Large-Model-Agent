"""Print a fair comparison table for BodyPressureSD synthetic PMM runs."""

import argparse
import json
from pathlib import Path


METRICS = ('3D MPJPE', 'PVE', 'height', 'chest', 'waist', 'hips',
           'v2vP', 'v2vP 1EA', 'v2vP 2EA')


def load_result(spec):
    if '=' in spec:
        label, filename = spec.split('=', 1)
    else:
        filename = spec
        label = Path(filename).parent.name
    payload = json.loads(Path(filename).read_text())
    metrics = payload.get('metrics', payload.get('metric'))
    if metrics is None:
        raise ValueError(f"{filename} has no final metrics; final diagnostics may still be running")
    overall = metrics['overall']
    if payload.get('modality') != 'pressure':
        raise ValueError(f"{filename} is not pressure-only: {payload.get('modality')}")
    if overall.get('count') != 12381:
        raise ValueError(
            f"{filename} has {overall.get('count')} samples; expected the 12,381-sample "
            "BodyPressureSD 71-80 validation split"
        )
    payload.setdefault('epoch', payload.get('epochs'))
    return label, payload, overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='+', help='LABEL=/path/to/metrics.json')
    parser.add_argument('--out')
    args = parser.parse_args()
    loaded = [load_result(spec) for spec in args.results]
    rows = []
    for label, payload, overall in loaded:
        row = {'model': label, 'epoch': payload.get('epoch'), 'samples': overall['count']}
        row.update({metric: overall.get(metric) for metric in METRICS})
        rows.append(row)
    headers = ('model', 'epoch', 'samples') + METRICS
    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in rows:
        print('| ' + ' | '.join(str(row.get(header)) for header in headers) + ' |')
    if args.out:
        Path(args.out).write_text(json.dumps({'rows': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
