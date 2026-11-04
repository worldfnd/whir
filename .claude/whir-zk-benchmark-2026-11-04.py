"""Measure the selected BN254 profile with globally budgeted query errors."""
from argparse import ArgumentParser
from json import dumps, loads
from os import environ, wait4, waitstatus_to_exitcode
from pathlib import Path
from runpy import run_path
from statistics import median
from subprocess import Popen
from sys import platform
from tempfile import TemporaryFile

parser = ArgumentParser(description=__doc__)
parser.add_argument('binary', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--witness-size', type=int)
args = parser.parse_args()
binary = args.binary.resolve()
args.output.mkdir(parents=True, exist_ok=True)
accounting = run_path(str(Path(__file__).with_name('whir-zk-accounting-2026-11-04.py')))
# v1 budgets are per component; these values give the global query margin checked below.
cases = [
    ('plain-list', 'plain', 130, 148, False, None),
    ('legacy-list', 'legacy-hiding', 132, 148, False, None),
    ('plain-unique', 'plain', 131, 195, True, None),
    ('prefix-list', 'prefix', 128, 148, False, 'list'),
    ('prefix-unique', 'prefix', 128, 195, False, 'unique'),
]
summary = []
with (args.output / 'measurements.jsonl').open('w') as raw:
    for threads in [1, 4, 8]:
        for name, mode, budget, queries, unique, decoding in cases:
            command = [str(binary), '--mode', mode, '--variables', '20', '--folds', '5',
                       '--queries', str(queries), '--ood', '1', '--log-inv-rate', '2',
                       '--base-size', '2048', '--security-bits', str(budget),
                       '--runs', '3', '--verifier-repetitions', '20']
            if unique:
                command.append('--unique-decoding')
            if args.witness_size is not None:
                command.extend(['--witness-size', str(args.witness_size)])
            with TemporaryFile(mode='w+t') as stdout, TemporaryFile(mode='w+t') as stderr:
                child = Popen(command, env=dict(environ, RAYON_NUM_THREADS=str(threads)),
                                         stdout=stdout, stderr=stderr)
                _, status, usage = wait4(child.pid, 0)
                child.returncode = waitstatus_to_exitcode(status)
                stderr.seek(0)
                if child.returncode:
                    raise RuntimeError(f'{name}: {stderr.read()}')
                stdout.seek(0)
                samples = [loads(line) for line in stdout]
            if len(samples) != 3:
                raise RuntimeError(f'{name}: expected three measurements')
            for sample in samples:
                raw.write(dumps(dict(sample, scenario=name)) + '\n')
            raw.flush()
            bound = accounting['check'](samples[0], decoding) if decoding else accounting['baseline_query_bound'](samples[0])
            if not bound['at_most_2_to_minus_129']:
                raise RuntimeError(f'{name}: global query/soundness budget failed')
            peak_bytes = usage.ru_maxrss if platform == 'darwin' else usage.ru_maxrss * 1024
            row = {
                'scenario': name, 'threads': threads, 'runs': 3,
                'prover_ms': median(x['commitment_ms'] + x['claim_evaluation_ms'] + x['opening_ms'] for x in samples),
                'verification_ms': median(x['verification_ms'] for x in samples),
                'proof_kib': median(x['proof_bytes'] for x in samples) / 1024,
                'peak_rss_mib': peak_bytes / 1024**2,
                'accounting': bound, 'command': command,
            }
            summary.append(row)
            (args.output / 'summary.json').write_text(dumps(summary, indent=2) + '\n')
            print(f"{name} threads={threads}: prove={row['prover_ms']:.2f}ms "
                  f"verify={row['verification_ms']:.2f}ms proof={row['proof_kib']:.2f}KiB", flush=True)
