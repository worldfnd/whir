"""Check the selected BN254 interactive bound against benchmark oracle metadata."""
from argparse import ArgumentParser
from json import dumps, loads
from math import isqrt
from fractions import Fraction
from pathlib import Path

Q = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def require(condition):
    if not condition:
        raise ValueError('measurement does not satisfy the selected certificate hypotheses')


def ceil_fraction(value):
    return (value.numerator + value.denominator - 1) // value.denominator


def baseline_query_bound(record):
    require(record['mode'] in ('Plain', 'LegacyHiding'))
    require(record['field_modulus'] == str(Q))
    require(record['variables'] == 20 and record['folds'] == 5)
    require(bool(record['oracles']))
    error = Fraction(0)
    for item in record['oracles']:
        rate = Fraction(item['column_coefficients'], item['rows'])
        if record['baseline_unique_decoding']:
            agreement = (1 + rate) / 2
        else:
            numerator, denominator = isqrt(rate.numerator), isqrt(rate.denominator)
            require(numerator**2 == rate.numerator and denominator**2 == rate.denominator)
            agreement = Fraction(21, 20) * Fraction(numerator, denominator)
        error += agreement**item['configured_in_domain_draws']
    return {
        'scope': 'Global query contribution only; other baseline terms use existing v1 analysis.',
        'at_most_2_to_minus_129': error <= Fraction(1, 2**129),
        'query_error_times_2_to_129': float(error * 2**129),
    }


def check(record, decoding):
    require(decoding in ('unique', 'list'))
    require(type(record['queries']) is int and record['queries'] > 0)
    require(record['field'] == 'BN254 scalar')
    require(record['field_modulus'] == str(Q))
    require(record['mode'] == 'Prefix')
    require(record['variables'] == 20 and record['folds'] == 5)
    oracles = record['oracles']
    require([item['name'] for item in oracles] == [
        'main.0', 'sumcheck_mask.0', 'main.1', 'sumcheck_mask.1',
        'main.2', 'sumcheck_mask.2', 'main.3', 'final_full_mask',
    ])
    main = [item for item in oracles if item['name'].startswith('main.')]
    masks = [item for item in oracles if item['name'].startswith('sumcheck_mask.')]
    require(len(main) == 4 and len(masks) == 3)
    witness_size = record.get('witness_size', 1 << record['variables'])
    require(type(witness_size) is int)
    require(2**19 < witness_size <= 2**20)
    packed = witness_size <= 2**20 - 32 * 256
    geometry = [(32, 32768, 131072), (32, 4096, 16384), (32, 512, 2048), (1, 2048, 8192)] if packed else [
        (32, 65536, 262144), (32, 8192, 32768), (32, 1024, 4096), (1, 4096, 16384)
    ]
    require([(item['columns'], item['column_coefficients'], item['rows']) for item in main] == geometry)
    for item in main:
        require(item['ood_answers'] == 1)
        require(item['prefix_per_column'] == 256)
        require(item['prefix_per_column'] >= item['configured_in_domain_draws'] + 1)
    for item in masks:
        require(item['ood_answers'] == 0)
        require((item['columns'], item['column_coefficients'], item['rows'], item['prefix_per_column']) == (1, 512, 2048, 256))
        require(item['prefix_per_column'] >= item['configured_in_domain_draws'])
    require(oracles[-1]['name'] == 'final_full_mask')
    require(oracles[-1]['columns'] == 1 and oracles[-1]['prefix_per_column'] == 0 and oracles[-1]['ood_answers'] == 0)
    for key in ['column_coefficients', 'coefficient_count', 'rows', 'leaf_bytes']:
        require(oracles[-1][key] == main[-1][key])
    for item in oracles:
        require(item['coefficient_count'] == item['columns'] * item['column_coefficients'])
        require(4 * item['column_coefficients'] == item['rows'])
        require(item['leaf_bytes'] == 32 * (item['columns'] + 2 - item['columns'] % 2))
        require(item['configured_in_domain_draws'] == record['queries'])
    folds = record['folds']
    list_size = 1 if decoding == 'unique' else 40
    radius = Fraction(3, 8) if decoding == 'unique' else Fraction(19, 40)
    for item in oracles:
        distance = item['rows'] - item['column_coefficients'] + 1
        rate = Fraction(item['column_coefficients'] - 1, item['rows'])
        if decoding == 'unique':
            require(2 * radius * item['rows'] < distance)
        else:
            require((1 - rate) / ((1 - radius)**2 - rate) <= list_size)
    sqrt_lower = Fraction(999, 2000) if packed else Fraction(1999, 4000)
    rate_lower = Fraction(511, 2048) if packed else Fraction(999, 4000)
    mu = 4 if decoding == 'unique' else 20
    mu_half = Fraction(2 * mu + 1, 2)
    require(Fraction(1, 2) / (Fraction(1, 2) - radius) <= mu)
    numerator_upper = 2 * mu_half**5 + 3 * mu_half * radius / 4
    h_slope = numerator_upper / (3 * rate_lower * sqrt_lower)
    h_intercept = mu_half / sqrt_lower
    exceptional = []
    for item in main:
        rate = Fraction(item['column_coefficients'] - 1, item['rows'])
        require(rate_lower <= rate < Fraction(1, 4))
        require(sqrt_lower**2 <= rate)
        # The bounds on sqrt(rate) give the asserted upper bound on mu.
        exceptional.append(ceil_fraction(h_slope * item['rows'] + h_intercept))
    field_coefficient = list_size
    query_error = Fraction(0)
    for index, mask in enumerate(masks):
        field_coefficient += (
            folds * exceptional[index]
            + (1 + 2 * folds) * list_size**2
            + list_size
        )
        query_error += list_size * (
            (1 - radius) ** main[index]['configured_in_domain_draws']
            + (1 - radius) ** mask['configured_in_domain_draws']
        )
    final = main[-1]
    final_bits = final['column_coefficients'].bit_length() - 1
    field_coefficient += exceptional[-1] + list_size**2 + 2 * final_bits
    query_error += (1 - radius) ** final['configured_in_domain_draws']
    field_error = Fraction(field_coefficient, Q)
    initial = main[0]
    identification_error = Fraction(
        list_size * (list_size - 1) // 2 * (initial['coefficient_count'] - 1),
        Q - initial['rows'],
    )
    # Each accepted OOD costs <=4Q/2^512; all other internal field draws cost <=Q/2^512.
    field_draws = (
        2 + sum(1 + folds + 2 + main[i]['configured_in_domain_draws']
                + mask['configured_in_domain_draws'] for i, mask in enumerate(masks))
        + 1 + final_bits + 3 * len(main)
    )
    scalar_bias = Fraction(field_draws * Q, 2**512)
    total = field_error + query_error + identification_error + scalar_bias
    leaves = sum(item['rows'] for item in oracles)
    nodes = sum(2 * item['rows'] - 1 for item in oracles)
    return {
        'scope': 'Interactive soundness and initial candidate identification; ideal binding oracles.',
        'decoding': decoding,
        'radius': str(radius),
        'noninteractive_security_certified': False,
        'merkle_privacy_proved': False,
        'queries_per_oracle': record['queries'],
        'list_size_upper_bound': list_size,
        'exceptional_seed_upper_bounds': exceptional,
        'field_error_coefficient': field_coefficient,
        'query_error_coefficient_at_equal_counts': 2 * len(masks) * list_size + 1,
        'at_most_2_to_minus_128': total <= Fraction(1, 2**128),
        'at_most_2_to_minus_129': total <= Fraction(1, 2**129),
        'error_times_2_to_128': float(total * 2**128),
        'error_times_2_to_129': float(total * 2**129),
        'field_challenge_bias_units': field_draws,
        'leaves': leaves,
        'merkle_node_labels': nodes,
    }


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('measurement', type=Path, help='JSONL output from prefix-experiment')
    parser.add_argument('--decoding', choices=['unique', 'list'], default='list')
    args = parser.parse_args()
    record = loads(args.measurement.read_text().splitlines()[0])
    print(dumps(check(record, args.decoding), indent=2))
