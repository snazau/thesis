import numpy as np


def unifrom_segments_sample(size, segments):
    prob = np.array([segment['end'] - segment['start'] for segment in segments])
    prob = prob / prob.sum()
    print(f'prob = {prob}')

    sample = np.array(
        [
            np.random.choice(
                [
                    np.random.uniform(segment['start'], segment['end'])
                    for segment in segments
                ],
                p=prob,
            ) for _ in range(size)
        ]
    )
    return sample


if __name__ == '__main__':
    n = 10000
    segments = [
        {
            'start': 1,
            'end': 3,
        },
        {
            'start': 5,
            'end': 6,
        },
        {
            'start': 8,
            'end': 9,
        }
    ]
    sample = unifrom_segments_sample(n, segments)
    empirical_distribution = [
        sample[np.logical_and(sample > segment['start'], sample < segment['end'])].size / n
        for segment in segments
    ]
    print(f'empirical_distribution: \n{empirical_distribution}')
