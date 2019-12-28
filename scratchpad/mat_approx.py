import numpy as np

D = 128
N = 30000


def outer_products(mat, dirs):
    return np.sum(dirs * (dirs @ mat), axis=-1, keepdims=True)


def random_directions():
    dirs = np.random.normal(size=(N, D))
    dirs /= np.sqrt(np.sum(dirs * dirs, axis=-1, keepdims=True))
    return dirs


def random_normal():
    x = np.random.normal(size=(D, D))
    x = x.T @ x
    return x


def approximate_linear(mat, samples):
    scales = np.sqrt(outer_products(mat, samples))
    return (samples * scales).T @ (samples * scales)


def approximate_boosted(mat, samples):
    res = np.zeros_like(mat)
    for sample in samples:
        target = (sample[None] @ mat @ sample[:, None]).item()
        current = (sample[None] @ res @ sample[:, None]).item()
        outer = sample[:, None] @ sample[None]
        res += (target - current) * outer
    return res


def main():
    mat = random_normal()
    samples = random_directions()

    approx = approximate_boosted(mat, samples)

    test_samples = random_directions()
    expected = outer_products(mat, test_samples).flatten()
    actual = outer_products(approx, test_samples).flatten()

    print(np.corrcoef(actual, expected))


main()
