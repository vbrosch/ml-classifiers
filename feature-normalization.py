"""
    Normalize a feature vector so that the mean equals 0 and the standard deviation equals 1
"""
import math

from lina.vectors import to_vec, vector_add, vector_init, vector_dim, vector_multiply, unwrap


def get_dimensionality():
    return vector_dim(samples[0])


def get_mean_vector():
    norm = 1 / len(samples)

    mean = vector_init(dim=get_dimensionality())

    for s in samples:
        x = vector_multiply(s, norm)
        mean = vector_add(mean, x)

    return mean


def get_deviation_vector():
    norm = 1 / (len(samples) - 1)

    variance_vec = [0] * get_dimensionality()

    for j in range(len(variance_vec)):
        variance_vec[j] = math.sqrt(norm * sum([math.pow(unwrap(x)[j] - unwrap(mean_vector)[j], 2) for x in samples]))

    return to_vec(variance_vec)


def normalize():
    n_samples = samples.copy()
    unwrapped_mean = unwrap(mean_vector)
    unwrapped_dev = unwrap(deviation_vector)

    for x in [unwrap(x) for x in n_samples]:
        for j in range(vector_dim(samples[0])):
            x[j] = round((x[j] - unwrapped_mean[j]) / unwrapped_dev[j], ndigits=3)

    return n_samples


samples = [to_vec(x) for x in [[100, 2.2], [98, 2.0], [99, 1.8], [101, 1.6], [102, 2.4], [99, 2.2], [101, 1.8]]]
mean_vector = get_mean_vector()
deviation_vector = get_deviation_vector()

print("Mean-Vector: {}".format(mean_vector))
print("Deviation-Vector: {}".format(deviation_vector))

normalized_samples = normalize()

print("Normalized Samples: {}".format(normalized_samples))
