import math

import numpy as np

print("Naive Bayes Classificator for male/female classification")
print("See https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Sex_classification for details.")

# arguments

prior_m = 0.5
prior_f = 1 - prior_m


def get_gaussian_probability(x: float, mean: float, sigma_square: float) -> float:
    return (1 / (math.sqrt(2 * math.pi * sigma_square))) * math.exp(-1 * (pow(x - mean, 2) / (2 * sigma_square)))


def get_posterior_probability(prior: float, sample: np.ndarray, means: np.ndarray, variances: np.ndarray):
    # because each feature is conditionally independent of all other features,
    # the posterior probability for the class equals to the product of all feature probabilities

    prob = prior

    for f_i in range(sample.size):
        prob *= get_gaussian_probability(sample.item(f_i), means.item(f_i), variances.item(f_i))

    return prob


def get_variances(c: np.ndarray, means: np.ndarray) -> np.ndarray:
    x_i_squares = np.multiply(1.0 / len(c), np.sum(np.multiply(c, c), axis=0))
    return np.subtract(x_i_squares, np.multiply(means, means))


def get_means(c: np.ndarray) -> np.ndarray:
    return np.multiply(1.0 / len(c), np.sum(c, axis=0))


# data

samples = np.matrix(([6, 180, 12], [5.92, 190, 11],
                     [5.58, 170, 12], [5.92, 165, 10],
                     [5, 100, 6], [5.5, 150, 8],
                     [5.42, 130, 7], [5.75, 150, 9]))

samples_m = samples[:4]
samples_f = samples[4:]

# ~ training by learning means and variances for each class

male_means = get_means(samples_m)
female_means = get_means(samples_f)

male_variances = get_variances(samples_m, male_means)
female_variances = get_variances(samples_f, female_means)

print("P(male)={}".format(prior_m))
print("P(female)={}".format(prior_f))

print("Male means for each feature: {}".format(male_means))
print("Female means for each feature: {}".format(female_means))

print("Male variances for each feature: {}".format(male_variances))
print("Female variances for each feature: {}".format(female_variances))

# classification of unknown sample

validation_sample = np.matrix([[6, 130, 8]])

posterior_male = get_posterior_probability(prior_m, validation_sample, male_means, male_variances)
posterior_female = get_posterior_probability(prior_f, validation_sample, female_means, female_variances)

print("P(x|male)= {}".format(posterior_male))
print("P(x|female)={}".format(posterior_female))

if posterior_male > posterior_female:
    print("x should be classified as male")
elif posterior_female > posterior_male:
    print("x should be classified as female")
else:
    print("the probability for each class is equal")