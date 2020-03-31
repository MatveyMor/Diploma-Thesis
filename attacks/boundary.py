__author__ = 'Matvey Morozov'

import warnings
warnings.filterwarnings("ignore")

import sys
import torch
import numpy as np
from scipy.special import softmax


# ортогональный шаг
def orthogonal_perturbation(delta, prev_sample, target_sample):
    perturb = np.random.randn(28, 28)
    perturb /= np.linalg.norm(perturb, ord=2)
    perturb *= delta * np.linalg.norm(target_sample - prev_sample, ord=2)

    mean = np.mean(prev_sample)

    # Проекция шума на сферу вокруг целевой точки
    diff = target_sample - prev_sample
    diff /= np.linalg.norm(target_sample - prev_sample, ord=2)

    perturb -= np.dot(perturb, diff) * diff

    overflow = (prev_sample + perturb) - np.ones((28, 28)) * (255. - mean)
    perturb -= overflow * (overflow > 0)
    underflow = -(prev_sample + perturb) + np.ones((28, 28)) * (0. - mean)
    perturb += underflow * (underflow > 0)

    return perturb


# прямой шаг
def forward_perturbation(epsilon, prev_sample, target_sample):
    perturb = target_sample - prev_sample
    perturb /= np.linalg.norm(target_sample - prev_sample, ord=2)
    perturb *= epsilon
    return perturb



def boundary_attack(net, initial_image, target_image, verbose=1, threshold=None, max_iter=None):
    if verbose > 0:
        print('Start boundary attack')

    predict_init = net(torch.tensor(initial_image.reshape(1, 1, 28, 28))).data.numpy()
    predict_target = net(torch.tensor(target_image.reshape(1, 1, 28, 28))).data.numpy()
    attack_class = np.argmax(predict_init)
    target_class = np.argmax(predict_target)

    adversarial_sample = initial_image

    n_steps = 0
    n_calls = 0
    epsilon = 1.
    delta = 0.1

    # Шаг 1. Находим проекцию на границу адверсальности
    while True:
        trial_sample = adversarial_sample + forward_perturbation(epsilon * np.linalg.norm(adversarial_sample - target_image, ord=2), adversarial_sample, target_image)
        prediction = net(torch.tensor(trial_sample.reshape(1, 1, 28, 28))).data.numpy()
        n_calls += 1
        if np.argmax(prediction) == attack_class:
            adversarial_last = trial_sample
            distance_last = np.linalg.norm(adversarial_last - target_image, ord=2)
            adversarial_sample = trial_sample
            break
        else:
            epsilon *= 0.9

    # Шаг 2. Дельта шаг
    eps_step = epsilon
    delta_step = delta
    while True:
        d_step = 0
        delta_init = delta
        while True:
            d_step += 1
            trial_samples = []
            for i in np.arange(10):
                trial_sample = adversarial_sample + orthogonal_perturbation(delta, adversarial_sample, target_image)
                trial_samples.append(trial_sample)
            trial_samples = np.array(trial_samples)
            predictions = net(torch.tensor(trial_samples.reshape(10, 1, 28, 28)).to(dtype=torch.float)).data.numpy()
            n_calls += 10
            predictions = np.argmax(predictions, axis=1)
            d_score = np.mean(predictions == attack_class)
            if d_score > 0.0:
                if d_score < 0.3:
                    delta *= 0.9
                elif d_score > 0.7:
                    delta /= 0.9
                adversarial_sample = np.array(trial_samples)[np.where(predictions == attack_class)[0][0]]
                break
            else:
                delta *= 0.9

            if delta < 1e-10:
                delta_init *= 2
                delta += delta_init

        e_step = 0
        while True:
            e_step += 1
            trial_sample = adversarial_sample + forward_perturbation(epsilon * np.linalg.norm(adversarial_sample - target_image, ord=2), adversarial_sample, target_image)
            prediction = net(torch.tensor(trial_sample.reshape(1, 1, 28, 28)).to(dtype=torch.float)).data.numpy()
            n_calls += 1
            if np.argmax(prediction) == attack_class:
                adversarial_sample = trial_sample
                epsilon /= 0.5
                break
            elif e_step > 500:
                break
            else:
                epsilon *= 0.5
        n_steps += 1

        distance = np.linalg.norm(adversarial_sample - target_image, ord=2)

        if distance > distance_last:
            adversarial_sample = adversarial_last
            epsilon = eps_step
            delta = delta_step
        else:
            adversarial_last = adversarial_sample
            distance_last = distance
            delta_step = delta
            eps_step = epsilon

        if verbose > 0:
            sys.stdout.write("\rdistance: %0.4f, itetarion: %d" % (distance_last, n_steps))

        prob_step = np.max(softmax(net(torch.tensor(adversarial_last.reshape(1, 1, 28, 28)).to(dtype=torch.float)).data.numpy()))

        if not threshold is None:
            if distance_last <= threshold:
                break
        if not max_iter is None:
            if n_steps > max_iter:
                break
    print('')
    return adversarial_last, prob_step