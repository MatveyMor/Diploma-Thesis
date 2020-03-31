__author__ = 'Matvey Morozov'

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from scipy.special import softmax
import sys
import utils


def normalize(x, name):
    return utils.apply_normalization(x, name)


def get_probs(model, x, y, name):
    output = model(normalize(x, name))
    probs = torch.nn.Softmax()(output)[:, y]
    return torch.diag(probs.data)


def simba_single(model, x, y, verbose=1, epsilon=0.0001, name='cifar'):
    if verbose > 0:
        print('Start simple black-box attack...')

    eps_init = epsilon
    n_dims = x.view(1, -1).size(1)
    last_prob = get_probs(model, x, y, name)
    init_label = np.argmax(model(x).data.numpy())
    iteration = 0
    difference = torch.zeros(n_dims).view(x.size())
    while True:
        diff = torch.zeros(n_dims)
        diff[torch.randint(0, x.size(0) * x.size(1) * x.size(2) * x.size(3), (1,))] = epsilon
        left_prob = get_probs(model, (x - diff.view(x.size())).clamp(0, 1), y,  name)
        if left_prob < last_prob:
            x = (x - diff.view(x.size())).clamp(0, 1)
            last_prob = left_prob
            difference += diff.view(x.size())
        else:
            right_prob = get_probs(model, (x + diff.view(x.size())).clamp(0, 1), y, name)
            if right_prob < last_prob:
                x = (x + diff.view(x.size())).clamp(0, 1)
                last_prob = right_prob
                difference += diff.view(x.size())

        adv_label = np.argmax(model(x).data.numpy())
        prob = np.max(softmax(model(x).data.numpy()[0])) #probability of the decision



        iteration += 1

        if iteration % 2000 == 0 and iteration > 0:
            epsilon += eps_init

        distance = np.linalg.norm(difference.data.numpy())

        if verbose > 0:
            sys.stdout.write("\rIteration: %d, epsilon: %0.5f, probability: %0.3f, init_label: %d, adv_label: %d, distance: %f"
                             % (iteration, epsilon, prob, init_label, adv_label, distance))

        if (init_label != adv_label) or iteration > 3e4:
            if verbose > 0:
                print('')
                print('Attack is succesful! Victory!')
            break

        if iteration > 2e4:
            if verbose > 0:
                print('')
                print('Attack is not succesful! Be careful!')
            break

    return x, difference