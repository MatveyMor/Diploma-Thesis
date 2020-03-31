__author__ = 'Matvey Morozov'

import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import sys


def prepare_data_grad(model, example, example_target):
    loss = torch.nn.CrossEntropyLoss()
    output = model(example)
    loss_val = loss(output, example_target)
    model.zero_grad()
    loss_val.backward()
    data_grad = example.grad.data
    return data_grad


def ifgsm_attack(model, image, target, epsilon, verbose=1):
    if verbose > 0:
        print('Start ITERATIVE fast gradient sign attack...')

    init_label = np.argmax(model(image).data.numpy())
    iteration = 0
    image.requires_grad = True
    while True:
        data_grad = prepare_data_grad(model, image, target)
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        adv_label = np.argmax(model(perturbed_image).data.numpy())
        image = torch.tensor(perturbed_image, requires_grad=True)

        if verbose > 0:
            sys.stdout.write("\rIteration: %d, epsilon: %0.5f, init_label: %d, adv_label: %d"
                         % (iteration, epsilon, init_label, adv_label))

        if init_label != adv_label:
            if verbose > 0:
                print('')
                print('Attack is succesful!')
            break

        iteration += 1

        if iteration > 2e4:
            if verbose > 0:
                print('')
                print('Attack is not succesful. Be careful!')
            break
    return perturbed_image, torch.clamp(epsilon * sign_data_grad, 0, 1)

