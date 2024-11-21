import sklearn.datasets
import matplotlib.pyplot as plt
import torch
import numpy as np
import random


def make_dataset(version=None, test=False):
    if test:
        random_state = None
    else:
        random_states = [27,33,38]
        if version is None:
            version = random.choice(range(len(random_states)))
            print(f"Dataset number: {version}")
        random_state = random_states[version]
    return sklearn.datasets.make_circles(factor=0.7, noise=0.1, random_state=random_state)
