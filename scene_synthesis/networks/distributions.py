import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.distributions.mixture_same_family import MixtureSameFamily


def get_guassian_mixture(weights_logit: torch.tensor,
                         )
