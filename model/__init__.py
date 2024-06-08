import random
import numpy as np
import torch

# Random seed to reproduce embeddings on each run
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)