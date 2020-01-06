import time

import numpy as np
import torch
import torchvision.models as models

from pca_compress import AttributeLayerLocation, sorted_locations, time_locations

ARCHITECTURE = 'resnet18'


def main():
    model = models.__dict__[ARCHITECTURE]()
    locations = AttributeLayerLocation.module_locations(model)
    dummy_inputs = torch.randn(10, 3, 224, 224)
    locations = sorted_locations(locations, model, dummy_inputs[:1])
    t1 = time.time()
    times = time_locations(locations, model, dummy_inputs)
    total_time = time.time() - t1

    for loc, layer_time in zip(locations, times):
        module = loc.get_module(model)
        shape = module.weight.shape
        num_params = np.prod(shape)
        print('layer %s: parameters=%d time_frac=%f' %
              (str(loc), num_params, layer_time/total_time))


if __name__ == '__main__':
    main()
