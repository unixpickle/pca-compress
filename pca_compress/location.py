from abc import ABC, abstractmethod

import torch.nn as nn


def sorted_locations(locations, model, inputs):
    """
    Returns a sorted list of locations from first to last
    in the forward pass of the model.

    Args:
        locations: an unsorted sequence of locations.
        model: an nn.Module.
        inputs: an input batch to feed to the model to
          detect the ordering.
    """
    backups = []
    sorted_locs = []
    for loc in locations:
        module = loc.get_module(model)
        b = module.forward
        backups.append(b)

        # Capture variables loc and b.
        def replace_forward(loc=loc, b=b):
            def new_forward(*x, **y):
                if loc not in sorted_locs:
                    sorted_locs.append(loc)
                return b(*x, **y)
            module.forward = new_forward
        replace_forward()

    try:
        model(inputs)
    finally:
        for loc, b in zip(locations, backups):
            loc.get_module(model).forward = b


class LayerLocation(ABC):
    """
    A LayerLocation represents a position in a network
    where activation values can be captured.
    """

    def layer_values(self, model, inputs):
        """
        Get a Tensor of activation values at the location.

        It is assumed that the second dimension of the
        result is the channel dimension, and all other
        dimensions are spatial or temporal.

        Args:
            model: the relevant module to extract the
              activations for.
            inputs: the input batch, which should be fed
              into the model.
        """
        module = self.get_module(model)
        backup = module.forward

        def new_forward(*x, **y):
            output = backup(*x, **y)
            raise _CaptureException(output)

        module.forward = new_forward

        try:
            model.forward(inputs)
        except _CaptureException as exc:
            return exc.value
        finally:
            module.forward = backup

    def forward(self, model, inputs):
        """
        Like layer_values, but also returns the final
        outputs of the model.

        This may take longer than layer_values(), since a
        full forward pass will always be performed.

        Returns:
            A tuple (outputs, activations).
        """
        module = self.get_module(model)
        backup = module.forward

        output_activations = [None]

        def new_forward(*x, **y):
            output_activations[0] = backup(*x, **y)
            return output_activations[0]

        module.forward = new_forward

        try:
            result = model.forward(inputs)
        finally:
            module.forward = backup

        return result, output_activations[0]

    @abstractmethod
    def get_module(self, model):
        """
        Get an nn.Module representing the linear module
        that produces the output of this layer.
        May be a linear or convolutional layer, as long as
        it has a weight and bias attribute.
        """
        pass

    @abstractmethod
    def set_module(self, model, module):
        """
        Replace the module returned by get_module().
        This can be used to drop in a factorized layer.
        """
        pass


class SequentialLayerLocation(LayerLocation):
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx

    def get_module(self, model):
        return model[self.layer_idx]

    def set_module(self, model, module):
        model[self.layer_idx] = module


class AttributeLayerLocation(LayerLocation):
    def __init__(self, name):
        self.name = name

    @classmethod
    def module_locations(cls, module, module_cls=nn.Conv2d):
        res = []
        for name, sub_module in module.named_modules():
            if not name:
                continue
            if isinstance(sub_module, module_cls):
                res.append(cls(name))
        return res

    def get_module(self, model):
        for x in self._path():
            model = getattr(model, x)
        return model

    def set_module(self, model, m):
        path = self._path()
        for x in path[:-1]:
            model = getattr(model, x)
        setattr(model, path[-1], m)

    def _path(self):
        return self.name.split('.')


class _CaptureException(BaseException):
    def __init__(self, value):
        self.value = value
