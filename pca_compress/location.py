from abc import ABC, abstractmethod
import time

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

    return sorted_locs


def time_locations(locations, model, inputs):
    """
    Times a forward pass of the model for each location
    separately.

    Args:
        locations: a sequence of locations.
        model: an nn.Module.
        inputs: an input batch to feed to the model for
          timing the execution.

    Returns:
        A list containing the time taken for each location
          in the given array. If a location is not hit,
          its time is 0.
    """
    backups = []
    times = [0.0] * len(locations)
    for i, loc in enumerate(locations):
        module = loc.get_module(model)
        b = module.forward
        backups.append(b)

        def replace_forward(i=i, b=b):
            def new_forward(*x, **y):
                t1 = time.time()
                result = b(*x, **y)
                times[i] = time.time() - t1
                return result
            module.forward = new_forward
        replace_forward()

    try:
        model(inputs)
    finally:
        for loc, b in zip(locations, backups):
            loc.get_module(model).forward = b

    return times


class LayerLocation(ABC):
    """
    A LayerLocation represents a position in a network
    where activation values can be captured.
    """

    def layer_values(self, model, inputs, before=False):
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
            before: if True, get the values before the
              location's module, not after.
        """
        module = self.get_module(model)
        backup = module.forward

        def new_forward(*x, **y):
            if before:
                raise _CaptureException(x[0])
            output = backup(*x, **y)
            raise _CaptureException(output)

        module.forward = new_forward

        try:
            model.forward(inputs)
        except _CaptureException as exc:
            return exc.value
        finally:
            module.forward = backup

    def forward(self, model, inputs, before=False, modify=None):
        """
        Like layer_values, but also returns the final
        outputs of the model.

        This may take longer than layer_values(), since a
        full forward pass will always be performed.

        Args:
            model: the model to run.
            inputs: the input batch.
            before: if specified, yield activations before
              the location, not after.
            modify: if not None, this is a lambda which
              modifies the activations before they are
              passed on through the model.

        Returns:
            A tuple (outputs, activations).
        """
        mod_before = (modify if before else None)
        mod_after = (modify if not before else None)
        outputs, befores, afters = self.forward_both(model, inputs,
                                                     modify_before=mod_before,
                                                     modify_after=mod_after)
        if before:
            return outputs, befores
        else:
            return outputs, afters

    def forward_both(self, model, inputs, modify_before=None, modify_after=None):
        """
        Like forward, but returns (result, before, after).
        """
        module = self.get_module(model)
        backup = module.forward

        output_activations = [None, None]

        def new_forward(*x, **y):
            if not x[0].requires_grad:
                x = tuple(v.clone().requires_grad_(True) for v in x)
            if modify_before is not None:
                x = tuple(modify_before(v) for v in x)
            output = backup(*x, **y)
            if modify_after is not None:
                output = modify_after(output)
            output_activations[0] = x[0]
            output_activations[1] = output
            return output

        module.forward = new_forward

        try:
            result = model.forward(inputs)
        finally:
            module.forward = backup

        return result, output_activations[0], output_activations[1]

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

    @abstractmethod
    def next_location(self, model):
        """
        Get the location that immediately follows this one
        in a sequential kind of layer.

        This may return None if no such location is known.
        """
        pass


class NestedLocation(LayerLocation):
    def __init__(self, *locs):
        self.locs = locs

    def get_module(self, model):
        for loc in self.locs:
            model = loc.get_module(model)
        return model

    def set_module(self, model, module):
        for loc in self.locs[:-1]:
            model = loc.get_module(model)
        self.locs[-1].set_module(model, module)

    def next_location(self, model):
        for loc in self.locs[:-1]:
            model = loc.get_module(model)
        loc = self.locs[-1].next_location(model)
        if loc is None:
            return None
        return NestedLocation(*self.locs[:-1], loc)

    def __repr__(self):
        return '/'.join(str(x) for x in self.locs)


class SequentialLayerLocation(LayerLocation):
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx

    def get_module(self, model):
        return model[self.layer_idx]

    def set_module(self, model, module):
        model[self.layer_idx] = module

    def next_location(self, model):
        if self.layer_idx + 1 == len(model):
            return None
        return SequentialLayerLocation(self.layer_idx + 1)

    def __repr__(self):
        return '[%d]' % self.layer_idx


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

    def next_location(self, model):
        # Don't make any assumptions about attribute structure.
        return None

    def _path(self):
        return self.name.split('.')

    def __repr__(self):
        return self.name


class _CaptureException(BaseException):
    def __init__(self, value):
        self.value = value
