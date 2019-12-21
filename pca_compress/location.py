from abc import ABC, abstractmethod


class LayerLocation(ABC):
    """
    A LayerLocation represents a position in a network
    where activation values can be captured.
    """
    @abstractmethod
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
        pass

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

    def layer_values(self, model, inputs):
        return model[:self.layer_idx + 1](inputs)

    def get_module(self, model):
        return model[self.layer_idx]

    def set_module(self, model, module):
        model[self.layer_idx] = module
