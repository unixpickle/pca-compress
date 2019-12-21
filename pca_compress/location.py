from abc import ABC, abstractmethod


class LayerLocation(ABC):
    """
    A LayerLocation represents a position in a network
    where activation values can be captured.
    """
    @abstractmethod
    def layer_values(self, model, inputs):
        """
        Get a 2-D Tensor of values of the activation.

        The outer dimension of the result may be a
        multiple of the input batch, since activations may
        be repeated across some spatial or temporal
        dimension.

        Args:
            model: the relevant module to extract the
              activations for.
            inputs: the input batch, which should be fed
              into the model.

        Returns:
            An [N x C] Tensor, where N is any value, but
              typically a multiple of the input batch
              size, and C is the number of channels.
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
        result = model[:self.layer_idx + 1](inputs)
        if len(result.shape) > 2:
            # Combine spatial dimensions.
            result = result.view(result.shape[0], result.shape[1], -1)
            result = result.permute(0, 2, 1)
            result = result.view(-1, result.shape[2])
        return result

    def get_module(self, model):
        return model[self.layer_idx]

    def set_module(self, model, module):
        model[self.layer_idx] = module
