"""
Use PCA to compress neural network initializations.
"""

__all__ = ['LayerLocation', 'SequentialLayerLocation', 'AttributeLayerLocation',
           'project_module', 'wrap_module_baseline']

from .location import LayerLocation, SequentialLayerLocation, AttributeLayerLocation
from .modules import wrap_module_baseline
from .project import project_module
