"""
Use PCA to compress neural network initializations.
"""

__all__ = ['LayerLocation', 'SequentialLayerLocation', 'AttributeLayerLocation',
           'project_module']

from .location import LayerLocation, SequentialLayerLocation, AttributeLayerLocation
from .project import project_module
