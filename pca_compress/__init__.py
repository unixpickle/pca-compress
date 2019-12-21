"""
Use PCA to compress neural network initializations.
"""

__all__ = ['LayerLocation', 'SequentialLayerLocation', 'project_module']

from .location import LayerLocation, SequentialLayerLocation
from .project import project_module
