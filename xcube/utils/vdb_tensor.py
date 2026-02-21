"""Lightweight VDBTensor compatible with fvdb-core 0.3.0."""
import torch
import fvdb


class VDBTensor:
    """Thin wrapper around GridBatch + JaggedTensor feature data."""
    def __init__(self, grid: fvdb.GridBatch, feature: fvdb.JaggedTensor, kmap=None):
        self._grid = grid
        self._feature = feature
        self._kmap = kmap

    @property
    def grid(self):
        return self._grid

    @property
    def feature(self):
        return self._feature

    @property
    def kmap(self):
        return self._kmap

    @property
    def device(self):
        return self._grid.device

    @property
    def jidx(self):
        return self._feature.jidx

    def __add__(self, other):
        if isinstance(other, VDBTensor):
            return VDBTensor(
                self._grid,
                self._grid.jagged_like(self._feature.jdata + other.feature.jdata),
                self._kmap,
            )
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return VDBTensor(
                self._grid,
                self._grid.jagged_like(self._feature.jdata * other),
                self._kmap,
            )
        if isinstance(other, torch.Tensor):
            return VDBTensor(
                self._grid,
                self._grid.jagged_like(self._feature.jdata * other),
                self._kmap,
            )
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def to_dense(self):
        return self._grid.to_dense(self._feature)

    @classmethod
    def cat(cls, tensors, dim=1):
        """Concatenate VDBTensors along feature dimension."""
        assert len(tensors) > 0
        grid = tensors[0].grid
        features = torch.cat([t.feature.jdata for t in tensors], dim=dim)
        return cls(grid, grid.jagged_like(features), tensors[0].kmap)
