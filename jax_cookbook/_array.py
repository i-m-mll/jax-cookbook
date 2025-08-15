from collections.abc import Sequence
from typing import Literal, Optional, TypeVar
from equinox import Module, field
import jax.tree as jt
from jaxtyping import Array, ArrayLike, PyTree

import jax_cookbook.tree as jtree
from jax_cookbook._func import is_type


class ArrayLikeWrapper(Module):
    """Metadata-carrying wrapper for ArrayLike objects.
    
    Metadata fields are static, and thus excluded from JAX transformations. Any modifications to 
    the metadata to remain valid after such transformations, must be implemented separately. 
    """
    value: ArrayLike   
    axes_names: Optional[Sequence[str]] = field(default=None, static=True)
    label: Optional[str | Sequence[str]] = field(default=None, static=True)
    
    def __check_init__(self):
        if self.axes_names is not None:
            if isinstance(self.value, Array):
                if len(self.axes_names) < self.value.ndim:
                    if Ellipsis not in (self.axes_names[0], self.axes_names[-1]):
                        raise ValueError(
                            f"axes_names length ({len(self.axes_names)}) does not match number of "
                            "axes in value ({self.value.ndim}), nor is there a leading/trailing "
                            "ellipsis"
                        )
                    if self.axes_names.count(Ellipsis) > 1:
                        raise ValueError(
                            "axes_names can only contain a single ellipsis (...) at either its start "
                            "or end."
                        )
                elif len(self.axes_names) > self.value.ndim:
                    raise ValueError(
                        f"axes_names length ({len(self.axes_names)}) cannot exceed number of axes "
                        f"in value ({self.value.ndim})"
                    )
                elif Ellipsis in self.axes_names:
                    #? Replace with None?
                    pass
            else:
                raise ValueError("axes_names should not be provided for non-array values")
                
            raise ValueError(
                f"Length of axes_names ({len(self.axes_names)}) must match the number of "
                "dimensions in value ({self.value.ndim})"
            )
            
        if self.label is not None:
            if (
                not isinstance(self.label, str)
                and not (
                    isinstance(self.label, Sequence) 
                    and all(isinstance(x, str) for x in self.label)
                )
            ):
                raise TypeError(f"label must be a string or sequence of strings; got {type(self.label)}")
    

def unwrap_arraylikes(tree: PyTree[ArrayLikeWrapper]) -> PyTree[ArrayLike]:
    """Unwrap ArrayLikeWrapper objects in a PyTree."""
    return jt.map(lambda x: x.value, tree, is_leaf=is_type(ArrayLikeWrapper))


T = TypeVar('T')


def unwrap_arraylikes_and_labels(
    tree: PyTree[ArrayLikeWrapper, 'T'],
    label_fmt: Literal['short', 'medium', 'full'] = 'medium',
) -> tuple[PyTree[ArrayLike, 'T'], PyTree[str, 'T']]:
    """Unwrap ArrayLikeWrapper objects in a PyTree and return both values and labels."""
    return jtree.unzip(jt.map(
        lambda x: (x.value, getattr(x.label, label_fmt)), 
        tree, 
        is_leaf=is_type(ArrayLikeWrapper),
    ))