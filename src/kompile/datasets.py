from typing import Any, Callable, Iterable, TypeVar, Generic, Sequence, List, Optional, Union
from torchvision import datasets
from kompile.utils import *
        
#famous Dataset
class MNIST(datasets.MNIST):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        trf = ToNumpy() if transform is None else transform
        super().__init__(root=root, train= train, transform = trf, target_transform= target_transform, download =download)

class FashionMNIST(datasets.FashionMNIST):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        trf = ToNumpy() if transform is None else transform
        super().__init__(root=root, train= train, transform = trf, target_transform= target_transform, download =download)
