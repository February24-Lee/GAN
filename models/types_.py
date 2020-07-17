from typing import List, Callable, Union, Any, TypeVar, Tuple

Tensor = TypeVar('tf.tensor')
Layer = TypeVar('tfk.layer')
DirectoryIterator = TypeVar('keras_preprocessing.image.directory_iterator.DirectoryIterator')
Optimizer = TypeVar('tfk.optimizer')

Gen_loss = TypeVar('generate loss')
Disc_loss = TypeVar('disc loss')