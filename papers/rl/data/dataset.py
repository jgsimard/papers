from typing import Dict, Iterable, Tuple, Union

import numpy as np
from flax.core import frozen_dict
from gym.utils import seeding

from papers.types import DataType

DatasetDict = Dict[str, DataType]


def _check_lengths(dataset_dict: DatasetDict, dataset_len: int | None = None) -> int:
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            if dataset_len != item_len:
                msg = "Inconsistent item lengths in the dataset."
                raise ValueError(msg)
        else:
            msg = "Unsupported type."
            raise TypeError(msg)
    return dataset_len


def _subselect(dataset_dict: DatasetDict, index: np.ndarray) -> DatasetDict:
    new_dataset_dict = {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            new_v = _subselect(v, index)
        elif isinstance(v, np.ndarray):
            new_v = v[index]
        else:
            msg = "Unsupported type."
            raise TypeError(msg)
        new_dataset_dict[k] = new_v
    return new_dataset_dict


def _sample(
        dataset_dict: Union[np.ndarray, DatasetDict], index: np.ndarray,
) -> DatasetDict:
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[index]
    if isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, index)
    else:
        msg = "Unsupported type."
        raise TypeError(msg)
    return batch


class Dataset:
    def __init__(self, dataset_dict: DatasetDict, seed: int | None = None) -> None:
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)

        # Seeding similar to OpenAI Gym:
        # https://github.com/openai/gym/blob/master/gym/spaces/space.py#L46
        self._np_random = None
        if seed is not None:
            self.seed(seed)

    @property
    def np_random(self) -> np.random.RandomState:
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: int | None = None) -> list:
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def __len__(self) -> int:
        return self.dataset_len

    def sample(
            self,
            batch_size: int,
            keys: Iterable[str] | None = None,
            index: np.ndarray | None = None,
    ) -> frozen_dict.FrozenDict:
        if index is None:
            if hasattr(self.np_random, "integers"):
                index = self.np_random.integers(len(self), size=batch_size)
            else:
                index = self.np_random.randint(len(self), size=batch_size)

        batch = {}

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], index)
            else:
                batch[k] = self.dataset_dict[k][index]

        return frozen_dict.freeze(batch)

    def split(self, ratio: float) -> Tuple["Dataset", "Dataset"]:
        if not (0.0 < ratio < 1.0) :  # noqa: PLR2004
            msg = "Bad split ratio"
            raise ValueError(msg)

        index = np.arange(len(self), dtype=np.int32)
        self.np_random.shuffle(index)
        train_index = index[: int(self.dataset_len * ratio)]
        test_index = index[int(self.dataset_len * ratio):]

        train_dataset_dict = _subselect(self.dataset_dict, train_index)
        test_dataset_dict = _subselect(self.dataset_dict, test_index)
        return Dataset(train_dataset_dict), Dataset(test_dataset_dict)

    def _trajectory_boundaries_and_returns(self) -> Tuple[list, list, list]:
        episode_starts = [0]
        episode_ends = []

        episode_return = 0
        episode_returns = []

        for i in range(len(self)):
            episode_return += self.dataset_dict["rewards"][i]

            if self.dataset_dict["dones"][i]:
                episode_returns.append(episode_return)
                episode_ends.append(i + 1)
                if i + 1 < len(self):
                    episode_starts.append(i + 1)
                episode_return = 0.0

        return episode_starts, episode_ends, episode_returns

    def filter(  # noqa: A003
            self, percentile: float | None = None, threshold: float | None = None,
    ) -> None :
        if (percentile is None) and (threshold is None):
            msg = "One of 'percentile' or 'threshold should be a number"
            raise ValueError(msg)
        if (percentile is not None) and (threshold is not None):
            msg = "Only one of 'percentile' or 'threshold should be a number"
            raise ValueError(msg)

        (
            episode_starts,
            episode_ends,
            episode_returns,
        ) = self._trajectory_boundaries_and_returns()

        if percentile is not None:
            threshold = np.percentile(episode_returns, 100 - percentile)

        bool_index = np.full((len(self),), fill_value = False, dtype=bool)

        for i in range(len(episode_returns)):
            if episode_returns[i] >= threshold:
                bool_index[episode_starts[i]: episode_ends[i]] = True

        self.dataset_dict = _subselect(self.dataset_dict, bool_index)

        self.dataset_len = _check_lengths(self.dataset_dict)

    def normalize_returns(self, scaling: float = 1000) -> None:
        (_, _, episode_returns) = self._trajectory_boundaries_and_returns()
        self.dataset_dict["rewards"] /= np.max(episode_returns) - np.min(
            episode_returns,
        )
        self.dataset_dict["rewards"] *= scaling
