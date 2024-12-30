from dataclasses import dataclass

import numpy
import torch


@dataclass
class AccuracyMeter:
    num_correct: int = 0
    num_total: int = 0

    def _astype(self, value):
        match value:
            case int() | float():
                return AccuracyMeter(value, 1)
            case numpy.ndarray():
                return AccuracyMeter(value.sum(), value.shape[0])
            case torch.Tensor():
                return AccuracyMeter(value.sum().item(), value.shape[0])
            case AccuracyMeter():
                return value
            case _:
                raise TypeError(f"Unsupported type: {type(value)}")

    def __add__(self, other):
        other = self._astype(other)
        return AccuracyMeter(
            self.num_correct + other.num_correct, self.num_total + other.num_total
        )

    def __radd__(self, other):
        return self.__add__(other)

    def acc(self):
        return self.num_correct / self.num_total


if __name__ == "__main__":
    import numpy as np
    import torch

    bool_list = [True, False, True, False]

    scores = AccuracyMeter()
    scores += np.array(bool_list)
    scores += torch.tensor(bool_list)
    scores += 1
    scores += 0.0

    # expected value: 5 / 10 = 0.5
    print(scores.acc())
