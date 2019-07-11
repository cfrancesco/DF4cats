import numpy as np
import pandas as pd
from typing import Union, Tuple


class CodedSeries:
    """
    Creates and manages encodings for categorical variables.
    Maps the unique values in series to integers > 0.
    The unique values are ordered, so the mapping is independent wrt permutations of series.

    Args:
        series: values of the categorical the variable.
        others_name: name given to values that are found when trying to encode
            but where not present when the object was created (after hardcoding for instance).
        others_value: value used to encode others_name.
        nan_value: value to use to encode missing values (numpy.nan)
    """

    def __init__(
        self,
        series: Union[pd.Series, list, np.ndarray],
        others_name: str = 'other',
        others_value: int = 0,
        nan_value: int = -1,
    ):
        if not isinstance(series, pd.Series):
            if isinstance(series, list) or isinstance(series, np.ndarray):
                series = pd.Series(series)
            else:
                raise TypeError('series must be either pandas.Series, numpy.ndarray or list')

        # set automatically orders the values
        self.values = [x for x in set(series.unique()) if not pd.isna(x)]

        self.direct_mapping = {value: int_value + 1 for int_value, value in enumerate(self.values)}
        self.inverse_mapping = {int_value + 1: value for int_value, value in enumerate(self.values)}
        if others_value in self.inverse_mapping.keys():
            print(
                f'Warning: label value for "other" category ({others_value}) is present in regular codes'
            )
        self.others_name = others_name
        if self.others_name in self.direct_mapping.keys():
            print(
                f'Warning: label name for "other" category ({self.others_name})  is present in regular values'
            )

        self.nan_value = nan_value
        if any(pd.isna(series.unique())):
            self.direct_mapping[np.nan] = nan_value
            self.values.append(np.nan)

        self.inverse_mapping[nan_value] = np.nan
        self.inverse_mapping[others_value] = self.others_name

        self.num_values = len(self.values)
        self.has_others = False
        self.others_value = others_value

    def get_mapping(self, value):
        """
        Return either the integer corresponding to value or other_value if value is not present in
        the mapping.
        """
        # Needed for more consistent behavior
        # d = pd.DataFrame([np.nan])
        # d[0][0] in [np.nan] -> False
        if pd.isna(value):
            return self.nan_value

        if value in self.direct_mapping:
            return self.direct_mapping[value]

        else:
            if not self.has_others:
                self.has_others = True
                self.num_values += 1
            return self.others_value
