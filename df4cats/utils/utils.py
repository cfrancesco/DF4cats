import string
import pandas as pd
from copy import deepcopy
from typing import Tuple


def remove_nonstd_chars(input_string: str, specials: str = '_.-/]', char_sub: dict = {}):
    """
    Removes or replaces (if a dictionary is given) character that do not appear
    in a..z A..Z 1..9 and specials.
    """

    allowed = string.ascii_letters + string.digits + specials
    ungood = [x for x in input_string if x not in allowed]
    for char in ungood:
        if char in char_sub:
            input_string = input_string.replace(char, char_sub[char])
        else:
            input_string = input_string.replace(char, '')
    return input_string


def df_inplace_option(func):
    def wrapper(df, *args, inplace=False, **kwargs):
        if inplace:
            func(df=df, *args, **kwargs)
        else:
            data = df.copy()
            return func(df=data, *args, **kwargs)

    return wrapper


def class_object_inplace_option(func):
    def wrapper(self, *args, inplace=False, **kwargs):
        if inplace:
            func(self, *args, **kwargs)
        else:
            new_self = deepcopy(self)
            return func(new_self, *args, **kwargs)

    return wrapper


def ask_column_types(df: pd.DataFrame) -> Tuple[list, list, list, list]:
    actions = {}
    for col in df.columns:
        print(col)
        print('null values:', df[col].isna().sum() / len(df))
        print('unique values:', df[col].nunique())
        print('sample values:', df[col].unique()[0:6])
        action = int(
            input('0 drop, 1 encode (categorical), 2 continuous, 3 label (other/DO NOT encode)')
        )
        actions[col] = action
        print()
    categorical_columns = []
    drop_cols = []
    label_columns = []
    continuous_columns = []
    for col in actions:
        if actions[col] == 0:
            drop_cols.append(col)
        if actions[col] == 1:
            categorical_columns.append(col)
        if actions[col] == 2:
            continuous_columns.append(col)
        if actions[col] == 3:
            label_columns.append(col)
    return categorical_columns, drop_cols, label_columns, continuous_columns


@df_inplace_option
def ask_fillna(df: pd.DataFrame, fill_dict=None) -> 'CodedDF':
    if fill_dict:
        df.fillna(fill_dict, inplace=True)
    else:
        fill_dict = {}
        for col in df.columns:
            if df[col].isna().sum() > 0:
                print(col)
                print('null values:', df[col].isna().sum() / len(df))
                print('unique values:', df[col].nunique())
                print('sample values:', df[col].unique()[0:6])
                na_value = input('input fill value ("nan" to skip, "mean" for column mean)')
                if not na_value == 'nan':
                    if na_value == 'mean':
                        df[col].fillna(value=df[col].mean(), inplace=True)
                        fill_dict[col] = 'mean'
                    else:
                        df[col].fillna(value=float(na_value), inplace=True)
                        fill_dict[col] = float(na_value)

            print()
    return df, fill_dict
