import pandas as pd
import numpy as np
from typing import Union, Tuple
from df4cats.utils import utils
from df4cats.utils import maps


class CleanDF:
    """
    Applies very basic operations to the input pandas dataframe:
        dropping empty columns
        dropping columns with single values
        explicitly fills nans with numpy.nan
    Can pass extra columns to drop.

    Args:
        df: pandas dataframe
        column_names_char_subs: maps chars in the column names.
        drop_cols: extra columns to drop.
        drop_empty: whether to drop empty cols or not
        drop_single_values: whether to drop columns with a single value or not
    """

    def __init__(
        self,
        df: pd.DataFrame,
        column_names_char_subs: dict = {},
        drop_cols: list = [],
        drop_empty: bool = True,
        drop_single_valued: bool = True,
    ):
        self.data = df.copy()
        self.dropped_cols = [x for x in drop_cols if x in self.data.columns]
        self.data = self.data.drop(columns=self.dropped_cols)
        if column_names_char_subs:
            self.standardize_column_names(char_sub=column_names_char_subs)
        if drop_empty:
            # drop empty columns
            col_before = self.data.columns
            self.data = self.data.dropna(axis='columns', how='all')
            self.empty_cols = [x for x in col_before if x not in self.data.columns]
            self.dropped_cols += self.empty_cols
        if drop_single_valued:
            # drop single valued columns
            self.single_valued_cols = self.data.columns[self.data.nunique() == 1]
            self.data = self.data.drop(columns=self.single_valued_cols)
            self.dropped_cols += list(self.single_valued_cols)
        self.data.fillna(value=np.nan, inplace=True)

    def standardize_column_names(self, char_sub: dict) -> None:
        """
        Removes non standard characters in the column names and applies a substition dictionary.
        """

        self.col_rename_dict = {
            x: utils.remove_nonstd_chars(x, char_sub=char_sub) for x in self.data.columns
        }
        self.data.rename(columns=self.col_rename_dict, inplace=True)


class CodedDF:
    """
    Create a dataframe that converts categorical variable values to numbers (encoding).
    Maintains direct and indirect mappings and can switch between them.
    Uses utils.maps.CodedSeries to manage the encodings.

    Args:
        df: pandas DataFrame
        ask_cols: if true, activates cli to inspect columns and assign type (continuous, label, categorical, drop)
        categorical_columns: list of columns to be included in categorical. If detect_categorical is False,
            this defines ALL the categorical columns.
        clean: whether to first create a CleanDF or not.
        column_names_char_subs: dictionary to pass to CleanDF to substitute chars in the column names.
        detect_categorical: add to categorical all columns that are not continuous columns nor labels.
        drop_columns: columns to be dropped.
        label_columns: columns that regardless of their type, should not be processed.

    Methods:
        encode: applies the direct mapping to obtain the coded df.
        decode: applies the reverse mapping to revert to the original df.
        hardcode_categories: applies dictionary to specifically select values for categorical variables.
        drop_columns: drops columns.
        fillna: fills missing numbers either with a dictionary or iteratively from cli.

    TODO: Add "Was missing" column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ask_cols: bool = False,
        categorical_columns: bool = [],
        clean: bool = True,
        column_names_char_subs: dict = {},
        detect_categorical: bool = True,
        drop_columns: bool = [],
        label_columns: bool = [],
    ):

        self.dropped_columns = drop_columns
        self.label_columns = label_columns

        if clean:
            clean_df = CleanDF(
                df, drop_cols=drop_columns, column_names_char_subs=column_names_char_subs
            )
            self.dropped_columns.extend(clean_df.dropped_cols)
            self.data = clean_df.data
        else:
            self.data = df.copy()
            self.drop_columns(drop_columns, inplace=True)

        if ask_cols:
            self.categorical_columns, drop_columns, label_columns, self.continuous_columns = utils.ask_column_types(
                self.data
            )
            self.label_columns.extend(label_columns)
            self.drop_columns(drop_columns, inplace=True)
        else:  # automatically detect continuous columns
            self.continuous_columns = [
                x
                for x in self.data._get_numeric_data().columns
                if x not in categorical_columns and (x not in label_columns)
            ]
            if detect_categorical:  # automatically detect categorical columns
                self.categorical_columns = [
                    x
                    for x in self.data.columns
                    if (x not in self.continuous_columns) and (x not in label_columns)
                ]
            else:
                self.categorical_columns = []

        self.categorical_columns.extend(
            [
                x
                for x in categorical_columns
                if (x in self.data.columns) and (x not in self.categorical_columns)
            ]
        )

        self.categorical_mapping = {}
        for cat in self.categorical_columns:
            self.categorical_mapping[cat] = maps.CodedSeries(self.data[cat])
        self.is_encoded = False
        self.encode(inplace=True)

    @utils.class_object_inplace_option
    def encode(self) -> 'CodedDF':
        if not self.is_encoded:
            for cat in self.categorical_columns:
                self.data[cat] = self.data[cat].map(
                    lambda x: self.categorical_mapping[cat].get_mapping(x)
                )
            self.is_encoded = True
        return self

    @utils.class_object_inplace_option
    def decode(self) -> 'CodedDF':
        """ Returns a CodedDF with the original columns values, before encodings. """
        if self.is_encoded:
            for cat in self.categorical_columns:
                self.data[cat] = self.data[cat].map(self.categorical_mapping[cat].inverse_mapping)
            self.is_encoded = False
        return self

    @utils.class_object_inplace_option
    def drop_columns(self, column_names):
        dropping = [x for x in column_names if x in self.data.columns]
        not_dropping = [x for x in column_names if x not in self.data.columns]
        self.data.drop(columns=dropping, inplace=True)
        self.dropped_columns.extend(dropping)
        if len(not_dropping) > 0:
            print(f'Could NOT drop {not_dropping}: columns not present.')
        return self

    @utils.class_object_inplace_option
    def hardcode_categories(
        self,
        hardcoded_dict: dict,
        others_name: str = 'other',
        others_value: int = 0,
        add_extra: bool = False,
        drop_missing: bool = True,
    ) -> 'CodedDF':
        """
        Filters the categories using only values from the dictionary.
        If a value is not found, it is replaced with others_value (with key others_name).
        """

        if self.is_encoded:
            self.decode(inplace=True)

        if drop_missing:
            self.categorical_mapping = {}
            discard_columns = [
                x for x in self.categorical_columns if x not in hardcoded_dict.keys()
            ]
            if len(discard_columns) > 0:
                print(f'Removing not present in dictionary:\n{discard_columns}')
            self.drop_columns(discard_columns, inplace=True)
            self.categorical_columns = [
                x for x in self.categorical_columns if x not in discard_columns
            ]

        if add_extra:
            extra_columns = [x for x in hardcoded_dict.keys() if x not in self.data.columns]
            empty = pd.DataFrame(
                [[np.nan] * len(extra_columns)] * len(self.data), columns=extra_columns
            )
            self.data = pd.concat([self.data, empty], axis=1)
            if len(extra_columns) > 0:
                print(f'Added (empty) columns previously not present:\n{extra_columns}')
            self.categorical_columns.extend(
                [x for x in hardcoded_dict.keys() if x not in self.categorical_columns]
            )

        for cat in hardcoded_dict.keys():
            # !! add mapping also for columns NOT IN self.categorical_columns !!
            self.categorical_mapping[cat] = maps.CodedSeries(
                hardcoded_dict[cat], others_value=others_value, others_name=others_name
            )

        self.encode(inplace=True)

        return self

    @utils.class_object_inplace_option
    def fillna(self, fill_dict=None):
        self.data, self.fill_dict = utils.ask_fillna(self.data, fill_dict=fill_dict, inplace=False)
        return self

    def get_dummies(
        self, add_empty: bool = False, dummy_na: bool = False, empty_others: bool = False
    ):
        if self.is_encoded:
            self.decode(inplace=True)
        dummified = pd.get_dummies(
            self.data,
            columns=self.categorical_columns,
            prefix=self.categorical_columns,
            prefix_sep='_',
            dummy_na=dummy_na,
        )
        # due to e.g. hardcoding, there could be values that do not appear in self.data
        if add_empty:
            for cat in self.categorical_columns:
                uniques = self.data[cat].unique()
                missing_values = [
                    x
                    for x in self.categorical_mapping[cat].direct_mapping.keys()
                    if x not in uniques and (not pd.isna(x) or dummy_na)
                ]
                for missing_value in missing_values:
                    dummified[f'{cat}_{missing_value}'] = 0
                if (empty_others) and (self.categorical_mapping[cat].others_name not in uniques):
                    dummified[f'{cat}_{self.categorical_mapping[cat].others_name}'] = 0

        return dummified
