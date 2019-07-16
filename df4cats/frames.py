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
        drop_columns: extra columns to drop.
        drop_empty: whether to drop empty cols or not
        drop_single_values: whether to drop columns with a single value or not


    Methods:
        drop_columns: drops columns.
        fillna: fills missing numbers either with a dictionary or iteratively from cli.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        column_names_char_subs: dict = {},
        drop_columns: list = [],
        drop_empty: bool = True,
        drop_single_valued: bool = True,
    ):
        self.data = df.copy()
        self.dropped_columns = []
        self.drop_columns(columns=drop_columns, inplace=True)
        if column_names_char_subs:
            self.standardize_column_names(char_sub=column_names_char_subs)
        if drop_empty:
            col_before = self.data.columns
            self.data = self.data.dropna(axis='columns', how='all')
            self.empty_cols = [x for x in col_before if x not in self.data.columns]
            self.dropped_columns += self.empty_cols
        if drop_single_valued:
            self.single_valued_cols = self.data.columns[self.data.nunique() == 1]
            self.drop_columns(columns=self.single_valued_cols, inplace=True)
        self.data.fillna(value=np.nan, inplace=True)

    def standardize_column_names(self, char_sub: dict) -> None:
        """
        Removes non standard characters in the column names and applies a substition dictionary.
        """

        self.col_rename_dict = {
            x: utils.remove_nonstd_chars(x, char_sub=char_sub) for x in self.data.columns
        }
        self.data.rename(columns=self.col_rename_dict, inplace=True)

    @utils.class_object_inplace_option
    def drop_columns(self, columns):
        """ Drops columns and keeps track of which ones are actually dropped. """
        dropping = [x for x in columns if x in self.data.columns]
        not_dropping = [x for x in columns if x not in self.data.columns]
        self.data.drop(columns=dropping, inplace=True)
        self.dropped_columns.extend(dropping)
        if len(not_dropping) > 0:
            print(f'Could NOT drop {not_dropping}: columns not present.')
        return self

    @utils.class_object_inplace_option
    def fillna(self, fill_dict=None):
        self.data, self.fill_dict = utils.ask_fillna(self.data, fill_dict=fill_dict, inplace=False)
        return self


class CodedDF(CleanDF):
    """
    Create a dataframe that converts categorical variable values to numbers (encoding).
    Maintains direct and indirect mappings and can switch between them.
    Uses utils.maps.CodedSeries to manage the encodings.

    Args:
        df: pandas DataFrame
        ask_cols: if true, activates cli to inspect columns and assign type (continuous, label, categorical, drop)
        categorical_columns: list of columns to be included in categorical. If detect_categorical is False,
            this defines ALL the categorical columns.
        detect_categorical: add to categorical all columns that are not continuous columns nor labels.
        label_columns: columns that regardless of their type, should not be processed.

    Methods:
        encode: applies the direct mapping to obtain the coded df.
        decode: applies the reverse mapping to revert to the original df.
        hardcode_categories: applies dictionary to specifically select values for categorical variables.
        get_dummies: get dummified dataset, optionally with hardcoded and missing values.

    TODO: Add 'Was missing' column.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ask_cols: bool = False,
        categorical_columns: list = [],
        column_names_char_subs: dict = {},
        detect_categorical: bool = True,
        drop_columns: list = [],
        drop_empty=True,
        drop_single_valued=True,
        label_columns: list = [],
    ):
        super().__init__(
            df=df,
            column_names_char_subs=column_names_char_subs,
            drop_columns=drop_columns,
            drop_empty=drop_empty,
            drop_single_valued=drop_single_valued,
        )

        self.label_columns = label_columns

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
    def hardcode_categories(
        self,
        hardcoded_dict: dict,
        add_extra: bool = False,
        drop_missing: bool = True,
        others_name: str = 'other',
        others_value: int = 0,
    ) -> 'CodedDF':
        """
        Filters the categories using only values from the dictionary.
        If a value is not found, it is replaced with others_value (with key others_name).

        Args:
            hardcoded_dict: keys (categorical) are column names, values are the possible
                values that the categorical variable can assume.
            add_extra: if True, adds empty columns for each dictionary key that does
                not appear in the dataframe.
            drop_missing: removes columns in the dataframe that do not appear in the dictionary keys.
            others_name: name given to values that appear in the dataframe but not in the values of
                the dictionary for a given key.
            others_value: integer value to assign as code for others.
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

    def get_dummies(
        self,
        add_empty: bool = False,
        dummy_na: bool = False,
        empty_others: bool = False,
        prefix_sep='_',
    ):
        """
            add_empty: if True, add an empty (0) dummy column for each value that does not appear in
                the dataset (e.g. after hardcoding)
            dummy_na: if True, adds a column for numpy.nan values.
            empty_others: if True, adds a 'other' column for each categorical variable, even if there
                is no 'other' value in the dataframe. Similar to dummy_na.
            prefix_sep: prefix used to separate categorical variable name and value.
                E.g. prefix_sep='_dum_', category='name', value='Francesco'
                -> dummy column name: 'name_dum_Francesco'
        """
        if self.is_encoded:
            self.decode(inplace=True)
        dummified = pd.get_dummies(
            self.data,
            columns=self.categorical_columns,
            prefix=self.categorical_columns,
            prefix_sep=prefix_sep,
            dummy_na=dummy_na,
        )
        if add_empty:
            for cat in self.categorical_columns:
                uniques = self.data[cat].unique()
                missing_values = [
                    x
                    for x in self.categorical_mapping[cat].direct_mapping.keys()
                    if x not in uniques and (not pd.isna(x) or dummy_na)
                ]
                for missing_value in missing_values:
                    dummified[f'{cat}{prefix_sep}{missing_value}'] = 0
                if (empty_others) and (self.categorical_mapping[cat].others_name not in uniques):
                    dummified[f'{cat}{prefix_sep}{self.categorical_mapping[cat].others_name}'] = 0

        return dummified
