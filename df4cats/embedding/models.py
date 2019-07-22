import pandas as pd
import numpy as np
from keras.layers import Input, Dense, Embedding, concatenate, Reshape, Flatten, Lambda
from keras.layers import SpatialDropout1D, Dropout, Add
from keras.models import Model
import keras.backend as K
from copy import deepcopy
from keras.optimizers import Adam
from typing import Union


class Embedder:
    """
    Given a dataset, builds embeddings for the categorical variables.
    These embeddings are concatenaed with the continuous variables and a row-embedding is returned.
    """

    def __init__(
        self,
        categorical_features: dict,
        continuous_features: list = [],
        n_dense=2,
        nodes_per_dense=500,
        output_dim: int = 100,
        max_embedding_dim=600,
        dropout_rate=0.2,
        residual=False,
    ):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.max_embedding_dim = max_embedding_dim
        self.model = self.build_net(
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            nodes_per_dense=nodes_per_dense,
            residual=residual,
            n_dense=n_dense,
        )
        self.model.name = 'Embedder'

    # def _get_info_from_CodedDF(self, cdf):
    #     self.cat_dimension = {}
    #     self.categorical_features = dict(zip(cdf.categorical_columns, cdf.data.nunique))
    #     self.continuous_features = cdf.continuous_columns
    #     for cat in cdf.categorical_columns:
    #         self.cat_dimension[cat] = cdf.data[cat].nunique()

    def get_input_dictionary(self, data, include_continuous=False):
        di = {}
        if include_continuous:
            columns = np.append(list(self.categorical_features.keys()), self.continuous_features)
        else:
            columns = list(self.categorical_features.keys())
        for cat in columns:
            #             if cat not in self.exclude_columns_from_samples:
            di[cat] = np.array(data[cat])
        return di

    def _get_feature_inputs(self, features: Union[dict, list], embed=True):
        feature_embeds = []
        feature_inputs = []
        assert (
            isinstance(features, dict) or embed == False
        ), 'Provide a dictionary with embedding dimensions.'

        for feature in features:
            feature_input = Input(shape=(1,), name=feature)
            if embed:
                in_dim = features[feature]
                embedding_dim = min(self.max_embedding_dim, round(1.6 * (in_dim) ** 0.56))
                feature_emb = Embedding(
                    output_dim=embedding_dim,
                    input_dim=in_dim + 1,
                    input_length=1,
                    name=feature + '_emb',
                )(feature_input)
                feature_emb = Flatten(name=feature + '_flat_emb')(feature_emb)
                feature_embeds.append(feature_emb)
            feature_inputs.append(feature_input)
        if embed:
            return feature_inputs, feature_embeds
        else:
            return feature_inputs

    def build_net(self, output_dim, dropout_rate, nodes_per_dense, residual, n_dense):
        categorical_inputs, categorical_embeddings = self._get_feature_inputs(
            features=self.categorical_features, embed=True
        )
        if len(self.continuous_features) > 0:
            continuous_inputs = self._get_feature_inputs(
                features=self.continuous_features, embed=False
            )
        else:
            continuous_inputs = []

        feature_inputs = list(np.append(categorical_inputs, continuous_inputs))
        x = concatenate(list(np.append(categorical_embeddings, continuous_inputs)))
        for i in range(n_dense - 1):
            if residual:
                if i == 0:
                    res = Dropout(dropout_rate)(
                        Dense(nodes_per_dense, activation='relu', name=f'Dense_{i}')(x)
                    )
                else:
                    x = Dropout(dropout_rate)(
                        Dense(nodes_per_dense, activation='relu', name=f'Dense_{i}')(res)
                    )
                    res = Add()([res, x])
            else:
                x = Dropout(dropout_rate)(
                    Dense(nodes_per_dense, activation='relu', name=f'Dense_{i}')(x)
                )
        if residual and n_dense > 1:
            out = Dense(nodes_per_dense, activation='sigmoid')(res)
        else:
            out = Dense(output_dim, activation='sigmoid')(x)
        return Model(inputs=feature_inputs, outputs=out)

    def get_embedding_model(self):
        categorical_inputs, categorical_embeddings = self._get_feature_inputs(
            features=self.categorical_features, embed=True
        )
        return Model(inputs=categorical_inputs, outputs=categorical_embeddings)

    # def get_categorical_embeddings(self):
    #     categorical_embeddings = {}
    #     for cat in self.categorical_features:
    #

    def predictions_to_df(self, predictions, dummify=True):
        di = {}
        for i, field in enumerate(self.categorical_features.keys()):
            if dummify:
                col_names = [f'{field}_emb_{x}' for x in range(predictions[i].shape[1])]
                if i == 0:
                    df = pd.DataFrame(predictions[i], columns=col_names)
                else:
                    df = pd.concat((df, pd.DataFrame(predictions[i], columns=col_names)), axis=1)
            else:
                di[field] = list(predictions[i])
                df = pd.DataFrame.from_dict(di)
        return df


class Siamese:
    """ Siamese model from twin. """

    def __init__(self, twin):
        self.twin = deepcopy(twin)
        self.joint = self.build_siamese()

    def build_siamese(self):
        left_input = []
        right_input = []
        for i, inp in enumerate(self.twin.inputs * 2):
            # name splitting to prevent keras from skrewing up the names
            layer_name = '_'.join(inp.name.split('_')[0:-1])
            if i >= len(self.twin.inputs):
                side = 'right'
                right_input.append(Input(shape=inp.shape[1:], name=layer_name + '_' + side))
            else:
                side = 'left'
                left_input.append(Input(shape=inp.shape[1:], name=layer_name + '_' + side))

        encoded_left_input = self.twin(left_input)
        encoded_right_input = self.twin(right_input)

        l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        l1_layer = l1_distance_layer([encoded_left_input, encoded_right_input])
        prediction = Dense(1, activation="sigmoid")(l1_layer)
        siamese_network = Model(inputs=[*left_input, *right_input], outputs=[prediction])
        return siamese_network

    def get_twin(self):
        return self.joint.get_layer(self.twin.name)
