import pandas as pd
import numpy as np
import logging

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
RANDOM_STATE = 42
CAT_COLS = [
    "merch", "cat_id", "name_1", "name_2",
    "gender", "street", "one_city", "us_state",
    "post_code", "jobs",
]

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r

def encode_categoricals(train_df=None, test_df=None, encoders=None):
    if encoders is None:
        encoders = {}

    for col in CAT_COLS:
        # - fit --------------------------------------------------------
        if col not in encoders:
            le = LabelEncoder()
            le.fit(train_df[col].astype(str))
            le.classes_ = np.append(le.classes_, "<UNK>")
            encoders[col] = le

        le = encoders[col]

        # - transform --------------------------------------------------------
        if train_df is not None and col in train_df.columns and train_df is not test_df:
            train_df[col] = le.transform(train_df[col].astype(str))

        if test_df is not None and col in test_df.columns:
            arr = test_df[col].astype(str).to_numpy()

            unk_mask = ~np.isin(arr, le.classes_)
            if unk_mask.any():
                arr[unk_mask] = "<UNK>"

            test_df[col] = le.transform(arr)

    return encoders

def add_time_features(df):
    logger.debug('Adding time features...')
    df = df.copy()

    df["transaction_time"] = pd.to_datetime(df["transaction_time"], errors="coerce")
    if df["transaction_time"].isna().any():
        logger.warning("Found %d unparsable transaction_time values.", df["transaction_time"].isna().sum())

    df["trans_year"] = df["transaction_time"].dt.year
    df["trans_month"] = df["transaction_time"].dt.month
    df["trans_day"] = df["transaction_time"].dt.day
    df["trans_dayofweek"] = df["transaction_time"].dt.dayofweek
    df["trans_hour"] = df["transaction_time"].dt.hour
    df.drop(columns=["transaction_time"], inplace=True)
    return df


def load_train_data():
    logger.info('Loading training data...')

    train = pd.read_csv('./train_data/train.csv')
    logger.info('Raw train data imported. Shape: %s', train.shape)

    train = add_time_features(train)
    train['distance'] = haversine(
        train['lat'], train['lon'], train['merchant_lat'], train['merchant_lon'])

    encoders = encode_categoricals(train)
    logger.info('Train data processed. Shape: %s', train.shape)
    return train, encoders


def run_preproc(train, input_df, encoders: dict):
    continuous_cols = ['amount', 'population_city']

    encode_categoricals(test_df=input_df, encoders=encoders)
    logger.info('Categorical merging completed. Output shape: %s', input_df.shape)
    
    input_df = add_time_features(input_df)
    input_df.reset_index(drop=True, inplace=True)
    logger.info('Added time features. Output shape: %s', input_df.shape)

    input_df['distance'] = haversine(
        input_df['lat'], input_df['lon'],
        input_df['merchant_lat'], input_df['merchant_lon'])
    continuous_cols.append('distance')

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(train[continuous_cols])

    cont_imputed = pd.DataFrame(imputer.transform(input_df[continuous_cols]),
                                columns=continuous_cols).reset_index(drop=True)
    out_df = pd.concat(
            [input_df.drop(columns=continuous_cols).reset_index(drop=True),
            cont_imputed],
            axis=1,
            ignore_index=False)
    
    for col in continuous_cols:
        out_df[col + '_log'] = np.log(out_df[col] + 1)
        out_df.drop(columns=col, inplace=True)

    logger.info('Continuous features done. Shape: %s', out_df.shape)
    return out_df