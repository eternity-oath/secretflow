# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from secretflow import reveal, wait
from secretflow.component.component import CompEvalContext
from secretflow.component.data_utils import SimpleVerticalBatchReader
from secretflow.component.storage import ComponentStorage
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.spec.v1.data_pb2 import (
    DistData,
    StorageConfig,
    TableSchema,
    VerticalTable,
)


def test_works(sf_production_setup_devices):
    alice = sf_production_setup_devices.alice
    bob = sf_production_setup_devices.bob
    scaler = StandardScaler()
    ds = load_breast_cancer()
    x, y = scaler.fit_transform(ds["data"]), ds["target"]

    expected_row_cnt = x.shape[0]
    data_dir = os.path.join("/", "tmp", f"{os.getuid()}")
    paths = {
        'alice': os.path.join("alice", "test_batch_reader", "alice.csv"),
        'bob': os.path.join("bob", "test_batch_reader", "bob.csv"),
    }

    def create_alice_data(p, x, y):
        p = os.path.join(data_dir, p)
        os.makedirs(
            os.path.dirname(p),
            exist_ok=True,
        )
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(p, index=False)

    wait(alice(create_alice_data)(paths["alice"], x, y))

    def create_bob_data(p, x):
        p = os.path.join(data_dir, p)
        os.path.join(data_dir, p)
        os.makedirs(
            os.path.dirname(p),
            exist_ok=True,
        )

        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(p, index=False)

    wait(bob(create_bob_data)(paths["bob"], x))

    input_ds = DistData(
        name="train_dataset",
        type="sf.table.vertical_table",
        data_refs=[
            DistData.DataRef(uri=paths["alice"], party="alice", format="csv"),
            DistData.DataRef(uri=paths["bob"], party="bob", format="csv"),
        ],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                ids=["id1"],
                id_types=["str"],
                feature_types=["float32"] * 15,
                features=[f"a{i}" for i in range(15)],
                labels=["y"],
                label_types=["float32"],
            ),
            TableSchema(
                ids=["id2"],
                id_types=["str"],
                feature_types=["float32"] * 15,
                features=[f"b{i}" for i in range(15)],
            ),
        ],
    )
    input_ds.meta.Pack(meta)

    ctx = CompEvalContext(
        data_dir=data_dir,
        comp_storage=ComponentStorage(
            StorageConfig(
                type="local_fs",
                local_fs=StorageConfig.LocalFSConfig(wd=data_dir),
            )
        ),
    )

    reader = SimpleVerticalBatchReader(
        ctx,
        input_ds,
        partitions_order=["bob", "alice"],
        col_selects=["a3", "b7", "b1", "a1", "a10", "b10"],  # out of order
        batch_size=47,
    )

    def _assert_df(df: VDataFrame, row_cnt):
        assert list(df.partitions.keys())[0].party == "bob"
        alice_nd = reveal(df.partitions[alice].values)
        bob_nd = reveal(df.partitions[bob].values)

        assert bob_nd.shape[0] == alice_nd.shape[0]

        expected_x = x[row_cnt : row_cnt + alice_nd.shape[0], [22, 16, 25, 3, 1, 10]]

        row_cnt += alice_nd.shape[0]

        assert alice_nd.shape[1] == 3

        assert bob_nd.shape[1] == 3

        if df.shape[0]:
            # ordered by partitions_order
            assert df.columns == ["b7", "b1", "b10", "a3", "a1", "a10"]

        assert row_cnt == reader.total_read_cnt

        batch_x = np.concatenate([bob_nd, alice_nd], axis=1)

        np.testing.assert_almost_equal(expected_x, batch_x, decimal=4)

        return row_cnt

    row_cnt = 0
    for df in reader:
        row_cnt = _assert_df(df, row_cnt)
        rand_len = reveal(alice(random.randint)(1, 9))
        df = reader.next(rand_len)
        row_cnt = _assert_df(df, row_cnt)

    assert expected_row_cnt == row_cnt
