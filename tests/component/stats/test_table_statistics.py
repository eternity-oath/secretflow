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

import logging

import pandas as pd

from secretflow.component.data_utils import DistDataType
from secretflow.component.stats.table_statistics import (
    gen_table_statistic_report,
    table_statistics_comp,
)
from secretflow.component.storage import ComponentStorage
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import (
    DistData,
    IndividualTable,
    TableSchema,
    VerticalTable,
)
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from secretflow.spec.v1.report_pb2 import Report
from secretflow.stats.table_statistics import table_statistics


def test_table_statistics_comp(comp_prod_sf_cluster_config):
    """
    This test shows that table statistics works on both pandas and VDataFrame,
        i.e. all APIs align and the result is correct.
    """
    alice_input_path = "test_table_statistics/alice.csv"
    bob_input_path = "test_table_statistics/bob.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

    test_data = pd.DataFrame(
        {"a": [9, 6, 5, 5], "b": [5, 5, 6, 7], "c": [1, 1, 2, 4], "d": [11, 55, 1, 99]}
    )
    test_data = test_data.astype("float32")

    if self_party == "alice":
        df_alice = test_data.iloc[:, :2]
        df_alice.to_csv(comp_storage.get_writer(alice_input_path), index=False)
    elif self_party == "bob":
        df_bob = test_data.iloc[:, 2:]
        df_bob.to_csv(comp_storage.get_writer(bob_input_path), index=False)

    param = NodeEvalParam(
        domain="stats",
        name="table_statistics",
        version="0.0.2",
        attr_paths=["input/input_data/features"],
        attrs=[
            Attribute(ss=["a", "b", "c", "d"]),
        ],
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                ],
            )
        ],
        output_uris=[""],
    )
    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["float32", "float32"],
                features=["a", "b"],
            ),
            TableSchema(
                feature_types=["float32", "float32"],
                features=["c", "d"],
            ),
        ],
    )
    param.inputs[0].meta.Pack(meta)

    res = table_statistics_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )
    comp_ret = Report()
    res.outputs[0].meta.Unpack(comp_ret)
    logging.info(comp_ret)
    assert comp_ret == gen_table_statistic_report(table_statistics(test_data))


def test_table_statistics_individual_comp(comp_prod_sf_cluster_config):
    """
    This test shows that table statistics works on both pandas and VDataFrame,
        i.e. all APIs align and the result is correct.
    """
    alice_input_path = "test_table_statistics/alice.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

    test_data = pd.DataFrame(
        {"a": [9, 6, 5, 5], "b": [5, 5, 6, 7], "c": [1, 1, 2, 4], "d": [11, 55, 1, 99]}
    )
    test_data = test_data.astype(dtype="float32")

    if self_party == "alice":
        df_alice = test_data
        df_alice.to_csv(comp_storage.get_writer(alice_input_path), index=False)

    param = NodeEvalParam(
        domain="stats",
        name="table_statistics",
        version="0.0.2",
        attr_paths=["input/input_data/features"],
        attrs=[
            Attribute(ss=["a", "b", "c", "d"]),
        ],
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.INDIVIDUAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv")
                ],
            )
        ],
        output_uris=[""],
    )
    meta = IndividualTable(
        schema=TableSchema(
            feature_types=["float32", "float32", "float32", "float32"],
            features=["a", "b", "c", "d"],
        )
    )
    param.inputs[0].meta.Pack(meta)

    res = table_statistics_comp.eval(
        param=param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    comp_ret = Report()
    res.outputs[0].meta.Unpack(comp_ret)
    logging.info(comp_ret)
    assert comp_ret == gen_table_statistic_report(table_statistics(test_data))
