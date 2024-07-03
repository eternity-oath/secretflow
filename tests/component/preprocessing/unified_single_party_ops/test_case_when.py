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

import pandas as pd
from google.protobuf.json_format import MessageToJson

from secretflow.component.data_utils import DistDataType
from secretflow.component.preprocessing.unified_single_party_ops.case_when import (
    case_when,
)
from secretflow.component.preprocessing.unified_single_party_ops.substitution import (
    substitution,
)
from secretflow.component.storage import ComponentStorage
from secretflow.spec.extend.case_when_rules_pb2 import CaseWhenRule
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam


def _build_test():
    def build_value(val, col=False):
        v = CaseWhenRule.ValueExpr()
        if col:
            v.type = CaseWhenRule.ValueExpr.ValueType.COLUMN
            v.column_name = val
        elif isinstance(val, str):
            v.type = CaseWhenRule.ValueExpr.ValueType.CONST_STR
            v.s = val
        elif isinstance(val, float):
            v.type = CaseWhenRule.ValueExpr.ValueType.CONST_FLOAT
            v.f = val
        elif isinstance(val, int):
            v.type = CaseWhenRule.ValueExpr.ValueType.CONST_INT
            v.i = val
        return v

    def build_cond(c, o, val, col=False):
        cond = CaseWhenRule.Cond()
        cond.cond_column = c
        cond.op = o
        cond.cond_value.CopyFrom(build_value(val, col))
        return cond

    names = []
    tests = []
    expected = []

    # --------TEST---------
    r = CaseWhenRule()

    when = CaseWhenRule.When()
    # when ids == [1,3,5]
    when.conds.append(build_cond("id1", CaseWhenRule.Cond.CondOp.EQ, "1"))
    when.connections.append(CaseWhenRule.When.ConnectType.OR)
    when.conds.append(build_cond("id1", CaseWhenRule.Cond.CondOp.EQ, "3"))
    when.connections.append(CaseWhenRule.When.ConnectType.OR)
    when.conds.append(build_cond("id1", CaseWhenRule.Cond.CondOp.EQ, "5"))
    # then 1
    when.then.CopyFrom(build_value(1))
    r.whens.append(when)

    when = CaseWhenRule.When()
    # when ids == [2,4,6]
    when.conds.append(build_cond("id1", CaseWhenRule.Cond.CondOp.EQ, "2"))
    when.connections.append(CaseWhenRule.When.ConnectType.OR)
    when.conds.append(build_cond("id1", CaseWhenRule.Cond.CondOp.EQ, "4"))
    when.connections.append(CaseWhenRule.When.ConnectType.OR)
    when.conds.append(build_cond("id1", CaseWhenRule.Cond.CondOp.EQ, "6"))
    # then 2
    when.then.CopyFrom(build_value(2))
    r.whens.append(when)

    # else 0
    r.else_value.CopyFrom(build_value(0))

    r.as_label = False
    r.float_epsilon = 1e-07
    r.output_column = "z"

    names.append("test1")
    tests.append(r)
    expected.append([0, 1, 2, 1, 2, 1, 2, 0, 0, 0])

    # --------TEST---------
    r = CaseWhenRule()

    when = CaseWhenRule.When()
    # a2 range (0.1 0.4] or a3 range [0.8, 1)
    when.conds.append(build_cond("a2", CaseWhenRule.Cond.CondOp.GT, 0.1))
    when.connections.append(CaseWhenRule.When.ConnectType.AND)
    when.conds.append(build_cond("a2", CaseWhenRule.Cond.CondOp.LE, 0.4))
    when.connections.append(CaseWhenRule.When.ConnectType.OR)
    when.conds.append(build_cond("a3", CaseWhenRule.Cond.CondOp.GE, 0.8))
    when.connections.append(CaseWhenRule.When.ConnectType.AND)
    when.conds.append(build_cond("a3", CaseWhenRule.Cond.CondOp.LT, 1.0))
    # then 1
    when.then.CopyFrom(build_value(1))
    r.whens.append(when)

    when = CaseWhenRule.When()
    # when a1 == [A, G]
    when.conds.append(build_cond("a1", CaseWhenRule.Cond.CondOp.EQ, "A"))
    when.connections.append(CaseWhenRule.When.ConnectType.OR)
    when.conds.append(build_cond("a1", CaseWhenRule.Cond.CondOp.EQ, "G"))
    # then 2
    when.then.CopyFrom(build_value(2))
    r.whens.append(when)

    # else 0
    r.else_value.CopyFrom(build_value(0))

    r.as_label = False
    r.float_epsilon = 1e-07
    r.output_column = "z"

    names.append("test2")
    tests.append(r)
    expected.append([2, 0, 1, 1, 1, 0, 2, 2, 1, 1])

    # --------TEST---------
    r = CaseWhenRule()

    when = CaseWhenRule.When()
    # a2 range (0.1 0.4] or a6 == [8,9] or a5 == 0.7
    when.conds.append(build_cond("a2", CaseWhenRule.Cond.CondOp.GT, 0.1))
    when.connections.append(CaseWhenRule.When.ConnectType.AND)
    when.conds.append(build_cond("a2", CaseWhenRule.Cond.CondOp.LE, 0.9))
    when.connections.append(CaseWhenRule.When.ConnectType.AND)
    when.conds.append(build_cond("a2", CaseWhenRule.Cond.CondOp.LE, 0.7))
    when.connections.append(CaseWhenRule.When.ConnectType.AND)
    when.conds.append(build_cond("a2", CaseWhenRule.Cond.CondOp.LE, 0.4))
    when.connections.append(CaseWhenRule.When.ConnectType.OR)
    when.conds.append(build_cond("a6", CaseWhenRule.Cond.CondOp.EQ, 8))
    when.connections.append(CaseWhenRule.When.ConnectType.OR)
    when.conds.append(build_cond("a6", CaseWhenRule.Cond.CondOp.EQ, 9))
    when.connections.append(CaseWhenRule.When.ConnectType.OR)
    when.conds.append(build_cond("a5", CaseWhenRule.Cond.CondOp.EQ, 0.7))
    # then 1
    when.then.CopyFrom(build_value(1))
    r.whens.append(when)

    # else 0 from col y
    r.else_value.CopyFrom(build_value("y", True))

    r.as_label = False
    r.float_epsilon = 1e-07
    r.output_column = "z"

    names.append("test3")
    tests.append(r)
    expected.append([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    return names, tests, expected


def test_onehot_encode(comp_prod_sf_cluster_config):
    alice_input_path = "test_onehot_encode/alice.csv"
    bob_input_path = "test_onehot_encode/bob.csv"
    inplace_encode_path = "test_onehot_encode/inplace_sub.csv"
    rule_path = "test_onehot_encode/onehot.rule"
    sub_path = "test_onehot_encode/substitution.csv"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

    if self_party == "alice":
        df_alice = pd.DataFrame(
            {
                "id1": [str(i) for i in range(10)],
                "a1": ["A", "B", "C", "D", "E", "F"] + ["G"] * 4,
                "a2": [i * 0.1 for i in range(10)],
                "a3": [i * 0.1 for i in range(10)],
                "a4": [i * 0.1 for i in range(10)],
                "a5": [i * 0.1 for i in range(10)],
                "a6": [i for i in range(10)],
                "y": [0] * 10,
            }
        )
        df_alice.to_csv(
            comp_storage.get_writer(alice_input_path),
            index=False,
        )
    elif self_party == "bob":
        df_bob = pd.DataFrame(
            {
                "id2": [str(i) for i in range(10)],
                "b4": [i for i in range(10)],
                "b5": [i for i in range(10)],
            }
        )
        df_bob.to_csv(
            comp_storage.get_writer(bob_input_path),
            index=False,
        )

    param = NodeEvalParam(
        domain="preprocessing",
        name="case_when",
        version="0.0.1",
        attr_paths=[
            "rules",
        ],
        attrs=[
            Attribute(s="{}"),
        ],
        inputs=[
            DistData(
                name="input_data",
                type=str(DistDataType.VERTICAL_TABLE),
                data_refs=[
                    DistData.DataRef(uri=bob_input_path, party="bob", format="csv"),
                    DistData.DataRef(uri=alice_input_path, party="alice", format="csv"),
                ],
            )
        ],
        output_uris=[
            inplace_encode_path,
            rule_path,
        ],
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                id_types=["str"],
                ids=["id2"],
                feature_types=["int32", "int32"],
                features=["b4", "b5"],
            ),
            TableSchema(
                id_types=["str"],
                ids=["id1"],
                feature_types=[
                    "str",
                    "float64",
                    "float64",
                    "float64",
                    "float64",
                    "int32",
                ],
                features=["a1", "a2", "a3", "a4", "a5", "a6"],
                label_types=["float32"],
                labels=["y"],
            ),
        ],
    )
    param.inputs[0].meta.Pack(meta)

    for n, t, e in zip(*_build_test()):
        param.attrs[0].s = MessageToJson(t)

        res = case_when.eval(
            param=param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )

        assert len(res.outputs) == 2

        if "alice" == sf_cluster_config.private_config.self_party:
            comp_storage = ComponentStorage(storage_config)
            a_out = pd.read_csv(comp_storage.get_reader(inplace_encode_path))
            z = a_out["z"]
            e = pd.Series(e)

            assert z.equals(
                e
            ), f"{n}\n===z===\n{z}\n===e===\n{e}\n===r===\n{param.attrs[0].s}"

        param2 = NodeEvalParam(
            domain="preprocessing",
            name="substitution",
            version="0.0.2",
            inputs=[param.inputs[0], res.outputs[1]],
            output_uris=[sub_path],
        )

        res = substitution.eval(
            param=param2,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )

        if "alice" == sf_cluster_config.private_config.self_party:
            sub_out = pd.read_csv(comp_storage.get_reader(sub_path))

            assert a_out.equals(sub_out)

        assert len(res.outputs) == 1
