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
import pytest

from secretflow.component.ml.boost.ss_xgb.ss_xgb import (
    ss_xgb_predict_comp,
    ss_xgb_train_comp,
)
from secretflow.component.storage import ComponentStorage
from secretflow.spec.v1.component_pb2 import Attribute
from secretflow.spec.v1.data_pb2 import DistData, TableSchema, VerticalTable
from secretflow.spec.v1.evaluation_pb2 import NodeEvalParam
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

NUM_BOOST_ROUND = 3


@pytest.mark.parametrize("with_checkpoint", [True, False])
def test_ss_xgb(comp_prod_sf_cluster_config, with_checkpoint):
    work_path = f"ut_test_ss_xgb_{with_checkpoint}"
    alice_path = f"{work_path}/x_alice.csv"
    bob_path = f"{work_path}/x_bob.csv"
    model_path = f"{work_path}/model.sf"
    predict_path = f"{work_path}/predict.csv"
    checkpoint_path = f"{work_path}/checkpoint"

    storage_config, sf_cluster_config = comp_prod_sf_cluster_config
    self_party = sf_cluster_config.private_config.self_party
    comp_storage = ComponentStorage(storage_config)

    scaler = StandardScaler()
    ds = load_breast_cancer()
    x, y = scaler.fit_transform(ds["data"]), ds["target"]
    if self_party == "alice":
        x = pd.DataFrame(x[:, :15], columns=[f"a{i}" for i in range(15)])
        y = pd.DataFrame(y, columns=["y"])
        ds = pd.concat([x, y], axis=1)
        ds.to_csv(comp_storage.get_writer(alice_path), index=False)

    elif self_party == "bob":
        ds = pd.DataFrame(x[:, 15:], columns=[f"b{i}" for i in range(15)])
        ds.to_csv(comp_storage.get_writer(bob_path), index=False)

    train_param = NodeEvalParam(
        domain="ml.train",
        name="ss_xgb_train",
        version="0.0.1",
        attr_paths=[
            "num_boost_round",
            "max_depth",
            "learning_rate",
            "objective",
            "reg_lambda",
            "subsample",
            "colsample_by_tree",
            "sketch_eps",
            "base_score",
            "input/train_dataset/label",
            "input/train_dataset/feature_selects",
        ],
        attrs=[
            Attribute(i64=NUM_BOOST_ROUND),
            Attribute(i64=3),
            Attribute(f=0.3),
            Attribute(s="logistic"),
            Attribute(f=0.1),
            Attribute(f=1),
            Attribute(f=1),
            Attribute(f=0.25),
            Attribute(f=0),
            Attribute(ss=["y"]),
            Attribute(ss=[f"a{i}" for i in range(15)] + [f"b{i}" for i in range(15)]),
        ],
        inputs=[
            DistData(
                name="train_dataset",
                type="sf.table.vertical_table",
                data_refs=[
                    DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                    DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                ],
            ),
        ],
        output_uris=[model_path],
        checkpoint_uri=checkpoint_path if with_checkpoint else "",
    )

    meta = VerticalTable(
        schemas=[
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"a{i}" for i in range(15)],
                label_types=["float32"],
                labels=["y"],
            ),
            TableSchema(
                feature_types=["float32"] * 15,
                features=[f"b{i}" for i in range(15)],
            ),
        ],
    )
    train_param.inputs[0].meta.Pack(meta)

    train_res = ss_xgb_train_comp.eval(
        param=train_param,
        storage_config=storage_config,
        cluster_config=sf_cluster_config,
    )

    def run_pred(predict_path, train_res):
        predict_param = NodeEvalParam(
            domain="ml.predict",
            name="ss_xgb_predict",
            version="0.0.2",
            attr_paths=[
                "receiver",
                "save_ids",
                "save_label",
                "input/feature_dataset/saved_features",
            ],
            attrs=[
                Attribute(ss=["alice"]),
                Attribute(b=False),
                Attribute(b=True),
                Attribute(ss=["a2", "a10"]),
            ],
            inputs=[
                train_res.outputs[0],
                DistData(
                    name="train_dataset",
                    type="sf.table.vertical_table",
                    data_refs=[
                        DistData.DataRef(uri=alice_path, party="alice", format="csv"),
                        DistData.DataRef(uri=bob_path, party="bob", format="csv"),
                    ],
                ),
            ],
            output_uris=[predict_path],
        )
        meta = VerticalTable(
            schemas=[
                TableSchema(
                    feature_types=["float32"] * 15,
                    features=[f"a{i}" for i in range(15)],
                    label_types=["float32"],
                    labels=["y"],
                ),
                TableSchema(
                    feature_types=["float32"] * 15,
                    features=[f"b{i}" for i in range(15)],
                ),
            ],
        )
        predict_param.inputs[1].meta.Pack(meta)

        predict_res = ss_xgb_predict_comp.eval(
            param=predict_param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )

        assert len(predict_res.outputs) == 1

        if "alice" == sf_cluster_config.private_config.self_party:
            comp_storage = ComponentStorage(storage_config)
            input_y = pd.read_csv(comp_storage.get_reader(alice_path))
            output_y = pd.read_csv(comp_storage.get_reader(predict_path))

            # label & pred
            assert output_y.shape[1] == 4

            assert input_y.shape[0] == output_y.shape[0]

            auc = roc_auc_score(input_y["y"], output_y["pred"])
            assert auc > 0.99, f"auc {auc}"

    run_pred(predict_path, train_res)
    if with_checkpoint:
        cp_num = NUM_BOOST_ROUND
        if "alice" == sf_cluster_config.private_config.self_party:
            comp_storage = ComponentStorage(storage_config)
            for i in range(int(cp_num / 2), cp_num):
                with comp_storage.get_writer(f"{checkpoint_path}_{i}") as f:
                    # destroy some checkpoint to rollback train progress
                    f.write(b"....")

        # run train again from checkpoint
        train_param.output_uris[0] = f"{work_path}/model.sf.2"
        train_res = ss_xgb_train_comp.eval(
            param=train_param,
            storage_config=storage_config,
            cluster_config=sf_cluster_config,
        )

        run_pred(f"{work_path}/predict.csv.2", train_res)
