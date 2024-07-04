import os
import re
import numpy as np
import pandas as pd
from typing import List
from secretflow.component.component import (
    Component,
    CompEvalError,
    IoType,
    TableColParam,
)
from secretflow.component.data_utils import (
    DistDataType,
    download_files,
    extract_distdata_info,
    merge_individuals_to_vtable,
    upload_files,
)

from secretflow.spec.v1.data_pb2 import DistData, IndividualTable, VerticalTable

federated_computing_comp = Component(
    "federated_computing",
    domain="user",
    version="0.0.1",
    desc="FC between two parties.",
)

federated_computing_comp.str_attr(
    name="expression",
    desc="expression.",
    is_list=False,
    is_optional=False,
)

# 定义输入输出
federated_computing_comp.io(
    io_type=IoType.INPUT,
    name="receiver_input",
    desc="Individual table for receiver",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join.",
            col_min_cnt_inclusive=1,
        )
    ],
)

federated_computing_comp.io(
    io_type=IoType.INPUT,
    name="sender_input",
    desc="Individual table for sender",
    types=[DistDataType.INDIVIDUAL_TABLE],
    col_params=[
        TableColParam(
            name="key",
            desc="Column(s) used to join.",
            col_min_cnt_inclusive=1,
        )
    ],
)

federated_computing_comp.io(
    io_type=IoType.OUTPUT,
    name="fc_output",
    desc="Output vertical table",
    types=[DistDataType.VERTICAL_TABLE],
)


def modify_schema(x: DistData, keys: List[str]) -> DistData:
    new_x = DistData()
    new_x.CopyFrom(x)
    if len(keys) == 0:
        return new_x
    assert x.type == "sf.table.individual"
    imeta = IndividualTable()
    assert x.meta.Unpack(imeta)

    new_meta = IndividualTable()
    names = []
    types = []

    # copy current ids to features and clean current ids.
    for i, t in zip(list(imeta.schema.ids), list(imeta.schema.id_types)):
        names.append(i)
        types.append(t)

    for f, t in zip(list(imeta.schema.features), list(imeta.schema.feature_types)):
        names.append(f)
        types.append(t)

    for k in keys:
        if k not in names:
            raise CompEvalError(f"key {k} is not found as id or feature.")

    for n, t in zip(names, types):
        if n in keys:
            new_meta.schema.ids.append(n)
            new_meta.schema.id_types.append(t)
        else:
            new_meta.schema.features.append(n)
            new_meta.schema.feature_types.append(t)

    new_meta.schema.labels.extend(list(imeta.schema.labels))
    new_meta.schema.label_types.extend(list(imeta.schema.label_types))
    new_meta.line_count = imeta.line_count

    new_x.meta.Pack(new_meta)

    return new_x


@federated_computing_comp.eval_fn
def two_party_balanced_fc_eval_fn(
        *,
        ctx,
        expression,
        receiver_input,
        receiver_input_key,
        sender_input,
        sender_input_key,
        fc_output,
):
    from secretflow.component.component import (
        CompEvalError,
    )
    from secretflow.component.data_utils import (
        extract_distdata_info,
    )

    receiver_path_format = extract_distdata_info(receiver_input)
    assert len(receiver_path_format) == 1
    receiver_party = list(receiver_path_format.keys())[0]
    sender_path_format = extract_distdata_info(sender_input)
    sender_party = list(sender_path_format.keys())[0]

    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    input_path = {
        receiver_party: os.path.join(
            ctx.data_dir, receiver_path_format[receiver_party].uri
        ),
        sender_party: os.path.join(ctx.data_dir, sender_path_format[sender_party].uri),
    }
    temppath = "temp"
    output_path = {
        receiver_party: os.path.join(ctx.data_dir, temppath, fc_output),
        sender_party: os.path.join(ctx.data_dir, temppath, fc_output),
    }
    fc_output_path = os.path.join(ctx.data_dir, fc_output)

    import logging
    import secretflow as sf
    from secretflow.device.device.pyu import PYU
    from secretflow.device.device.spu import SPU
    from secretflow.component.federated_computing_imp import (
        ExpressionToSf, add_calculation, sub_calculation, mul_calculation, div_calculation, equal_calculation,
        greater_calculation, less_calculation, greater_equal_calculation, not_equal_calculation,
        less_equal_calculation, logical_and, logical_or, logical_not, read_data_with_pandas)

    logging.warning(spu_config)
    # init devices.
    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])
    print(f"------------sender_party ------------{sender_party}")
    print(f"------------receiver_party ------------{receiver_party}")
    print(f"------------fc_output ------------{fc_output}")
    alice = PYU(sender_party)
    bob = PYU(receiver_party)

    uri = {
        receiver_party: receiver_path_format[receiver_party].uri,
        sender_party: sender_path_format[sender_party].uri,
    }
    with ctx.tracer.trace_io():
        download_files(ctx, uri, input_path)

    def save_ori_file(df_path, out_path, input_key, res):
        df = pd.read_csv(df_path)
        print(f"------------df columns ------------{list(df.columns)}")
        id_df = df[input_key]
        id_df["result"] = np.array(res).flatten()
        print(f"------------out_path ------------{out_path}")
        id_df.to_csv(out_path, index=False)

    #  数据对齐
    print("------------psi start ------------}")
    report = spu.psi(
        keys={receiver_party: receiver_input_key, sender_party: sender_input_key},
        input_path=input_path,
        output_path=output_path,
        receiver=sender_party,
        broadcast_result=True,
        protocol="PROTOCOL_KKRT",
        ecdh_curve="CURVE_FOURQ",
        skip_duplicates_check=False,
        disable_alignment=False,
        check_hash_digest=False,
    )
    print("------------psi done ------------}")
    from secretflow.data.vertical import read_csv

    print("------------read csv  v_df ------------}")
    v_df = read_csv({alice: output_path[sender_party], bob: output_path[receiver_party]})
    print(f"------------v_df------------: {v_df}")

    # 处理表达式，并获取所有参与计算的字段
    expression_tsf = ExpressionToSf()
    expression_new = expression_tsf.formula_to_function(expression, v_df,
                                                        spu)  # result = spu(add_calculation)(data_1, data_2)
    expression_columns = expression_tsf.columns
    print(f"------------expression_columns------------: {expression_columns}")
    print(f"------------expression_new------------: {expression_new}")

    res = sf.reveal(expression_new)
    print(f"------------res------------: {res}")

    print("------------start save ------------")
    sender_out_path = output_path[sender_party]
    receiver_out_path = output_path[receiver_party]
    print(f"------------out_path ------------{sender_out_path}")
    print(f"------------sender_input_key ------------{list(sender_input_key)}")
    print(f"------------receiver_input_key ------------{list(receiver_input_key)}")
    sf.wait(alice(save_ori_file)(sender_out_path, fc_output_path, sender_input_key, res))
    sf.wait(bob(save_ori_file)(receiver_out_path, fc_output_path, receiver_input_key, res))
    print("保存计算结果")

    output_db = DistData(
        name=fc_output,
        type=str(DistDataType.VERTICAL_TABLE),
        system_info=receiver_input.system_info,
        data_refs=[
            DistData.DataRef(
                uri=fc_output,
                party=receiver_party,
                format="csv",
            ),
            DistData.DataRef(
                uri=fc_output,
                party=sender_party,
                format="csv",
            ),
        ],
    )

    output_db = merge_individuals_to_vtable(
        [
            modify_schema(receiver_input, receiver_input_key),
            modify_schema(sender_input, sender_input_key),
        ],
        output_db,
    )
    vmeta = VerticalTable()
    assert output_db.meta.Unpack(vmeta)
    vmeta.line_count = report[0]['intersection_count']
    output_db.meta.Pack(vmeta)

    return {"fc_output": output_db}
