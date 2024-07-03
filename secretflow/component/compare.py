from secretflow.component.component import (
    Component,
    IoType,
    TableColParam,
)

from secretflow.component.data_utils import DistDataType

ss_compare_comp = Component(
    "ss_compare",
    domain="user",
    version="0.0.1",
    desc="""compare two tables.""",
)
ss_compare_comp.int_attr(
    name="tolerance",
    desc="two numbers to be equal if they are within tolerance.",
    is_list=False,
    is_optional=True,
    default_value=10,
    allowed_values=None,
    lower_bound=0,
    lower_bound_inclusive=True,
    upper_bound=None,
)
ss_compare_comp.io(
    io_type=IoType.INPUT,
    name="input_table",
    desc="Input vertical table",
    types=[DistDataType.VERTICAL_TABLE],
    col_params=[
        TableColParam(
            name="alice_value",
            desc="Column(s) used to compare.",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=1,
        ),
        TableColParam(
            name="bob_value",
            desc="Column(s) used to compare.",
            col_min_cnt_inclusive=1,
            col_max_cnt_inclusive=1,
        ),
    ],
)

ss_compare_comp.io(
    io_type=IoType.OUTPUT,
    name="alice_output",
    desc="Output for alice",
    types=[DistDataType.INDIVIDUAL_TABLE],
)

ss_compare_comp.io(
    io_type=IoType.OUTPUT,
    name="bob_output",
    desc="Output for bob",
    types=[DistDataType.INDIVIDUAL_TABLE],
)


@ss_compare_comp.eval_fn
def ss_compare_eval_fn(
        *,
        ctx,
        tolerance,
        input_table,
        input_table_alice_value,
        input_table_bob_value,
        alice_output,
        bob_output,
):
    import os

    from secretflow.component.component import CompEvalError
    from secretflow.component.data_utils import (
        DistDataType,
        load_table,
    )
    from secretflow.data import FedNdarray, PartitionWay
    from secretflow.device.device.pyu import PYU
    from secretflow.device.device.spu import SPU
    from secretflow.device.driver import wait
    from secretflow.protos.component.data_pb2 import (
        DistData,
        IndividualTable,
        TableSchema,
        VerticalTable,
    )

    # only local fs is supported at this moment.
    data_dir = ctx.data_dir

    # get spu config from ctx
    if ctx.spu_configs is None or len(ctx.spu_configs) == 0:
        raise CompEvalError("spu config is not found.")
    if len(ctx.spu_configs) > 1:
        raise CompEvalError("only support one spu")
    spu_config = next(iter(ctx.spu_configs.values()))

    # load inputs
    meta = VerticalTable()
    input_table.meta.Unpack(meta)

    # get alice and bob party
    for data_ref, schema in zip(list(input_table.data_refs), list(meta.schemas)):
        if input_table_alice_value[0] in list(schema.features):
            alice_party = data_ref.party
            alice_ids = list(schema.ids)
            alice_id_types = list(schema.id_types)
        elif input_table_bob_value[0] in list(schema.features):
            bob_party = data_ref.party
            bob_ids = list(schema.ids)
            bob_id_types = list(schema.id_types)

    # init devices.
    alice = PYU(alice_party)
    bob = PYU(bob_party)
    spu = SPU(spu_config["cluster_def"], spu_config["link_desc"])

    input_df = load_table(
        ctx,
        input_table,
        load_features=True,
        load_ids=True,
        load_labels=True,
        col_selects=input_table_alice_value + input_table_bob_value,
    )

    # pass inputs from alice and bob PYUs to SPU
    alice_input_spu_object = input_df.partitions[alice].data.to(spu)
    bob_input_spu_object = input_df.partitions[bob].data.to(spu)

    from secretflow.device import SPUCompilerNumReturnsPolicy

    def compare_fn(x, y, tolerance):
        return (x - tolerance) > y, (y - tolerance) > x

    # do comparison
    output_alice_spu_obj, output_bob_spu_obj = spu(
        compare_fn,
        num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
        user_specified_num_returns=2,
    )(alice_input_spu_object, bob_input_spu_object, tolerance)

    # convert to FedNdarray
    res = FedNdarray(
        partitions={
            alice: output_alice_spu_obj.to(alice),
            bob: output_bob_spu_obj.to(bob),
        },
        partition_way=PartitionWay.VERTICAL,
    )

    def save(id, id_key, res, res_key, path):
        import pandas as pd

        x = pd.DataFrame(id, columns=id_key)
        label = pd.DataFrame(res, columns=res_key)
        x = pd.concat([x, label], axis=1)

        x.to_csv(path, index=False)

    alice_id_df = load_table(
        ctx,
        input_table,
        load_features=False,
        load_ids=True,
        load_labels=False,
        col_selects=alice_ids,
    )

    wait(
        alice(save)(
            alice_id_df.partitions[alice].data,
            alice_ids,
            res.partitions[alice].data,
            ['result'],
            os.path.join(local_fs_wd, alice_output),
        )
    )

    bob_id_df = load_table(
        ctx,
        input_table,
        load_features=False,
        load_ids=True,
        load_labels=False,
        col_selects=bob_ids,
    )

    wait(
        bob(save)(
            bob_id_df.partitions[bob].data,
            bob_ids,
            res.partitions[bob].data,
            ['result'],
            os.path.join(local_fs_wd, bob_output),
        )
    )

    # generate DistData
    alice_db = DistData(
        name='result',
        type=str(DistDataType.INDIVIDUAL_TABLE),
        data_refs=[DistData.DataRef(uri=alice_output, party=alice.party, format="csv")],
    )

    alice_meta = IndividualTable(
        schema=TableSchema(
            ids=alice_ids,
            id_types=alice_id_types,
            features=['result'],
            feature_types=['bool'],
        ),
        num_lines=-1,
    )

    alice_db.meta.Pack(alice_meta)

    bob_db = DistData(
        name='result',
        type=str(DistDataType.INDIVIDUAL_TABLE),
        data_refs=[DistData.DataRef(uri=bob_output, party=bob.party, format="csv")],
    )

    bob_meta = IndividualTable(
        schema=TableSchema(
            ids=bob_ids,
            id_types=bob_id_types,
            features=['result'],
            feature_types=['bool'],
        ),
        num_lines=-1,
    )

    bob_db.meta.Pack(bob_meta)

    return {"alice_output": alice_db, "bob_output": bob_db}
