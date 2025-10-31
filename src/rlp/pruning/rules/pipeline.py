"""Apply structured pruning decisions to a model."""

from __future__ import annotations

from typing import Any, Dict, List, Set

import torch.nn as nn

from .rankers import l2_scores_conv2d, l2_scores_linear, select_keep_idx
from .slicers import (
    slice_linear_out,
    slice_conv_transpose_out,
    slice_depthwise_conv,
    slice_conv_out,
    slice_consumer_in,
    slice_linear_in,
    slice_batchnorm,
)
from .adjustments import adjust_bn_chain, adjust_concat_pre_norms
from .concat import concat_sources, direct_prunable_consumers, compute_concat_indices
from .helpers import (
    infer_conv_type_from_module,
    expand_linear_indices,
    sanitize_idx,
    nearest_prunable_producer,
)


def apply_pruning(
    model: nn.Module, dg, pg, decisions: Dict[int, int]
) -> Dict[int, List[int]]:
    name2mod = dict(model.named_modules())
    id2node = {n["id"]: n for n in dg.nodes}
    keep_map: Dict[int, List[int]] = {}

    for group in pg.groups:
        gid = group["group_id"]
        k = int(decisions.get(gid, group["max_channels"]))
        if group["divisibility"] > 1:
            k = (k // group["divisibility"]) * group["divisibility"]
        k = max(group["min_channels"], min(k, group["max_channels"]))
        if group["divisibility"] > 1 and (k % group["divisibility"] != 0):
            k = ((k // group["divisibility"]) + 1) * group["divisibility"]
            k = min(k, group["max_channels"])

        producer = group["producer"]
        producer_module = name2mod.get(producer)
        if producer_module is None:
            continue

        if isinstance(producer_module, nn.Linear):
            scores = l2_scores_linear(producer_module)
        else:
            scores = l2_scores_conv2d(producer_module, group.get("producer_conv_type"))
        keep_idx = select_keep_idx(scores, k)
        keep_map[gid] = keep_idx

        if isinstance(producer_module, nn.Linear):
            slice_linear_out(producer_module, keep_idx)
        elif isinstance(producer_module, nn.ConvTranspose2d):
            slice_conv_transpose_out(producer_module, keep_idx)
        elif (
            isinstance(producer_module, nn.Conv2d)
            and group.get("producer_conv_type") == "depthwise"
        ):
            slice_depthwise_conv(producer_module, keep_idx)
        elif isinstance(producer_module, nn.Conv2d):
            slice_conv_out(producer_module, keep_idx)

        adjust_bn_chain(dg, name2mod, id2node, producer, keep_idx)

        for member in group["members"]:
            if member == producer:
                continue
            member_module = name2mod.get(member)
            if member_module is None:
                continue
            m_conv_type = (
                infer_conv_type_from_module(member_module)
                if isinstance(member_module, nn.Conv2d)
                else None
            )
            if isinstance(member_module, nn.Linear):
                slice_linear_out(member_module, keep_idx)
            elif isinstance(member_module, nn.ConvTranspose2d):
                slice_conv_transpose_out(member_module, keep_idx)
            elif isinstance(member_module, nn.Conv2d) and (m_conv_type == "depthwise"):
                slice_depthwise_conv(member_module, keep_idx)
            elif isinstance(member_module, nn.Conv2d):
                slice_conv_out(member_module, keep_idx)
            adjust_bn_chain(dg, name2mod, id2node, member, keep_idx)

    processed_inputs: Set[str] = set()
    for group in pg.groups:
        gid = group["group_id"]
        keep_idx = keep_map[gid]
        for consumer in group["consumers"]:
            cname = consumer["name"]
            if cname in processed_inputs:
                continue
            consumer_module = name2mod.get(cname)
            if consumer_module is None:
                continue

            consumer_gid = pg.layer_to_group.get(cname, -1)
            if consumer.get("needs_concat"):
                full_idx, _ = compute_concat_indices(
                    dg, pg, cname, keep_map, name2mod, id2node
                )
                slice_consumer_in(
                    consumer_module,
                    full_idx,
                    consumer.get("conv_type"),
                    group["producer_type"],
                    group["base_channels"],
                )
                adjust_concat_pre_norms(dg, id2node, name2mod, cname, full_idx)
            else:
                concat_candidates: List[Any] = []
                full_idx = None
                treat_concat = False
                try:
                    concat_candidates = concat_sources(dg, cname, id2node, pg)
                    if len(concat_candidates) > 1:
                        full_idx, _ = compute_concat_indices(
                            dg, pg, cname, keep_map, name2mod, id2node
                        )
                        if full_idx:
                            treat_concat = True
                except Exception:
                    concat_candidates = []

                if treat_concat and full_idx is not None:
                    slice_consumer_in(
                        consumer_module,
                        full_idx,
                        consumer.get("conv_type"),
                        group["producer_type"],
                        group["base_channels"],
                    )
                    adjust_concat_pre_norms(dg, id2node, name2mod, cname, full_idx)
                else:
                    if (
                        isinstance(consumer_module, nn.Linear)
                        and group["producer_type"] != "Linear"
                    ):
                        processed_inputs.add(cname)
                        continue
                    target_idx = keep_idx
                    if (
                        consumer_gid != -1
                        and consumer_gid != gid
                        and consumer.get("conv_type") == "depthwise"
                    ):
                        target_idx = keep_map.get(consumer_gid, keep_idx)
                    if isinstance(consumer_module, nn.Linear) and concat_candidates:
                        expanded = expand_linear_indices(
                            consumer_module,
                            target_idx,
                            [entry[1] for entry in concat_candidates],
                            name2mod,
                            pg,
                        )
                        if expanded:
                            target_idx = expanded

                    slice_consumer_in(
                        consumer_module,
                        target_idx,
                        consumer.get("conv_type"),
                        group["producer_type"],
                        group["base_channels"],
                    )

            processed_inputs.add(cname)

        extras = direct_prunable_consumers(dg, group["producer"], id2node, name2mod)
        for extra in extras:
            cname = extra["name"]
            if cname in processed_inputs:
                continue
            consumer_module = name2mod.get(cname)
            if consumer_module is None:
                continue
            consumer_gid = pg.layer_to_group.get(cname, -1)
            target_idx = keep_idx
            if (
                consumer_gid != -1
                and consumer_gid != gid
                and extra.get("conv_type") == "depthwise"
            ):
                target_idx = keep_map.get(consumer_gid, keep_idx)
            slice_consumer_in(
                consumer_module,
                target_idx,
                extra.get("conv_type"),
                group["producer_type"],
                group["base_channels"],
            )
            processed_inputs.add(cname)

    for lname, lmod in name2mod.items():
        if not isinstance(lmod, nn.Linear) or lname in processed_inputs:
            continue
        try:
            sources = concat_sources(dg, lname, id2node, pg)
            if sources:
                full_idx, _ = compute_concat_indices(
                    dg, pg, lname, keep_map, name2mod, id2node
                )
                src_names = [entry[1] for entry in sources]
                full_idx = expand_linear_indices(
                    lmod, full_idx, src_names, name2mod, pg
                )
                current_in = getattr(lmod, "in_features", 0)
                if full_idx and len(full_idx) < current_in:
                    slice_linear_in(lmod, full_idx)
                    processed_inputs.add(lname)
                    continue
                sanitized = sanitize_idx(full_idx, current_in)
                if sanitized and len(sanitized) < current_in:
                    slice_linear_in(lmod, sanitized)
                    processed_inputs.add(lname)
                    continue
        except Exception:
            pass
        producer = nearest_prunable_producer(pg, id2node, lname)
        if producer is None:
            continue
        producer_gid = pg.layer_to_group.get(producer, -1)
        if producer_gid == -1:
            continue
        base_idx = keep_map[producer_gid]
        base_idx = expand_linear_indices(lmod, base_idx, [producer], name2mod, pg)
        slice_linear_in(lmod, base_idx)
        processed_inputs.add(lname)

    for bname, bmod in name2mod.items():
        if not isinstance(bmod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            continue
        try:
            sources = concat_sources(dg, bname, id2node, pg)
        except Exception:
            continue
        if len(sources) <= 1:
            continue
        full_idx, _ = compute_concat_indices(dg, pg, bname, keep_map, name2mod, id2node)
        if not full_idx:
            continue
        slice_batchnorm(bmod, full_idx)

    for lname, lmod in name2mod.items():
        if not isinstance(lmod, nn.Linear):
            continue
        try:
            sources = concat_sources(dg, lname, id2node, pg)
        except Exception:
            continue
        if len(sources) <= 1:
            continue
        try:
            full_idx, _ = compute_concat_indices(
                dg, pg, lname, keep_map, name2mod, id2node
            )
        except Exception:
            continue
        current_in = getattr(lmod, "in_features", 0)
        if full_idx and len(full_idx) < current_in:
            slice_linear_in(lmod, full_idx)

    conv_like = {
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "ConvTranspose1d",
        "ConvTranspose2d",
        "ConvTranspose3d",
    }

    for group in pg.groups:
        gid = group["group_id"]
        base_keep = keep_map[gid]
        for consumer in group["consumers"]:
            cname = consumer["name"]
            module = name2mod.get(cname)
            if module is None:
                continue

            conv_type = consumer.get("conv_type")
            producer_type = group["producer_type"]
            producer_base = group["base_channels"]
            consumer_gid = pg.layer_to_group.get(cname, -1)

            target_idx: List[int] = list(base_keep)
            needs_concat = consumer.get("needs_concat")
            concat_candidates: List[Any] = []
            treat_concat = False
            full_idx: Optional[List[int]] = None

            if needs_concat:
                try:
                    full_idx, _ = compute_concat_indices(
                        dg, pg, cname, keep_map, name2mod, id2node
                    )
                except Exception:
                    full_idx = None
            else:
                try:
                    concat_candidates = concat_sources(dg, cname, id2node, pg)
                except Exception:
                    concat_candidates = []
                if len(concat_candidates) > 1:
                    try:
                        full_idx, _ = compute_concat_indices(
                            dg, pg, cname, keep_map, name2mod, id2node
                        )
                        if full_idx:
                            treat_concat = True
                    except Exception:
                        full_idx = None

            if needs_concat and full_idx:
                target_idx = list(full_idx)
            elif treat_concat and full_idx:
                target_idx = list(full_idx)
            else:
                target_idx = list(base_keep)
                if (
                    consumer_gid != -1
                    and consumer_gid != gid
                    and conv_type == "depthwise"
                ):
                    target_idx = list(keep_map.get(consumer_gid, base_keep))
                if isinstance(module, nn.Linear) and concat_candidates:
                    expanded = expand_linear_indices(
                        module,
                        target_idx,
                        [entry[1] for entry in concat_candidates],
                        name2mod,
                        pg,
                    )
                    if expanded:
                        target_idx = list(expanded)

            target_len = len(target_idx)
            if isinstance(module, nn.Linear):
                expected = target_len
                if producer_type in conv_like and producer_base:
                    stride = module.in_features // max(1, producer_base)
                    if stride > 0:
                        expected = stride * target_len
                if expected and module.in_features != expected:
                    slice_consumer_in(
                        module, target_idx, conv_type, producer_type, producer_base
                    )
            elif isinstance(module, nn.ConvTranspose2d):
                expected = target_len
                if module.out_channels != expected:
                    slice_consumer_in(
                        module, target_idx, conv_type, producer_type, producer_base
                    )
            elif isinstance(module, nn.Conv2d):
                expected = None
                if conv_type == "depthwise" or (
                    module.groups == module.in_channels == module.out_channels
                    and module.groups > 1
                ):
                    expected = target_len
                elif module.groups == 1:
                    expected = target_len
                if expected is not None and module.in_channels != expected:
                    slice_consumer_in(
                        module, target_idx, conv_type, producer_type, producer_base
                    )

    return keep_map


__all__ = ["apply_pruning"]
