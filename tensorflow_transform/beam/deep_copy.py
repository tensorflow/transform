# Copyright 2018 Google Inc. All Rights Reserved.
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
"""PCollection deep copy utility.

This utility allows a Beam pipeline author to perform a deep copy of a
PCollection up to the nearest materialization boundary. Transforms up to such
a boundary will be copied, and a new deep copied PCollection will be returned.

Materialization boundaries are determined by the following heuristic: we
assume all side inputs are materialized, and do not perform deep copies past
such a boundary, and we also treat any transform listed below in
_MATERIALIZATION_BARRIER_TRANSFORMS as a materialization boundary for this
purpose.
"""

import queue
import re

import apache_beam as beam
from apache_beam import pipeline as beam_pipeline
from apache_beam import pvalue
from apache_beam.pvalue import PCollection
from apache_beam.transforms import resources

_MATERIALIZATION_BARRIER_TRANSFORMS = set(
    [
        beam.GroupByKey,
        # CombinePerKey is included here to allow combiner lifting to occur.
        beam.CombinePerKey,
    ]
)


def _is_at_materialization_boundary(pcollection):
    """Determines whether a PCollection is at a materialization boundary."""
    # Ascend the hierarchy of composite PTransforms. In the Beam pipeline
    # graph, each AppliedPTransform has its "composite parent" stored in its
    # .parent field. Here, we check to see whether, at any of the composite
    # levels, the transform is a materialization boundary.
    current = pcollection.producer
    while current:
        if (
            current.transform
            and current.transform.__class__ in _MATERIALIZATION_BARRIER_TRANSFORMS
        ):
            return True
        if pcollection in current.parent.outputs.values():
            current = current.parent
        else:
            break
    return False


def _get_items_to_clone(pcollection):
    """Get dependency-sorted list of PCollections and PTransforms to clone.

    This method returns a list of items, each of which is either a PCollection or
    PTransform, that need to be cloned when creating a deep copy. This list is
    sorted in dependency order, i.e. each PCollection or PTransform in the list
    occurs before any of its downstream consumers.

    Args:
    ----
      pcollection: PCollection to be deep-copied.

    Returns:
    -------
      A dependency-sorted list of PCollections and PTransforms to clone.

    Raises:
    ------
      ValueError: if the input PCollection is invalid.
    """
    assert isinstance(pcollection, PCollection)
    # List of items (either PCollection or PTransform, in reverse dependency
    # order (i.e. here, consumers occur before producers).
    reversed_to_clone = []
    # Queue of PCollections to be processed in traversal of pipeline graph.
    to_process = queue.Queue()
    # Set of items (PCollections and PTransforms) already seen during pipeline
    # graph traversal.
    seen = set()

    to_process.put(pcollection)
    seen.add(pcollection)
    while not to_process.empty():
        current_pcollection = to_process.get()

        # Stop if we have reached the beginning of the pipeline, or at a
        # materialization boundary.
        if isinstance(
            current_pcollection, pvalue.PBegin
        ) or _is_at_materialization_boundary(current_pcollection):
            continue

        reversed_to_clone.append(current_pcollection)
        applied_transform = current_pcollection.producer
        if applied_transform is None:
            raise ValueError(
                "PCollection node has invalid producer: %s" % current_pcollection
            )

        # Visit the input PCollection(s), and also add other outputs of that applied
        # PTransform.
        if applied_transform in seen:
            continue
        for output in applied_transform.outputs.values():
            assert isinstance(output, PCollection), output
            if output not in seen:
                reversed_to_clone.append(output)
        seen.add(applied_transform)
        reversed_to_clone.append(applied_transform)
        for input_pcollection in applied_transform.inputs:
            if input_pcollection not in seen:
                to_process.put(input_pcollection)
                seen.add(input_pcollection)

    return list(reversed(reversed_to_clone))


def _clone_items(pipeline, to_clone):
    """Clones dependency-sorted list of PCollections and PTransforms.

    Returns mappings of PCollection and PTransform replacements.

    Args:
    ----
      pipeline: The beam.Pipeline.
      to_clone: A dependency-sorted list of PCollections and PTransforms.

    Returns:
    -------
      pcollection_replacements: a dict mapping original to cloned PCollections.

    Raises:
    ------
      ValueError: if a clone is requested of an invalid object.
    """
    pcollection_replacements = {}
    ptransform_replacements = {}
    for item in to_clone:
        if isinstance(item, pvalue.PCollection):
            assert item not in pcollection_replacements
            copied = pvalue.PCollection(
                pipeline,
                tag=item.tag,
                element_type=item.element_type,
                windowing=item.windowing,
            )
            copied.producer = item.producer
            # Update copied PCollection producer if its producer was copied as well.
            if copied.producer in ptransform_replacements:
                original_producer = copied.producer
                copied.producer = ptransform_replacements[original_producer]
                # Update producer outputs,
                for tag, output in original_producer.outputs.items():
                    if output == item:
                        copied.producer.outputs[tag] = copied
            assert copied.producer.transform is not None
            pcollection_replacements[item] = copied
        elif isinstance(item, beam_pipeline.AppliedPTransform):
            assert item.transform is not None
            assert item not in ptransform_replacements
            # The Beam pipeline graph keeps track of composite PTransforms by having
            # AppliedPTransform.parts be a list of "children" AppliedPTransforms that
            # are part of the "parent" AppliedPTransform. Any of these "composite
            # wrapper" AppliedPTransforms does not actually produce output independent
            # of the child non-composite transform. We therefore shouldn't ever clone
            # AppliedPTransforms with non-empty parts, since such AppliedPTransforms
            # are not reachable by tracing outputs in the pipeline graph.
            assert not item.parts, (
                "Reached invalid composite AppliedPTransform: %r." % item
            )

            # TODO(b/217271822): Implement resource hint 'close to resources' for
            # Beam/Dataflow, as when CSE makes it to Dataflow, 'close to resources'
            # cannot be recognized. Once this is fixed, we can change the tag prefix
            # to 'beam'.
            # TODO(b/238243699): Obviate the need for setting 'close to resources'
            # hints.
            close_to_resources_available = resources.ResourceHint.is_registered(
                "close_to_resources"
            )

            if close_to_resources_available:
                # Assign close_to_resources resource hint to the orginal PTransforms.
                # The reason of adding this annotation is to prevent root Reads that are
                # generated from deep copy being merged due to common subexpression
                # elimination (CSE).
                item.resource_hints["beam:resources:close_to_resources:v1"] = (
                    b"/fake/DeepCopy.Original[0]"
                )

            # Assign new label.
            count = 0
            copy_suffix = f"Copy{count}"
            new_label = f"{item.full_label}.{copy_suffix}"
            while new_label in pipeline.applied_labels:
                count += 1
                copy_suffix = f"Copy{count}"
                new_label = f"{item.full_label}.{copy_suffix}"
            pipeline.applied_labels.add(new_label)

            # Update inputs.
            new_inputs = {
                tag: pcollection_replacements.get(old_input, old_input)
                for tag, old_input in item.main_inputs.items()
            }

            # Create the copy. Note that in the copy, copied.outputs will start out
            # empty. Any outputs that are used will be repopulated in the PCollection
            # copy branch above.
            maybe_int = lambda s: int(s) if re.match(r"^\d+$", s) else s
            semver = tuple(maybe_int(s) for s in beam.__version__.split("."))
            if semver >= (2, 63):
                extra_args = {
                    "environment_id": item.environment_id,
                    "annotations": item.annotations,
                }
            else:
                extra_args = {}
            copied = beam_pipeline.AppliedPTransform(
                item.parent, item.transform, new_label, new_inputs, **extra_args
            )

            # Add a 'close to resource' resource hint to the copied PTransforms. The
            # PTransforms that are generated from each deep copy have the same unique
            # 'close to resource' resource hint. This is to make sure that the
            # PTransforms that are cloned from each deep copy can be fused together,
            # but not across copies nor with the original.
            if close_to_resources_available:
                copied.resource_hints["beam:resources:close_to_resources:v1"] = (
                    f"/fake/DeepCopy.{copy_suffix}[0]".encode()
                )

            ptransform_replacements[item] = copied

            # Update composite transform parent to include this copy.
            # TODO(b/111366378): Reconcile the composite PTransform nesting hierarchy,
            # especially in the case where copied PTransforms should be copied in an
            # "all-or-nothing" manner. This would allow the deep copy operation to be
            # safe in the case runners replace well-known composite PTransforms in
            # their entirety during execution.
            copied.parent.parts.append(copied)
        else:
            raise ValueError("Invalid object to clone: %s" % item)

    return pcollection_replacements


# TODO(ccy): When this method is written as a PTransform, the resulting Beam
# pipeline graph incorrectly includes a stray DeepCopy transform, which the
# runner cannot interpret. When this is fixed, we should express this method
# as a DeepCopy PTransform.
def deep_copy(pcollection):
    """Create a deep copy of a PCollection up to materialization boundaries."""
    if not isinstance(pcollection, pvalue.PCollection):
        raise ValueError("Input to deep_copy must be a PCollection.")

    # AppliedPTransform.update_input_refcounts() is a vestigial method that
    # uses an incorrect heuristic; it will be removed in a future version of
    # Beam, since its results aren't used anyway.  Until then, we work around
    # this (see https://issues.apache.org/jira/browse/BEAM-4593).
    if (
        getattr(beam_pipeline.AppliedPTransform, "update_input_refcounts", None)
        is not None
    ):
        beam_pipeline.AppliedPTransform.update_input_refcounts = lambda _: None

    to_clone = _get_items_to_clone(pcollection)
    pcollection_replacements = _clone_items(pcollection.pipeline, to_clone)

    return pcollection_replacements[pcollection]
