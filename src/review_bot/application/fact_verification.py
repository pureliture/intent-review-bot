from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Callable

from review_bot.adapters.byte_budget import effective_output_reservation
from review_bot.application import review_fact_check as fact_check
from review_bot.application.context import hash_json
from review_bot.application.source_partition import PartitionError


def execute_fact_verification(final, originals, meta, *, mode, review_client, profile, budget,
    request: Callable, measure: Callable, execute: Callable, choose: Callable,
    max_rewrites: int, stop_error: type[Exception], first_record: int = 0, language="en",
    final_citation_source_ids=None):
    verification = {}

    def fact_call(stage, rows, candidate, record, extra=None):
        call = {"measurements": [], "extra": deepcopy(extra), "rows": deepcopy(rows)}
        record["check" if stage == "fact_check" else "rewrite"] = call
        context = verification["original_final"]

        def execute_split(packet, packet_rows, ranges, active_stage, parts):
            packet["children"] = []
            child_results = []
            for child_rows, child_ranges in parts:
                child = {}
                packet["children"].append(child)
                child_stage = active_stage
                if active_stage == "fact_rewrite" and packet["split"] == "sources" and child_results:
                    first_raw, first_refs, _, _ = child_results[0]
                    replacement = fact_check.accept_rewrite(first_raw, packet_rows[0], first_refs, mode, meta)
                    child_rows = [{**packet_rows[0], "item": replacement}]
                    child_stage = "fact_check"
                child_results.append(execute_packet(child, child_rows, child_ranges, child_stage))
            combined_refs = fact_check.merge_reference_contexts(
                child_refs for _, child_refs, _, _ in child_results)
            combined_spans = list({(span.evidence_id, span.start_byte, span.end_byte): span
                for _, _, child_spans, _ in child_results for span in child_spans}.values())
            if active_stage == "fact_check":
                raw = fact_check.assemble_checks(packet_rows, packet["children"])
            elif packet["split"] == "items":
                raw = fact_check.assemble_rewrites(packet_rows, packet["children"])
            else:
                raw = deepcopy(packet["children"][0]["response"])
                packet["candidate_page_checks"] = [deepcopy(child["response"])
                    for child in packet["children"][1:]]
            packet["cited_evidence"] = [evidence for child in packet["children"] for evidence in child["cited_evidence"]]
            final_index = child_results[-1][3]
            if active_stage == "fact_check" and packet["split"] == "sources" and ranges is None:
                packet["discovery_response"] = deepcopy(raw)
                focused_ranges, history = fact_check.whole_claim_packet(packet, packet_rows, originals, packet["extra"])
                packet["discovery"] = deepcopy(history)
                if focused_ranges:
                    packet["whole_claim"] = {}
                    focused, focused_refs, focused_spans, focused_index = execute_packet(
                        packet["whole_claim"], packet_rows, focused_ranges, "fact_check", history)
                    if focused is not None:
                        raw = focused
                        combined_refs = fact_check.merge_reference_contexts((combined_refs, focused_refs))
                        combined_spans = list({(span.evidence_id, span.start_byte, span.end_byte): span
                            for span in [*combined_spans, *focused_spans]}.values())
                        packet["cited_evidence"].extend(packet["whole_claim"]["cited_evidence"])
                        final_index = focused_index
            packet.update(response=deepcopy(raw), response_hash=hash_json(raw))
            packet["delivered_ranges"] = [{"evidence_id": span.evidence_id,
                "start_byte": span.start_byte, "end_byte": span.end_byte} for span in combined_spans]
            return raw, combined_refs, combined_spans, final_index

        def execute_packet(packet, packet_rows, ranges=None, packet_stage=None, whole_claim=None):
            active_stage = packet_stage or stage
            packet_extra = fact_check.project_rewrite_context(extra, packet_rows)
            packet.update(rows=deepcopy(packet_rows), extra=deepcopy(packet_extra), measurements=[], call_stage=active_stage)
            if ranges is not None:
                packet["source_ranges"] = deepcopy(ranges)
            if whole_claim is not None:
                packet["whole_claim_discovery"] = deepcopy(whole_claim)
            for kind in (("cited_ranges",) if ranges is not None else
                    ("complete_originals", "cited_originals", "cited_ranges")):
                payload, schema, refs, spans = fact_check.request_spec(active_stage, packet_rows, context,
                    originals, meta, kind=kind, extra=packet_extra, ranges=ranges, whole_claim=whole_claim,
                    final_citation_source_ids=final_citation_source_ids)
                req = request(active_stage, payload, schema, profile)
                packet["measurements"].append({"stage_index": len(budget.records) - first_record, "delivery": kind})
                if measure(req, review_client, profile, active_stage + ":" + kind + ":admission", for_generation=True):
                    break
            else:
                if whole_claim is not None:
                    packet.update(admission_rejected=True, delivery=kind,
                        payload_hash=hash_json(payload), schema_hash=hash_json(schema),
                        delivered_ranges=[{"evidence_id": span.evidence_id, "start_byte": span.start_byte,
                            "end_byte": span.end_byte} for span in spans])
                    return None, refs, spans, None
                if len(packet_rows) > 1:
                    middle = len(packet_rows) // 2
                    parts = [(packet_rows[:middle], None), (packet_rows[middle:], None)]
                    packet["split"] = "items"
                else:
                    if ranges is None:
                        _, _, _, spans = fact_check.request_spec(active_stage, packet_rows, context,
                            originals, meta, kind="complete_originals", extra=packet_extra,
                            final_citation_source_ids=final_citation_source_ids)
                    cover = ranges if ranges is not None else [{"evidence_id": span.evidence_id,
                        "start_byte": span.start_byte, "end_byte": span.end_byte} for span in spans]
                    if not cover:
                        raise stop_error("fact_fixed_request_exceeds_context")
                    left, right = fact_check.split_ranges(cover, originals)
                    parts = [(packet_rows, left), (packet_rows, right)]
                    packet["split"] = "sources"
                packet["split_cause"] = "input_limit"
                return execute_split(packet, packet_rows, ranges, active_stage, parts)
            packet.update(delivery=kind, payload_hash=hash_json(payload), schema_hash=hash_json(schema))
            packet["delivered_ranges"] = [{"evidence_id": span.evidence_id,
                "start_byte": span.start_byte, "end_byte": span.end_byte} for span in spans]
            try:
                raw, index = execute(req, review_client, profile, active_stage)
            except stop_error as error:
                failed = len(budget.records) - 1
                event = budget.records[failed]
                rejected = budget.rejected_outputs.get(failed, [])
                known = (str(error) == "model_output_limit" and event.reason == "model_output_limit" and
                    bool(event.attempts) and bool(rejected) and not budget.operational_totals()[2] and
                    all(attempt.delivery == "sent" and attempt.usage is not None and
                        attempt.usage["output_tokens"] == effective_output_reservation(
                            req.max_output_tokens, profile.provider) for attempt in event.attempts))
                if len(packet_rows) < 2 or not known:
                    raise
                packet["split"] = "items"
                packet["split_cause"] = "known_output_limit"
                packet["output_failure"] = {"stage_index": failed - first_record,
                    "rejected_output_hashes": [item["output_hash"] for item in rejected],
                    "attempt_request_hashes": [attempt.request_hash for attempt in event.attempts]}
                middle = len(packet_rows) // 2
                parts = [(packet_rows[:middle], ranges), (packet_rows[middle:], ranges)]
                return execute_split(packet, packet_rows, ranges, active_stage, parts)
            unique, citation_changes = fact_check.deduplicate_exact_citation_arrays(raw)
            if citation_changes:
                packet.update(raw_response=deepcopy(raw), raw_response_hash=hash_json(raw),
                    citation_deduplication=citation_changes)
                raw = unique
            packet.update(response=deepcopy(raw), response_hash=hash_json(raw), stage_index=index - first_record)
            try:
                packet["cited_evidence"] = fact_check.response_citations(raw, refs)
                if active_stage == "fact_check":
                    fact_check.accept_check(raw, packet_rows, refs,
                        [o["evidence_id"] for o in fact_check.unread_inventory(originals, spans)])
                else:
                    fact_check.accept_rewrites(raw, packet_rows, refs, mode, meta)
            except PartitionError as error:
                packet["validation_error"] = str(error)
                budget.records[index] = replace(budget.records[index], status="failed", reason=str(error))
                raise
            budget.records[index] = replace(budget.records[index], status="completed", reason="validated")
            return raw, refs, spans, index

        return execute_packet(call, rows)

    def factual_check(rows, candidate, record):
        raw, refs, spans, index = fact_call("fact_check", rows, candidate, record, fact_check.check_extra(record))
        return deepcopy(raw["items"])

    def factual_sources(row, candidate, record):
        raw = record["check"]["response"]
        parent = fact_check.source_parent(meta["input_hash"], verification["original_final_hash"], row, record)
        plan = {"version": 1, "parent_plan_id": parent,
            "reason": "required_context_from_existing_frozen_originals",
            "prior_response_hash": hash_json(raw), "gaps": fact_check.source_gaps(record),
            "complete": True, "requests": fact_check.planned_source_requests(record)}
        plan["plan_id"] = hash_json(plan)
        record["source_plan"] = plan

    def factual_rewrite(rows, candidate, record):
        extra = fact_check.rewrite_batch_extra(rows, record["batch_records"])
        raw, refs, spans, index = fact_call("fact_rewrite", rows, candidate, record, extra)
        replacements = fact_check.accept_rewrites(raw, rows, refs, mode, meta)
        record["rewrite"]["accepted"] = deepcopy(replacements if len(rows) > 1 else replacements[rows[0]["id"]])
        fact_check.retain_sources(candidate, spans)
        budget.records[index] = replace(budget.records[index], status="completed", reason="validated")
        return replacements

    original_final = deepcopy(final)
    try:
        final = fact_check.run(final, originals, meta, mode=mode, check=factual_check,
            rewrite=factual_rewrite, choose=choose, resolve_sources=factual_sources,
            max_rewrites=max_rewrites, ledger=verification, language=language,
            final_citation_source_ids=final_citation_source_ids)
    except (stop_error, PartitionError) as error:
        verification.update(status="unverified", interruption_reason=str(error))
        final = deepcopy(verification.get("accepted_final", final))
        verification["accepted_final_hash"] = hash_json(final)
    return final, original_final, verification
