"""DynamoDB implementation of ``JobStorePort`` using conditional writes.

Table: partition key ``pk`` (S). Jobs use ``pk=job#<id>``, deliveries ``pk=delivery#<id>``.
GSI ``pr_index`` on ``pr_key`` and sparse GSI ``active_index`` on ``active`` (set only for
active jobs). Only job metadata and the metadata-only trace are stored: no source, prompts
or model responses.
"""
from __future__ import annotations

import json
from dataclasses import asdict

from review_bot.application.ondemand import ACTIVE_STATES, ReviewJob


class DynamoDbJobStore:
    def __init__(self, client, table_name: str) -> None:
        self.client, self.table = client, table_name

    @staticmethod
    def _item(job: ReviewJob) -> dict:
        item = {"pk": {"S": f"job#{job.job_id}"}, "version": {"N": str(job.version)},
            "pr_key": {"S": f"{job.repository_id}#{job.pr_number}"}, "doc": {"S": json.dumps(asdict(job), sort_keys=True)}}
        if job.state in ACTIVE_STATES:
            item["active"] = {"S": "1"}
        return item

    @staticmethod
    def _job(item: dict) -> ReviewJob:
        return ReviewJob(**json.loads(item["doc"]["S"]))

    def _conditional(self, call, **kwargs) -> bool:
        try:
            call(**kwargs)
            return True
        except self.client.exceptions.ConditionalCheckFailedException:
            return False

    def record_delivery(self, delivery_id: str, payload_digest: str, received_at: float) -> str | None:
        if self._conditional(self.client.put_item, TableName=self.table, Item={"pk": {"S": f"delivery#{delivery_id}"},
                "digest": {"S": payload_digest}, "received_at": {"N": repr(received_at)}},
                ConditionExpression="attribute_not_exists(pk)"):
            return None
        stored = self.client.get_item(TableName=self.table, Key={"pk": {"S": f"delivery#{delivery_id}"}},
            ConsistentRead=True)["Item"]
        return stored["digest"]["S"]

    def get(self, job_id: str) -> ReviewJob | None:
        item = self.client.get_item(TableName=self.table, Key={"pk": {"S": f"job#{job_id}"}}, ConsistentRead=True).get("Item")
        return self._job(item) if item else None

    def put_new(self, job: ReviewJob) -> bool:
        return self._conditional(self.client.put_item, TableName=self.table, Item=self._item(job),
            ConditionExpression="attribute_not_exists(pk)")

    def replace_if_version(self, job: ReviewJob, expected_version: int) -> bool:
        if job.version != expected_version + 1:
            return False
        return self._conditional(self.client.put_item, TableName=self.table, Item=self._item(job),
            ConditionExpression="#v = :expected", ExpressionAttributeNames={"#v": "version"},
            ExpressionAttributeValues={":expected": {"N": str(expected_version)}})

    def _query(self, **kwargs) -> list[ReviewJob]:
        jobs, start = [], None
        while True:
            page = self.client.query(TableName=self.table, **kwargs, **({"ExclusiveStartKey": start} if start else {}))
            # GSIs are eventually consistent: re-read each job before acting on it.
            jobs.extend(job for item in page["Items"]
                if (job := self.get(item["pk"]["S"].removeprefix("job#"))) is not None)
            start = page.get("LastEvaluatedKey")
            if not start:
                return jobs

    def list_for_pr(self, repository_id: int, pr_number: int) -> list[ReviewJob]:
        return self._query(IndexName="pr_index", KeyConditionExpression="pr_key = :k",
            ExpressionAttributeValues={":k": {"S": f"{repository_id}#{pr_number}"}})

    def list_active(self) -> list[ReviewJob]:
        return [j for j in self._query(IndexName="active_index", KeyConditionExpression="active = :a",
            ExpressionAttributeValues={":a": {"S": "1"}}) if j.state in ACTIVE_STATES]
