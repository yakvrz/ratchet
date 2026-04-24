from __future__ import annotations

from dataclasses import dataclass
import re


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class RunbookDoc:
    doc_id: str
    title: str
    body: str
    distilled: str


RUNBOOK_DOCS: tuple[RunbookDoc, ...] = (
    RunbookDoc(
        doc_id="redis_memory",
        title="Redis memory saturation",
        body=(
            "When Redis memory exceeds the eviction threshold, the first step is to inspect memory details with "
            "`redis-cli INFO memory` before changing eviction policies."
        ),
        distilled="Redis memory alert -> first run `redis-cli INFO memory`.",
    ),
    RunbookDoc(
        doc_id="k8s_crashloop",
        title="Kubernetes CrashLoopBackOff",
        body=(
            "For pods in CrashLoopBackOff, collect the previous container logs with "
            "`kubectl logs <pod> --previous` before restarting or rolling back."
        ),
        distilled="CrashLoopBackOff -> first run `kubectl logs <pod> --previous`.",
    ),
    RunbookDoc(
        doc_id="postgres_replication",
        title="Postgres replication lag",
        body=(
            "When replica lag spikes, first check the receive LSN on the replica with "
            "`SELECT pg_last_wal_receive_lsn();`."
        ),
        distilled="Replication lag -> first run `SELECT pg_last_wal_receive_lsn();`.",
    ),
    RunbookDoc(
        doc_id="deploy_5xx",
        title="5xx spike after deployment",
        body=(
            "If a deploy coincides with a broad 5xx spike, roll back the last deploy before deeper debugging."
        ),
        distilled="Broad 5xx spike after deploy -> first step is `roll back the last deploy`.",
    ),
    RunbookDoc(
        doc_id="disk_pressure",
        title="Disk pressure on app host",
        body=(
            "When disk usage exceeds 95 percent, identify the largest log directories with "
            "`du -sh /var/log/* | sort -h`."
        ),
        distilled="Disk pressure -> first run `du -sh /var/log/* | sort -h`.",
    ),
    RunbookDoc(
        doc_id="cdn_stale",
        title="Stale assets behind CDN",
        body=(
            "If a release succeeded but users still see stale assets, purge the CDN cache for the affected paths."
        ),
        distilled="Stale CDN assets -> first step is `purge the CDN cache for the affected paths`.",
    ),
    RunbookDoc(
        doc_id="tls_renewal",
        title="TLS certificate expiring",
        body=(
            "When a production certificate is within 24 hours of expiry, renew the certificate immediately."
        ),
        distilled="Certificate expiring soon -> first step is `renew the certificate`.",
    ),
    RunbookDoc(
        doc_id="queue_backlog",
        title="Consumer queue backlog",
        body=(
            "If the consumer deployment is healthy but queue backlog grows after a rollout, scale the consumer deployment."
        ),
        distilled="Queue backlog after rollout -> first step is `scale the consumer deployment`.",
    ),
)


def tokenize(text: str) -> set[str]:
    return {token for token in TOKEN_PATTERN.findall(text.lower()) if token}
