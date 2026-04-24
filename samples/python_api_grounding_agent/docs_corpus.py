from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PublicDoc:
    doc_id: str
    title: str
    body: str
    distilled: str


PUBLIC_DOCS = [
    PublicDoc(
        doc_id="pathlib_core",
        title="pathlib.Path core helpers",
        body=(
            "Path.cwd() returns a new path pointing at the current working directory. "
            "Path.home() returns a new path pointing at the user's home directory. "
            "Path.exists() returns True when the path points to an existing filesystem entry. "
            "Path.is_file() returns True when the path points to a regular file."
        ),
        distilled=(
            "Path.cwd(): current working directory. "
            "Path.home(): user's home directory. "
            "Path.exists(): True if the path exists. "
            "Path.is_file(): True if the path is a regular file."
        ),
    ),
    PublicDoc(
        doc_id="json_basics",
        title="json encode and decode",
        body=(
            "json.dumps() serializes a Python object into a JSON-formatted string. "
            "json.loads() parses a JSON document and returns the corresponding Python object."
        ),
        distilled=(
            "json.dumps(): object -> JSON string. "
            "json.loads(): JSON document -> Python object."
        ),
    ),
    PublicDoc(
        doc_id="subprocess_run",
        title="subprocess.run arguments",
        body=(
            "subprocess.run(..., capture_output=True) captures both stdout and stderr in the returned "
            "CompletedProcess object. The check=True argument raises CalledProcessError when the command exits non-zero. "
            "The text=True argument decodes stdout and stderr to strings."
        ),
        distilled=(
            "capture_output=True captures stdout and stderr. "
            "check=True raises CalledProcessError on non-zero exit. "
            "text=True decodes captured streams to strings."
        ),
    ),
    PublicDoc(
        doc_id="statistics_core",
        title="statistics helpers",
        body=(
            "statistics.fmean() computes the floating-point arithmetic mean of data. "
            "statistics.median() returns the median of numeric data."
        ),
        distilled=(
            "statistics.fmean(): floating-point arithmetic mean. "
            "statistics.median(): median of numeric data."
        ),
    ),
    PublicDoc(
        doc_id="counter_common",
        title="collections.Counter methods",
        body=(
            "Counter.most_common() returns a list of the most common elements and their counts, "
            "ordered from most common to least common."
        ),
        distilled="Counter.most_common(): elements ordered from most common to least common.",
    ),
    PublicDoc(
        doc_id="itertools_chain",
        title="itertools chain helpers",
        body=(
            "itertools.chain.from_iterable() builds an iterator that yields elements from each iterable "
            "in an iterable of iterables."
        ),
        distilled="itertools.chain.from_iterable(): chain iterables from an iterable of iterables.",
    ),
    PublicDoc(
        doc_id="datetime_isoformat",
        title="datetime.date formatting",
        body=(
            "date.isoformat() returns the date formatted as an ISO 8601 string such as YYYY-MM-DD."
        ),
        distilled="date.isoformat(): ISO 8601 date string.",
    ),
]


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_().=-]+", text.lower()))
