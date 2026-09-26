#!/usr/bin/env python3
"""Business entity resolution pipeline through phase 6.

The implementation uses SQLite as a disk-backed blocking index so that the
millions of source records are never loaded into one Python dictionary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple

from rapidfuzz import fuzz

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - exercised only when LightGBM is absent
    lgb = None

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:  # pragma: no cover
    HistGradientBoostingClassifier = None


FIELDS = ("entity_id", "business_name", "business_address", "country")
LEGAL = {
    "corp": "corporation", "co": "company", "inc": "incorporated",
    "ltd": "limited", "pvt": "private", "llc": "limited liability company",
}
STREET = {"rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard", "dr": "drive"}


def normalize(value: str) -> str:
    value = (value or "").lower().replace("&", " and ")
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def normalize_name(value: str) -> str:
    tokens = [LEGAL.get(token, token) for token in normalize(value).split()]
    return " ".join(tokens)


def normalize_address(value: str) -> str:
    tokens = [STREET.get(token, token) for token in normalize(value).split()]
    return " ".join(tokens)


def numeric_key(address: str) -> str:
    return " ".join(re.findall(r"\d+", address or ""))


def first_token(value: str) -> str:
    return (value.split(" ", 1)[0] if value else "")[:4]


def row_features(row: Dict[str, str]) -> Dict[str, str]:
    name = normalize_name(row.get("business_name", ""))
    address = normalize_address(row.get("business_address", ""))
    return {
        "name": name,
        "name_sorted": " ".join(sorted(name.split())),
        "address": address,
        "country": normalize(row.get("country", "")),
        "numbers": numeric_key(address),
        "name_prefix": first_token(name),
    }


SOURCE_COLUMNS = {"entity_id", "business_name", "business_address", "country"}
GROUND_TRUTH_COLUMNS = {"source1_entity_id", "matched_entity_ids"}


def read_rows(path: Path, expected_columns: Set[str] | None = None) -> Iterator[Dict[str, str]]:
    expected = expected_columns or (GROUND_TRUTH_COLUMNS if path.name == "train_ground_truth.tsv" else SOURCE_COLUMNS)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not expected.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Invalid TSV header in {path}: found {reader.fieldnames}; expected {sorted(expected)}")
        yield from reader


def create_index(db_path: Path, source_paths: Sequence[Path], reuse_existing: bool = True) -> None:
    required_columns = {"id", "name", "name_sorted", "address", "country", "numbers", "name_prefix"}
    if db_path.exists() and reuse_existing:
        check_connection = None
        try:
            check_connection = sqlite3.connect(str(db_path), timeout=30)
            columns = {row[1] for row in check_connection.execute("PRAGMA table_info(records)")}
            integrity = check_connection.execute("PRAGMA integrity_check").fetchone()[0]
            if required_columns.issubset(columns) and integrity == "ok":
                check_connection.close()
                return
        except sqlite3.DatabaseError:
            pass
        finally:
            if check_connection is not None:
                check_connection.close()
        db_path.unlink(missing_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("CREATE TABLE records (id TEXT PRIMARY KEY, name TEXT, name_sorted TEXT, address TEXT, country TEXT, numbers TEXT, name_prefix TEXT)")
    connection.execute("CREATE INDEX records_name ON records(name, country)")
    connection.execute("CREATE INDEX records_name_sorted ON records(name_sorted, country)")
    connection.execute("CREATE INDEX records_prefix ON records(name_prefix, country)")
    connection.execute("CREATE INDEX records_numbers ON records(numbers, country)")
    batch = []
    for path in source_paths:
        for row in read_rows(path):
            values = row_features(row)
            batch.append((row["entity_id"], values["name"], values["name_sorted"], values["address"], values["country"], values["numbers"], values["name_prefix"]))
            if len(batch) >= 10000:
                connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                connection.commit()
                batch.clear()
    if batch:
        connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
    connection.commit()
    connection.close()


def candidate_rows(connection: sqlite3.Connection, source_row: Dict[str, str], cap: int) -> List[Tuple[str, str, str, str, str, str]]:
    values = row_features(source_row)
    found: Dict[str, Tuple[str, str, str, str, str, str]] = {}
    queries = []
    if values["name"]:
        queries.append(("SELECT id,name,name_sorted,address,country,numbers FROM records WHERE name=? AND country=? LIMIT ?", (values["name"], values["country"], cap)))
    if values["name_sorted"]:
        queries.append(("SELECT id,name,name_sorted,address,country,numbers FROM records WHERE name_sorted=? AND country=? LIMIT ?", (values["name_sorted"], values["country"], cap)))
    if values["name_prefix"]:
        queries.append(("SELECT id,name,name_sorted,address,country,numbers FROM records WHERE name_prefix=? AND country=? LIMIT ?", (values["name_prefix"], values["country"], cap)))
    if values["numbers"]:
        queries.append(("SELECT id,name,name_sorted,address,country,numbers FROM records WHERE numbers=? AND country=? LIMIT ?", (values["numbers"], values["country"], cap)))
    for sql, params in queries:
        for candidate in connection.execute(sql, params):
            found.setdefault(candidate[0], candidate)
    def rank(candidate):
        _, name, name_sorted, address, country, numbers = candidate
        return (int(name == values["name"]) * 1000 + int(name_sorted == values["name_sorted"]) * 500 + int(numbers and numbers == values["numbers"]) * 300 + fuzz.token_sort_ratio(values["name"], name) + fuzz.ratio(values["address"], address) * 0.25)
    return sorted(found.values(), key=rank, reverse=True)[:cap]


def similarity_features(left: Dict[str, str], right: Tuple[str, str, str, str, str, str]) -> List[float]:
    _, right_name, right_name_sorted, right_address, right_country, right_numbers = right
    left_values = row_features(left)
    left_name = left_values["name"]
    left_address = left_values["address"]
    left_tokens, right_tokens = set(left_name.split()), set(right_name.split())
    address_tokens, other_address_tokens = set(left_address.split()), set(right_address.split())
    name_union = len(left_tokens | right_tokens) or 1
    address_union = len(address_tokens | other_address_tokens) or 1
    return [
        float(fuzz.ratio(left_name, right_name)) / 100.0,
        float(fuzz.token_sort_ratio(left_name, right_name)) / 100.0,
        float(left_values["name_sorted"] == right_name_sorted),
        len(left_tokens & right_tokens) / name_union,
        float(fuzz.ratio(left_address, right_address)) / 100.0,
        len(address_tokens & other_address_tokens) / address_union,
        float(bool(left_values["numbers"] and left_values["numbers"] == right_numbers)),
        float(left_values["country"] == right_country),
        float(not left_address or not right_address),
        float(abs(len(left_tokens) - len(right_tokens))),
    ]


def truth_map(path: Path) -> Dict[str, Set[str]]:
    result = {}
    for row in read_rows(path, GROUND_TRUTH_COLUMNS):
        result[row["source1_entity_id"]] = {item for item in row["matched_entity_ids"].split(",") if item}
    return result


def f05(actual: Set[str], predicted: Set[str]) -> float:
    if not actual and not predicted:
        return 1.0
    if not predicted or not actual:
        return 0.0
    precision = len(actual & predicted) / len(predicted)
    recall = len(actual & predicted) / len(actual)
    denominator = 0.25 * precision + recall
    if denominator == 0:
        return 0.0
    return 1.25 * precision * recall / denominator


def make_training_data(s1_path: Path, gt_path: Path, connection: sqlite3.Connection, cap: int, sample_size: int, seed: int) -> Tuple[List[List[float]], List[int], List[str]]:
    labels = truth_map(gt_path)
    features: List[List[float]] = []
    targets: List[int] = []
    groups: List[str] = []
    for index, row in enumerate(read_rows(s1_path)):
        digest = int(hashlib.md5(row["entity_id"].encode()).hexdigest(), 16)
        if digest % 10 == 7 or digest % 100000 >= max(1, sample_size // 25):
            continue
        candidates = candidate_rows(connection, row, cap)
        actual = labels.get(row["entity_id"], set())
        for candidate in candidates:
            features.append(similarity_features(row, candidate))
            targets.append(int(candidate[0] in actual))
            groups.append(row["entity_id"])
    if not features:
        raise RuntimeError("No training candidates were generated; loosen blocking or inspect the input paths.")
    return features, targets, groups


def fit_model(features: List[List[float]], targets: List[int]):
    if lgb is not None:
        model = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.05, num_leaves=31, max_depth=8, random_state=17, verbosity=-1)
    elif HistGradientBoostingClassifier is not None:
        model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, random_state=17)
    else:
        raise RuntimeError("Install lightgbm or scikit-learn before running the pipeline.")
    model.fit(features, targets)
    return model


def predict_probability(model, features: List[List[float]]) -> List[float]:
    if hasattr(model, "predict_proba"):
        return [float(item[1]) for item in model.predict_proba(features)]
    return [float(item) for item in model.predict(features)]


def tune_threshold(model, s1_path: Path, gt_path: Path, connection: sqlite3.Connection, cap: int, validation_mod: int) -> Tuple[float, float]:
    labels = truth_map(gt_path)
    validation: List[Tuple[str, Set[str], List[Tuple[str, float]]]] = []
    for row in read_rows(s1_path):
        if int(hashlib.md5(row["entity_id"].encode()).hexdigest(), 16) % 10 != validation_mod:
            continue
        candidates = candidate_rows(connection, row, cap)
        probabilities = predict_probability(model, [similarity_features(row, candidate) for candidate in candidates])
        validation.append((row["entity_id"], labels.get(row["entity_id"], set()), [(candidate[0], score) for candidate, score in zip(candidates, probabilities)]))
    best_threshold, best_score = 0.5, -1.0
    for threshold in [x / 100 for x in range(30, 100, 2)]:
        score = sum(f05(actual, {item for item, probability in scored if probability >= threshold}) for _, actual, scored in validation) / max(1, len(validation))
        if score > best_score:
            best_threshold, best_score = threshold, score
    return best_threshold, best_score


def write_outputs(model, threshold: float, s1_path: Path, connection: sqlite3.Connection, cap: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    with matching_path.open("w", encoding="utf-8", newline="") as matching, candidate_path.open("w", encoding="utf-8", newline="") as candidates:
        matching_writer = csv.writer(matching, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidates, delimiter="\t", lineterminator="\n")
        matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for row in read_rows(s1_path):
            possible = candidate_rows(connection, row, cap)
            probabilities = predict_probability(model, [similarity_features(row, candidate) for candidate in possible])
            candidate_ids = [candidate[0] for candidate in possible]
            match_ids = [candidate_id for candidate_id, probability in zip(candidate_ids, probabilities) if probability >= threshold]
            candidate_writer.writerow([row["entity_id"], ",".join(candidate_ids)])
            matching_writer.writerow([row["entity_id"], ",".join(match_ids)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="student_resource directory")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--candidate-cap", type=int, default=60)
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = args.data_root / "dataset" / "train"
    test = args.data_root / "dataset" / "test"
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="entity_resolution_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "source23.sqlite"
    create_index(db_path, [train / "train_source2.tsv", train / "train_source3.tsv"])
    connection = sqlite3.connect(str(db_path))
    features, targets, groups = make_training_data(train / "train_source1.tsv", train / "train_ground_truth.tsv", connection, args.candidate_cap, args.sample_size, 17)
    model = fit_model(features, targets)
    threshold, score = tune_threshold(model, train / "train_source1.tsv", train / "train_ground_truth.tsv", connection, args.candidate_cap, 7)
    print(f"training_pairs={len(targets)} positives={sum(targets)} validation_macro_f05={score:.6f} threshold={threshold:.2f}")
    if not args.skip_inference:
        connection.close()
        test_db = work_dir / "test_source23.sqlite"
        create_index(test_db, [test / "test_source2.tsv", test / "test_source3.tsv"])
        test_connection = sqlite3.connect(str(test_db))
        write_outputs(model, threshold, test / "test_source1.tsv", test_connection, args.candidate_cap, args.output_dir)
        test_connection.close()
    else:
        connection.close()


if __name__ == "__main__":
    main()