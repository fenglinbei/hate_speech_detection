from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CATEGORY_FLAGS = [
    ("is_about_nationality", "nationality"),
    ("is_about_ethnicity", "ethnicity"),
    ("is_about_religion", "religion"),
    ("is_about_gender", "gender"),
    ("is_about_sexual_orientation", "sexual_orientation"),
    ("is_about_disability", "disability"),
    ("is_about_class", "class"),
]


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def page_number(path: Path) -> int:
    match = re.search(r"page_(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def clean_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text


def normalize_categories(raw: dict[str, Any]) -> list[str]:
    categories = [name for field, name in CATEGORY_FLAGS if bool(raw.get(field))]
    return categories or ["other"]


def normalize_term(raw: dict[str, Any]) -> dict[str, Any]:
    categories = normalize_categories(raw)
    term = clean_value(raw.get("term")) or ""
    variants = []
    for key in ("plural_of", "variant_of", "transliteration_of"):
        value = clean_value(raw.get(key))
        if value and value.lower() != term.lower() and value not in variants:
            variants.append(value)

    metadata_keys = [
        "vocabulary_id",
        "is_unambiguous_in",
        "plural_of",
        "variant_of",
        "transliteration_of",
        "number_of_sightings",
        "number_of_sightings_this_year",
        "number_of_sightings_this_month",
        "created_on",
        "updated_on",
    ]
    metadata = {key: raw.get(key) for key in metadata_keys if key in raw}

    return {
        "term": term,
        "variants": variants,
        "language": "en",
        "category": categories[0],
        "categories": categories,
        "definition": clean_value(raw.get("hateful_meaning")) or "",
        "nonhateful_meaning": clean_value(raw.get("nonhateful_meaning")) or "",
        "average_offensiveness": raw.get("average_offensiveness"),
        "is_unambiguous": bool(raw.get("is_unambiguous")),
        "metadata": metadata,
    }


def convert_hatebase(
        input_dir: str | Path,
        output: str | Path,
        language: str = "eng",
        expected_total: int = 1531,
        expected_pages: int = 16,
        ) -> dict[str, Any]:
    input_dir = Path(input_dir)
    pages = sorted(input_dir.glob("page_*.json"), key=page_number)
    if not pages:
        raise FileNotFoundError(f"No page_*.json files found under {input_dir}.")

    raw_terms: list[dict[str, Any]] = []
    page_numbers: list[int] = []
    for page_path in pages:
        payload = load_json(page_path)
        page_numbers.append(int(payload.get("page", page_number(page_path))))
        if int(payload.get("number_of_pages", expected_pages)) != expected_pages:
            raise ValueError(f"Unexpected number_of_pages in {page_path}: {payload.get('number_of_pages')}")
        if int(payload.get("number_of_results", expected_total)) != expected_total:
            raise ValueError(f"Unexpected number_of_results in {page_path}: {payload.get('number_of_results')}")
        result = payload.get("result", [])
        if not isinstance(result, list):
            raise ValueError(f"Expected 'result' list in {page_path}.")
        raw_terms.extend(result)

    if sorted(page_numbers) != list(range(1, expected_pages + 1)):
        raise ValueError(f"Expected pages 1..{expected_pages}, got {sorted(page_numbers)}.")

    filtered = [
        raw
        for raw in raw_terms
        if str(raw.get("language", "")).strip().lower() == language
    ]
    terms = [normalize_term(raw) for raw in filtered]
    if len(terms) != expected_total:
        raise ValueError(f"Expected {expected_total} {language} terms, got {len(terms)}.")

    output_payload = {
        "source": "hatebase",
        "language": "en",
        "input_dir": str(input_dir),
        "total_terms": len(terms),
        "terms": terms,
    }
    dump_json(output_payload, output)

    report = {
        "input_dir": str(input_dir),
        "output": str(output),
        "pages": len(pages),
        "raw_terms": len(raw_terms),
        "terms": len(terms),
        "language": language,
    }
    dump_json(report, Path(output).with_name("hatebase_conversion_report.json"))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local HateBase page JSON files to lexicon format.")
    parser.add_argument("--input-dir", default="data/lexicon/hateBase")
    parser.add_argument("--output", default="data/lexicon/hateBase/processed/hatebase_en.json")
    parser.add_argument("--language", default="eng")
    parser.add_argument("--expected-total", type=int, default=1531)
    parser.add_argument("--expected-pages", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_hatebase(
        input_dir=args.input_dir,
        output=args.output,
        language=args.language,
        expected_total=args.expected_total,
        expected_pages=args.expected_pages,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
