"""Prepare local text/jsonl data files for ArcLM examples."""

import argparse
import json
from pathlib import Path


def read_records(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    return [{"text": path.read_text(encoding="utf-8")}]


def _read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def record_to_text(record):
    if isinstance(record, str):
        return record.strip()
    if not isinstance(record, dict):
        return str(record).strip()

    question = record.get("question") or record.get("prompt") or record.get("instruction")
    answer = record.get("answer") or record.get("response") or record.get("completion")
    if question and answer:
        return f"Question: {question}\nAnswer: {answer}".strip()

    if "text" in record:
        return str(record["text"]).strip()

    parts = [str(value).strip() for value in record.values() if isinstance(value, (str, int, float))]
    return "\n".join(part for part in parts if part)


def split_texts(texts, validation_split):
    clean = [text for text in texts if text]
    if not clean or validation_split <= 0:
        return clean, []
    val_count = max(1, int(len(clean) * validation_split))
    if len(clean) <= val_count:
        return clean, []
    return clean[:-val_count], clean[-val_count:]


def write_lines(path, texts):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(texts).strip() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prepare data/pretrain.txt, data/finetune.txt, and data/val.txt.")
    parser.add_argument("inputs", nargs="+", help="Input .txt or .jsonl files.")
    parser.add_argument("--output-dir", default="data", help="Directory for prepared files.")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Fraction of records held out for val.txt.")
    args = parser.parse_args()

    texts = []
    qa_texts = []
    for input_path in args.inputs:
        for record in read_records(input_path):
            text = record_to_text(record)
            if text:
                texts.append(text)
            if isinstance(record, dict) and (
                (record.get("question") or record.get("prompt") or record.get("instruction"))
                and (record.get("answer") or record.get("response") or record.get("completion"))
            ):
                qa_texts.append(text)

    train_texts, val_texts = split_texts(texts, args.validation_split)
    finetune_texts = qa_texts or train_texts

    output_dir = Path(args.output_dir)
    write_lines(output_dir / "pretrain.txt", train_texts)
    write_lines(output_dir / "finetune.txt", finetune_texts)
    write_lines(output_dir / "val.txt", val_texts)

    print(f"Wrote {output_dir / 'pretrain.txt'} ({len(train_texts)} records)")
    print(f"Wrote {output_dir / 'finetune.txt'} ({len(finetune_texts)} records)")
    print(f"Wrote {output_dir / 'val.txt'} ({len(val_texts)} records)")


if __name__ == "__main__":
    main()
