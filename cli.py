"""Run: python cli.py examples/sample_inbox.json --today 2026-10-24 --out brief.md"""
import argparse
import json
from pathlib import Path
from quietqueue.core import triage, to_markdown


def main():
    parser = argparse.ArgumentParser(description='Offline human-in-the-loop AI task triage')
    parser.add_argument('input', help='Input JSON list of task objects, stored locally')
    parser.add_argument('--today', default='2026-10-24', help='Reference date (YYYY-MM-DD)')
    parser.add_argument('--out', default='brief.md', help='Markdown report filename')
    parser.add_argument('--json-out', default='', help='Optional structured JSON output filename')
    args = parser.parse_args()
    records = json.loads(Path(args.input).read_text(encoding='utf-8'))
    result = triage(records, today=args.today)
    Path(args.out).write_text(to_markdown(result), encoding='utf-8')
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(to_markdown(result))
    print(f'Report written to {args.out}. No network calls, mail sends, or deletions.')


if __name__ == '__main__':
    main()
