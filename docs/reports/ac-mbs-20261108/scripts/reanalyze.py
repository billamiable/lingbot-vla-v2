#!/usr/bin/env python3
"""Recompute archived metrics without a GPU or changes to original evidence."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import re
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path(__file__).resolve().parents[1] / 'evidence')
    parser.add_argument('--output', type=Path, required=True, help='New directory for recomputed JSON files')
    args = parser.parse_args()
    source = args.evidence.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error('--output must be a new directory')
    with tempfile.TemporaryDirectory(prefix='lingbot-reanalyze-') as temporary:
        root = Path(temporary)
        (root / 'runs').symlink_to(source / 'runs', target_is_directory=True)
        for name in ['summarize_study.py', 'audit_study.py', 'aggregate_study.py']:
            code = (source / name).read_text()
            code, count = re.subn(r"^ROOT\s*=\s*Path\('/root/experiments/lingbot-ac-mbs-20261108'\)",
                                  'ROOT = Path(' + repr(str(root)) + ')', code, flags=re.MULTILINE)
            if count != 1:
                raise ValueError(f'Unexpected ROOT declaration in {name}')
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code, str(source / name), 'exec'), {'__name__': '__main__'})
        result = json.loads((root / 'aggregate.json').read_text())
        assert result['ready'] and all(c['finite'] for c in result['cells'].values())
        expected = json.loads((source / 'aggregate.json').read_text())
        assert result == expected, 'Recomputed aggregate differs from archived result'
        audit = json.loads((root / 'consistency_audit.json').read_text())
        pairs = [p for p in audit['input_pairs'] if p['a'].startswith('alloc_')]
        assert len(pairs) == 4 and all(p['compared_updates'] == 24 and not p['differences'] for p in pairs)
        output.mkdir(parents=True)
        for name in ['summary_all.json', 'consistency_audit.json', 'aggregate.json']:
            (output / name).write_bytes((root / name).read_bytes())
    print('PASS: all four cells, two complete runs each; archived aggregate reproduced exactly.')
    print(output)


if __name__ == '__main__':
    main()
