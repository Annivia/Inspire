#!/usr/bin/env python3
from __future__ import annotations
import csv, sys
from pathlib import Path
from typing import Dict, List, Tuple

def load_concept_csv(path: Path):
    concepts=[]; series={}; meta=[]
    with path.open('r', newline='') as f:
        reader=csv.reader(f); rows=list(reader)
        i=0
        if rows and rows[0] and isinstance(rows[0][0], str) and rows[0][0].startswith('#'):
            meta=rows[0]; i=1
        for r in rows[i+1:]:
            if not r: continue
            name=r[0]
            try:
                vals=[int(x) for x in r[1:]]
            except Exception:
                vals=[]
            concepts.append(name); series[name]=vals
    return concepts, series, meta

def write_concept_csv(path: Path, order: List[str], merged: Dict[str,List[int]], meta: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w=csv.writer(f)
        if meta: w.writerow(meta)
        max_t=max((len(merged.get(c, [])) for c in order), default=0)
        w.writerow(['concept']+[f't{i}' for i in range(max_t)])
        for c in order:
            vals=merged.get(c, [])
            if len(vals)<max_t: vals=vals+[0]*(max_t-len(vals))
            w.writerow([c]+vals)

def main(argv: List[str])->int:
    if len(argv)<2:
        print('Usage: combine_concepts.py <concepts_root> [output_dir]', file=sys.stderr); return 2
    root=Path(argv[1]).resolve()
    out_dir=Path(argv[2]).resolve() if len(argv)>2 else (root/'combined')
    if not root.exists():
        print(f'ERROR: concepts_root not found: {root}', file=sys.stderr); return 2
    files=list(root.glob('gpu_process_*/**/*__relations.csv'))
    if not files:
        files=list(root.glob('*__relations.csv'))
    if not files:
        print(f'No concept CSVs found under {root}'); return 0
    groups={}
    for p in files:
        groups.setdefault(p.name, []).append(p)
    print(f'[combine] Found {sum(len(v) for v in groups.values())} files across {len(groups)} tasks')
    for base, paths in groups.items():
        order0, series0, meta0=load_concept_csv(paths[0])
        merged={c: series0.get(c, [])[:] for c in order0}
        for p in paths[1:]:
            order_i, series_i, _=load_concept_csv(p)
            for c in order0:
                merged.setdefault(c, [])
                merged[c].extend(series_i.get(c, []))
            for c in order_i:
                if c not in merged:
                    merged[c]=series_i.get(c, [])[:]
                    order0.append(c)
        out_path=out_dir/base
        write_concept_csv(out_path, order0, merged, meta0)
        print(f'[combine] Wrote {out_path}')
    return 0

if __name__=='__main__':
    from typing import List
    raise SystemExit(main(sys.argv))
