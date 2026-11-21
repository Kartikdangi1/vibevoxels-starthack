"""Merge physics probes (2016-2026) into an Eidothea sweep catalog and abstracts file,
tag with ``physics_taxonomy``, and prepare for ``fetch-citations``, ``fetch-references``,
and ``graph build``.

    python -m eidothea.analysis.build_physics_sweep data/physics_probes --out data/physics_graph
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

from eidothea.analysis.analyze_robotics_sweep import salience, venue
from eidothea.taxonomy.physics_taxonomy import SUBFIELDS, tag, tag_title


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("probe_dir", help="directory written by scan-arxiv-physics")
    ap.add_argument("--out", required=True, help="output sweep directory")
    ap.add_argument("--top", type=int, default=60)
    args = ap.parse_args()

    probe_dir = Path(args.probe_dir)
    out = Path(args.out)
    (out / "top").mkdir(parents=True, exist_ok=True)

    papers: dict[str, dict] = {}
    for fp in sorted(probe_dir.glob("*.jsonl")):
        probe_name = fp.stem
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            pid = r["id"]
            if pid not in papers:
                r["probes"] = [probe_name]
                papers[pid] = r
            elif probe_name not in papers[pid]["probes"]:
                papers[pid]["probes"].append(probe_name)

    plist = sorted(papers.values(), key=lambda x: (x.get("created", ""), x["id"]), reverse=True)
    print(f"Loaded {len(plist)} unique papers across {len(list(probe_dir.glob('*.jsonl')))} physics probes (2016-2026)")

    by_year: collections.Counter[str] = collections.Counter()
    cats: collections.Counter[str] = collections.Counter()
    venues: collections.Counter[str] = collections.Counter()
    tag_papers: dict[str, list[dict]] = {i: [] for i, *_ in SUBFIELDS}
    untagged = 0

    with gzip.open(out / "catalog.tsv.gz", "wt", encoding="utf-8") as cat:
        cat.write("id\tcreated\tcategories\ttags\tsalience\tvenue\tcode\ttitle\n")
        for r in plist:
            tags = tag(r.get("title", ""), r.get("abstract", ""))
            r["tags"] = tags
            r["salience"] = salience(r) + min(len(r.get("probes", [])) - 1, 3)
            year = (r.get("created") or "unknown")[:4]
            by_year[year] += 1
            for c in (r.get("categories") or "").split():
                cats[c] += 1
            v = venue(r)
            if v:
                venues[v] += 1
            if not tags:
                untagged += 1
            for t in tags:
                tag_papers[t].append(r)
            cat.write("\t".join([
                r["id"],
                r.get("created", ""),
                r.get("categories", ""),
                ",".join(tags),
                str(r["salience"]),
                v,
                "1" if r.get("code") else "",
                (r.get("title") or "").replace("\t", " "),
            ]) + "\n")

    with gzip.open(out / "abstracts.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in plist:
            f.write(json.dumps({
                "id": r["id"],
                "month": (r.get("created") or "")[:7],
                "created": r.get("created", ""),
                "categories": r.get("categories", ""),
                "title": r.get("title", ""),
                "authors": r.get("authors", []),
                "n_authors": r.get("n_authors", 0),
                "comments": r.get("comments", ""),
                "journal_ref": r.get("journal_ref", ""),
                "doi": r.get("doi", ""),
                "code": r.get("code", []),
                "probes": r.get("probes", []),
                "tags": r.get("tags", []),
                "salience": r.get("salience", 0),
                "abstract": r.get("abstract", ""),
            }) + "\n")

    summary = {
        "n_papers": len(plist),
        "window": "2016-01 to 2026-11",
        "untagged": untagged,
        "years": dict(sorted(by_year.items())),
        "categories": dict(cats.most_common(40)),
        "venues": dict(venues.most_common(30)),
        "subfields": {},
    }

    for sid, group, label, _rx in SUBFIELDS:
        tp = tag_papers[sid]
        summary["subfields"][sid] = {
            "group": group,
            "label": label,
            "count": len(tp),
            "with_code": sum(1 for p in tp if p.get("code")),
        }
        top = sorted(
            tp,
            key=lambda p: (
                sid in tag_title(p.get("title", "")),
                p.get("salience", 0),
                p.get("created", ""),
            ),
            reverse=True,
        )
        with open(out / "top" / f"{sid}.jsonl", "w", encoding="utf-8") as f:
            for p in top[:args.top]:
                f.write(json.dumps({
                    k: p.get(k) for k in (
                        "id", "created", "title", "categories", "authors",
                        "comments", "journal_ref", "doi", "salience",
                        "code", "probes", "tags", "abstract"
                    )
                }) + "\n")

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Untagged: {untagged} ({untagged / max(len(plist), 1):.1%})")
    for sid, s in sorted(summary["subfields"].items(), key=lambda kv: -kv[1]["count"]):
        print(f"  {sid:<18s} | {s['count']:4d} papers | {s['with_code']:3d} code | {s['label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
