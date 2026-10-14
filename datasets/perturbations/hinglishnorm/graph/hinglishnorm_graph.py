# /// script
# requires-python = ">=3.10"
# dependencies = ["networkx>=3.2", "pyvis>=0.3.2"]
# ///
"""Build a variant graph from hinglishNorm (Makhija et al., COLING 2020).

Nodes are lowercase word forms (raw spellings and normalized spellings alike).
A directed edge raw -> normalized is added for every aligned token pair where the
two differ; its weight is how many times annotators made that mapping. Tokens left
unchanged (raw == normalized) are counted on the node as `kept` instead of a self-loop.

Inconsistent normalization (nhi -> nahin vs nhi -> nahi) is fine: both targets end up
in the same connected component, which is the variant cluster.

Usage:
    uv run hinglishnorm_graph.py                    # build + summary + exports
    uv run hinglishnorm_graph.py --ego nhi          # neighborhood of a word
    uv run hinglishnorm_graph.py --path ni naee     # how two words are connected
    uv run hinglishnorm_graph.py --html nhi         # interactive view of nhi's component
    uv run hinglishnorm_graph.py --min-share 0.2 --min-weight 2   # pruning thresholds
    uv run hinglishnorm_graph.py --crack            # which edges/words to disambiguate first

Manual disambiguation: add rows to decisions.csv (raw,normalized,verdict) with verdict
`cut` (wrong link, drop it) or `keep` (checked, real variant; never flagged or cut).
normalized `*` cuts every edge out of raw. Every mode applies it before doing anything.

Outputs (default graph/out/):
    nodes.csv, edges.csv          per-node / per-edge stats
    components.csv                raw connected components
    suspect_edges.csv             bridges, weak edges, and high-edit-distance edges
    clusters.csv                  components after pruning suspect edges
    graph.gexf                    open in Gephi for visual exploration
    crack_edges.csv, crack_words.csv   (--crack) disambiguation priorities
    viewer.html                   browser viewer (open directly from disk)
"""

import argparse
import ast
import csv
import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

DATA_URL = "https://raw.githubusercontent.com/piyushmakhija5/hinglishNorm/master/dataset/hinglishNorm.json"
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "hinglishNorm.json"
DEFAULT_OUT = HERE / "out"


# ---------------------------------------------------------------- loading + alignment

def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found; get it from {DATA_URL}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    for r in rows:
        r["tags"] = ast.literal_eval(r["tags"]) if isinstance(r["tags"], str) else r["tags"]
    return rows


def align(raw: list[str], norm: list[str], include_phrases: bool):
    """Yield (raw_index, raw_token, norm_token, kind) pairs.

    Equal-length sentences are zipped (kind="1:1"). Otherwise difflib anchors on
    identical tokens, equal-length replace blocks are zipped, and 1->n / n->1 blocks
    are emitted as phrase edges (kind="expand"/"merge") only if include_phrases.
    """
    if len(raw) == len(norm):
        for i, (a, b) in enumerate(zip(raw, norm)):
            yield i, a, b, "1:1"
        return
    sm = difflib.SequenceMatcher(a=raw, b=norm, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                yield i1 + k, raw[i1 + k], norm[j1 + k], "1:1"
        elif include_phrases and op == "replace":
            if i2 - i1 == 1:
                yield i1, raw[i1], " ".join(norm[j1:j2]), "expand"
            elif j2 - j1 == 1:
                yield i1, " ".join(raw[i1:i2]), norm[j1], "merge"


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_lev(a: str, b: str) -> float:
    return lev(a, b) / max(len(a), len(b), 1)


# ---------------------------------------------------------------- graph

def build_graph(rows: list[dict], include_phrases: bool) -> tuple[nx.DiGraph, Counter]:
    g = nx.DiGraph()
    stats = Counter()
    for r in rows:
        raw = r["inputText"].lower().split()
        norm = r["normalizedText"].lower().split()
        tags = r["tags"]
        stats["rows"] += 1
        stats["rows_equal_len" if len(raw) == len(norm) else "rows_unequal_len"] += 1
        for i, a, b, kind in align(raw, norm, include_phrases):
            tag = tags[i] if i < len(tags) else "?"
            for n in (a, b):
                if n not in g:
                    g.add_node(n, kept=0, as_raw=0, as_norm=0, tags=Counter())
            g.nodes[a]["as_raw"] += 1
            g.nodes[a]["tags"][tag] += 1
            g.nodes[b]["as_norm"] += 1
            stats[f"pairs_{kind}"] += 1
            if a == b:
                g.nodes[a]["kept"] += 1
                continue
            if g.has_edge(a, b):
                g[a][b]["weight"] += 1
                g[a][b]["tags"][tag] += 1
            else:
                g.add_edge(a, b, weight=1, kind=kind, tags=Counter({tag: 1}))
    # derived edge stats
    for a, b, d in g.edges(data=True):
        out_total = sum(x["weight"] for _, _, x in g.out_edges(a, data=True)) + g.nodes[a]["kept"]
        d["share"] = d["weight"] / out_total  # fraction of a's occurrences that went to b
        d["edit"] = round(norm_lev(a, b), 3)
    return g, stats


def canonical(g: nx.DiGraph, nodes) -> str:
    """Most-endorsed form: times chosen as a normalization target (incl. kept as-is)."""
    return max(nodes, key=lambda n: (g.nodes[n]["as_norm"], g.nodes[n]["as_raw"], -len(n)))


def suspect_edges(g: nx.DiGraph, min_share: float, min_weight: int, max_edit: float) -> dict:
    """Edges that plausibly join two different words rather than two spellings of one word."""
    flags = defaultdict(set)
    for a, b, d in g.edges(data=True):
        if d["share"] < min_share and d["weight"] < min_weight:
            flags[(a, b)].add("weak")
        if d["edit"] > max_edit:
            flags[(a, b)].add("far")
    u = g.to_undirected()
    for comp in nx.connected_components(u):
        if len(comp) < 4:
            continue
        sub = u.subgraph(comp)
        for x, y in nx.bridges(sub):
            h = sub.copy()
            h.remove_edge(x, y)
            sides = [len(c) for c in nx.connected_components(h)]
            if min(sides) >= 2:  # bridge between two real clusters, not a leaf
                e = (x, y) if g.has_edge(x, y) else (y, x)
                flags[e].add(f"bridge({min(sides)}|{max(sides)})")
    return flags


# ---------------------------------------------------------------- reporting

def write_csv(path: Path, header: list[str], rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt_tags(c: Counter) -> str:
    return "; ".join(f"{k}:{v}" for k, v in c.most_common())


def export(g: nx.DiGraph, flags: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    u = g.to_undirected()
    comps = sorted(nx.connected_components(u), key=len, reverse=True)
    comp_of = {n: i for i, c in enumerate(comps) for n in c}

    write_csv(out / "nodes.csv",
              ["word", "component", "kept", "as_raw", "as_norm", "in_deg", "out_deg", "tags"],
              [(n, comp_of[n], d["kept"], d["as_raw"], d["as_norm"], g.in_degree(n), g.out_degree(n),
                fmt_tags(d["tags"])) for n, d in g.nodes(data=True)])
    write_csv(out / "edges.csv",
              ["raw", "normalized", "weight", "share", "edit", "kind", "tags", "flags"],
              [(a, b, d["weight"], round(d["share"], 3), d["edit"], d["kind"], fmt_tags(d["tags"]),
                " ".join(sorted(flags.get((a, b), ())))) for a, b, d in g.edges(data=True)])
    write_csv(out / "components.csv",
              ["component", "size", "canonical", "total_weight", "members"],
              [(i, len(c), canonical(g, c), sum(d["weight"] for _, _, d in g.subgraph(c).edges(data=True)),
                " ".join(sorted(c, key=lambda n: -(g.nodes[n]["as_raw"] + g.nodes[n]["as_norm"]))))
               for i, c in enumerate(comps) if len(c) > 1])
    write_csv(out / "suspect_edges.csv",
              ["raw", "normalized", "weight", "share", "edit", "flags"],
              sorted(((a, b, g[a][b]["weight"], round(g[a][b]["share"], 3), g[a][b]["edit"], " ".join(sorted(f)))
                      for (a, b), f in flags.items()), key=lambda r: -r[2]))

    pruned = g.copy()
    pruned.remove_edges_from(flags)
    clusters = sorted((c for c in nx.connected_components(pruned.to_undirected()) if len(c) > 1),
                      key=len, reverse=True)
    write_csv(out / "clusters.csv", ["cluster", "size", "canonical", "members"],
              [(i, len(c), canonical(g, c), " ".join(sorted(c))) for i, c in enumerate(clusters)])

    gx = nx.DiGraph()
    for n, d in g.nodes(data=True):
        gx.add_node(n, kept=d["kept"], as_raw=d["as_raw"], as_norm=d["as_norm"], component=comp_of[n],
                    top_tag=d["tags"].most_common(1)[0][0] if d["tags"] else "")
    for a, b, d in g.edges(data=True):
        gx.add_edge(a, b, weight=d["weight"], share=d["share"], edit=d["edit"],
                    suspect=" ".join(sorted(flags.get((a, b), ()))))
    nx.write_gexf(gx, out / "graph.gexf")
    write_viewer(g, flags, out / "viewer.html")
    return comps, clusters


def write_viewer(g: nx.DiGraph, flags: dict, path: Path):
    """Fill scripts/graph_viewer.html with the graph so it opens straight from disk."""
    data = {
        "nodes": [{"id": n, "kept": d["kept"], "raw": d["as_raw"], "norm": d["as_norm"], "tags": fmt_tags(d["tags"])}
                  for n, d in g.nodes(data=True)],
        "edges": [{"s": a, "t": b, "w": d["weight"], "share": round(d["share"], 3), "edit": d["edit"],
                   "flags": " ".join(sorted(flags.get((a, b), ())))} for a, b, d in g.edges(data=True)],
    }
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = (Path(__file__).parent / "graph_viewer.html").read_text(encoding="utf-8")
    path.write_text(template.replace("/*__GRAPH_DATA__*/null", blob), encoding="utf-8")


def summarize(g, stats, comps, clusters, flags, show: int):
    print(f"rows {stats['rows']}  (equal-length {stats['rows_equal_len']}, unequal {stats['rows_unequal_len']})")
    print(f"aligned pairs: " + ", ".join(f"{k[6:]} {v}" for k, v in stats.items() if k.startswith("pairs_")))
    print(f"nodes {g.number_of_nodes()}  edges {g.number_of_edges()}  "
          f"(isolated, never changed: {sum(1 for n in g if g.degree(n) == 0)})")
    sizes = Counter(len(c) for c in comps if len(c) > 1)
    print("component sizes (size: count):", dict(sorted(sizes.items())[:12]), "...")
    print(f"suspect edges: {len(flags)}  "
          f"({', '.join(f'{k} {v}' for k, v in Counter(f.split('(')[0] for fs in flags.values() for f in fs).items())})")
    print(f"clusters after pruning: {len(clusters)} (largest {len(clusters[0]) if clusters else 0})")
    print(f"\nlargest {show} raw components:")
    for c in comps[:show]:
        if len(c) < 2:
            break
        top = sorted(c, key=lambda n: -(g.nodes[n]["as_raw"] + g.nodes[n]["as_norm"]))[:15]
        print(f"  [{len(c):4}] canon={canonical(g, c):12} {' '.join(top)}{' ...' if len(c) > 15 else ''}")


def ego(g: nx.DiGraph, word: str, radius: int):
    if word not in g:
        print(f"'{word}' not in graph")
        return
    sub = nx.ego_graph(g, word, radius=radius, undirected=True)
    d = g.nodes[word]
    print(f"{word}: kept {d['kept']}, as_raw {d['as_raw']}, as_norm {d['as_norm']}, tags {fmt_tags(d['tags'])}")
    print(f"radius-{radius} neighborhood: {sub.number_of_nodes()} nodes")
    for a, b, e in sorted(sub.edges(data=True), key=lambda x: -x[2]["weight"]):
        print(f"  {a:>14} -> {b:<14} w={e['weight']:<4} share={e['share']:.2f} edit={e['edit']:.2f}")


def path(g: nx.DiGraph, a: str, b: str):
    u = g.to_undirected()
    if a not in u or b not in u or not nx.has_path(u, a, b):
        print(f"no path between '{a}' and '{b}'")
        return
    p = nx.shortest_path(u, a, b)
    print(f"{len(p) - 1} hops: " + " - ".join(p))
    for x, y in zip(p, p[1:]):
        e, arrow = (g[x][y], "->") if g.has_edge(x, y) else (g[y][x], "<-")
        print(f"  {x} {arrow} {y}  w={e['weight']} share={e['share']:.2f} edit={e['edit']:.2f}")


def html(g: nx.DiGraph, word: str, flags: dict, out: Path):
    from pyvis.network import Network
    if word not in g:
        print(f"'{word}' not in graph")
        return
    comp = nx.node_connected_component(g.to_undirected(), word)
    net = Network(height="900px", width="100%", directed=True, cdn_resources="in_line")
    for n in comp:
        d = g.nodes[n]
        net.add_node(n, label=n, size=8 + 4 * (d["as_norm"] + d["as_raw"]) ** 0.5,
                     color="#e4572e" if n == word else "#4c78a8",
                     title=f"kept {d['kept']} | raw {d['as_raw']} | norm {d['as_norm']} | {fmt_tags(d['tags'])}")
    for a, b, e in g.subgraph(comp).edges(data=True):
        net.add_edge(a, b, value=e["weight"], color="#d62728" if (a, b) in flags else "#999999",
                     title=f"w={e['weight']} share={e['share']:.2f} edit={e['edit']:.2f} "
                           f"{' '.join(sorted(flags.get((a, b), ())))}")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"component_{word}.html"
    f.write_text(net.generate_html(), encoding="utf-8")
    print(f"wrote {f} ({len(comp)} nodes; suspect edges in red)")


# ---------------------------------------------------------------- manual decisions + cracking the giant

def load_decisions(path: Path) -> tuple[set, set, set]:
    """decisions.csv: raw,normalized,verdict  with verdict cut|keep; normalized '*' = every edge out of raw."""
    cut, keep, cut_nodes = set(), set(), set()
    if not path.exists():
        return cut, keep, cut_nodes
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raw, norm, v = r["raw"].strip().lower(), r["normalized"].strip().lower(), r["verdict"].strip().lower()
            if not raw or v not in ("cut", "keep"):
                continue
            if norm == "*":
                (cut_nodes if v == "cut" else set()).add(raw)
            else:
                (cut if v == "cut" else keep).add((raw, norm))
    return cut, keep, cut_nodes


def apply_decisions(g: nx.DiGraph, flags: dict, cut: set, keep: set, cut_nodes: set) -> int:
    drop = {e for e in cut if g.has_edge(*e)} | {(a, b) for a in cut_nodes if a in g for b in list(g.successors(a))}
    g.remove_edges_from(drop)
    for e in drop | keep:
        flags.pop(e, None)
    return len(drop)


def crack(g: nx.DiGraph, keep: set, steps: int):
    """Girvan-Newman on the largest component: repeatedly remove the highest-betweenness edge
    (the one most shortest paths squeeze through) and record what splits off. Edges marked
    `keep` in decisions.csv are never removed."""
    u = nx.Graph()
    for a, b in g.edges():
        if u.has_edge(a, b):
            u[a][b]["dirs"].append((a, b))
        else:
            u.add_edge(a, b, dirs=[(a, b)])
    h = u.subgraph(max(nx.connected_components(u), key=len)).copy()
    start = h.number_of_nodes()
    log = []
    for step in range(1, steps + 1):
        if h.number_of_nodes() < 4:
            break
        bc = nx.edge_betweenness_centrality(h)
        cands = [(v, e) for e, v in bc.items() if not all(x in keep for x in h.edges[e]["dirs"])]
        if not cands:
            break
        val, (x, y) = max(cands)
        dirs = h.edges[x, y]["dirs"]
        h.remove_edge(x, y)
        parts = sorted(nx.connected_components(h), key=len, reverse=True)
        split = [(len(p), canonical(g, p), sorted(p, key=lambda n: -(g.nodes[n]["as_raw"] + g.nodes[n]["as_norm"]))[:8])
                 for p in parts[1:]]
        h = h.subgraph(parts[0]).copy()
        log.append({"step": step, "dirs": dirs, "betweenness": round(val, 4), "giant_after": len(parts[0]), "split": split})
    return start, log


def articulation_leverage(g: nx.DiGraph, top: int):
    """Single words whose removal (i.e. splitting them into senses) shrinks the giant the most."""
    u = g.to_undirected()
    giant = u.subgraph(max(nx.connected_components(u), key=len))
    n = giant.number_of_nodes()
    rows = []
    for v in nx.articulation_points(giant):
        h = giant.copy()
        h.remove_node(v)
        rest = max((len(c) for c in nx.connected_components(h)), default=0)
        rows.append((n - 1 - rest, v))
    rows.sort(reverse=True)
    return n, rows[:top]


def report_crack(g: nx.DiGraph, start: int, log: list, n: int, nodes: list, out: Path):
    def e_str(a, b):
        d = g[a][b]
        return f"{a} -> {b} (w={d['weight']}, share={d['share']:.2f}, edit={d['edit']:.2f})"

    print(f"\n== highest-leverage EDGES (giant starts at {start} words) ==")
    print("   remove in this order; each step is the edge the most shortest paths run through")
    for s in log:
        print(f"{s['step']:3}. {' | '.join(e_str(*d) for d in s['dirs'])}  -> giant {s['giant_after']}")
        for size, canon, members in s["split"]:
            print(f"       split off [{size}] {canon}: {' '.join(members)}{' ...' if size > 8 else ''}")
    print(f"\n== highest-leverage WORDS (ambiguous forms; removing one shrinks the giant from {n}) ==")
    for drop, v in nodes:
        outs = ", ".join(f"{b}({g[v][b]['weight']})" for b in sorted(g.successors(v), key=lambda b: -g[v][b]["weight"])[:6])
        print(f"  -{drop:<4} {v:<10} -> {outs or '(only a target)'}")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "crack_edges.csv", ["step", "raw", "normalized", "weight", "share", "edit", "betweenness", "giant_after", "split_off"],
              [(s["step"], a, b, g[a][b]["weight"], round(g[a][b]["share"], 3), g[a][b]["edit"], s["betweenness"], s["giant_after"],
                "; ".join(f"[{sz}] {c}" for sz, c, _ in s["split"])) for s in log for a, b in s["dirs"]])
    write_csv(out / "crack_words.csv", ["word", "giant_shrinks_by", "normalized_to"],
              [(v, drop, " ".join(f"{b}:{g[v][b]['weight']}" for b in g.successors(v))) for drop, v in nodes])
    print(f"\nwrote crack_edges.csv + crack_words.csv to {out}")
    print("record verdicts in decisions.csv (raw,normalized,verdict = cut|keep; normalized '*' cuts every edge out of raw), then rerun")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--include-phrases", action="store_true", help="also add 1->n / n->1 phrase edges")
    ap.add_argument("--min-share", type=float, default=0.15, help="weak edge: share below this ...")
    ap.add_argument("--min-weight", type=int, default=2, help="... and weight below this")
    ap.add_argument("--max-edit", type=float, default=0.67, help="far edge: normalized edit distance above this")
    ap.add_argument("--show", type=int, default=20, help="components to print")
    ap.add_argument("--ego", metavar="WORD")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--path", nargs=2, metavar=("A", "B"))
    ap.add_argument("--html", metavar="WORD", help="write interactive HTML of WORD's component")
    ap.add_argument("--decisions", type=Path, default=HERE / "decisions.csv", help="your manual cut/keep verdicts")
    ap.add_argument("--crack", type=int, nargs="?", const=30, metavar="STEPS",
                    help="rank the edges/words whose disambiguation best breaks up the giant component (default 30 steps)")
    args = ap.parse_args()

    g, stats = build_graph(load_rows(args.data), args.include_phrases)
    flags = suspect_edges(g, args.min_share, args.min_weight, args.max_edit)
    cut, keep, cut_nodes = load_decisions(args.decisions)
    if cut or keep or cut_nodes:
        print(f"decisions.csv: removed {apply_decisions(g, flags, cut, keep, cut_nodes)} edges, kept {len(keep)} protected")
    if args.crack:
        start, log = crack(g, keep, args.crack)
        n, nodes = articulation_leverage(g, 25)
        report_crack(g, start, log, n, nodes, args.out)
        return
    if args.ego or args.path or args.html:
        if args.ego:
            ego(g, args.ego.lower(), args.radius)
        if args.path:
            path(g, args.path[0].lower(), args.path[1].lower())
        if args.html:
            html(g, args.html.lower(), flags, args.out)
        return
    comps, clusters = export(g, flags, args.out)
    summarize(g, stats, comps, clusters, flags, args.show)
    print(f"\nwrote CSVs + graph.gexf to {args.out}")


if __name__ == "__main__":
    main()
