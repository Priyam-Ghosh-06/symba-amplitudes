"""S4b - Feynman diagram topology as a genuine edge list.

The previous builder matched ``OffShell A(V_1)`` inside vertex ``V_1`` and
emitted the self-loop ``(V_1, V_1, A)``, so the internal line joining the two
vertices - the thing that determines the denominator channel - was never
represented at all (02 SS2.4). Here off-shell stubs are matched *across*
vertices by particle type, producing a real edge with distinct endpoints.
"""

import re
from collections import defaultdict

_VERTEX_RE = re.compile(r"(V_\d+)\s*:\s*(.*?)(?=(?:V_\d+\s*:|$))")
_OFFSHELL_RE = re.compile(r"(AntiPart\s+)?OffShell\s+(\S+?)\s*\((V_\d+)\)")
_FIELD_RE = re.compile(r"^(AntiPart\s+)?([A-Za-z][A-Za-z0-9]*)\s*\(\s*(X_\d+)\s*\)$")
_LEG_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*)(?:_.*)?\(X\)(\^\(\*\))?$")


class GraphError(ValueError):
    pass


def parse_legs(interaction_clean: str):
    """Split the interaction into ``(in_legs, out_legs)``.

    Each leg is ``(particle, is_antipart)``. ``AntiPart`` is a separate word in
    the source and binds to the token that follows it.
    """
    if " to " not in interaction_clean:
        raise GraphError(f"no ' to ' in interaction: {interaction_clean!r}")

    in_str, out_str = interaction_clean.split(" to ", 1)

    def side(text):
        legs, anti = [], False
        for token in text.split():
            if token == "AntiPart":
                anti = True
                continue
            match = _LEG_RE.match(token)
            if not match:
                raise GraphError(f"unparseable leg {token!r}")
            legs.append((match.group(1), anti))
            anti = False
        return legs

    return side(in_str), side(out_str)


class FeynmanGraph:
    """Vertices, external legs, and resolved internal propagators.

    Vertices are stored in sorted id order so that two identical diagrams
    written in different orders serialise identically (02 SS2.4).
    """

    def __init__(self, interaction_clean: str, vertices_clean: str):
        self.in_legs, self.out_legs = parse_legs(interaction_clean)
        self.nodes = {}         # V_i -> [(particle, is_antipart, x_index)]
        self.stubs = {}         # V_i -> [(particle, is_antipart)] off-shell fields
        self.edges = []         # (V_i, V_j, particle), i < j
        self._build(vertices_clean)

    def _build(self, vertices_clean: str):
        matches = _VERTEX_RE.findall(vertices_clean)
        if not matches:
            raise GraphError(f"no vertices in {vertices_clean!r}")

        for v_id, body in matches:
            fields, stubs = [], []
            for raw in body.split(","):
                entry = raw.strip().rstrip(",").strip()
                if not entry:
                    continue

                offshell = _OFFSHELL_RE.match(entry)
                if offshell:
                    stubs.append((offshell.group(2), bool(offshell.group(1))))
                    continue

                field = _FIELD_RE.match(entry)
                if not field:
                    raise GraphError(f"unparseable vertex field {entry!r}")
                fields.append((field.group(2), bool(field.group(1)),
                               field.group(3)))

            self.nodes[v_id] = fields
            self.stubs[v_id] = stubs

        self.nodes = dict(sorted(self.nodes.items(), key=_vertex_sort_key))
        self.stubs = dict(sorted(self.stubs.items(), key=_vertex_sort_key))
        self._resolve_propagators()

    def _resolve_propagators(self):
        """Join off-shell stubs of the same particle type across vertices.

        A fermion propagator appears as ``OffShell e(V_1)`` at one end and
        ``AntiPart OffShell e(V_0)`` at the other; both name the same particle,
        so the type is what pairs them. The anti flag records orientation and
        is not part of the pairing key.
        """
        by_particle = defaultdict(list)
        for v_id, stubs in self.stubs.items():
            for particle, _anti in stubs:
                by_particle[particle].append(v_id)

        for particle, vertices in sorted(by_particle.items()):
            if len(vertices) != 2:
                raise GraphError(
                    f"propagator {particle!r} touches {len(vertices)} vertices "
                    f"({vertices}); expected exactly 2")
            a, b = sorted(vertices, key=_vertex_index)
            if a == b:
                raise GraphError(f"propagator {particle!r} is a self-loop at {a}")
            self.edges.append((a, b, particle))

        self.edges.sort()

    def is_connected(self) -> bool:
        """Gate G6: every vertex reachable from any other through propagators."""
        if len(self.nodes) <= 1:
            return True

        adjacency = defaultdict(set)
        for a, b, _p in self.edges:
            adjacency[a].add(b)
            adjacency[b].add(a)

        start = next(iter(self.nodes))
        seen, stack = {start}, [start]
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return seen == set(self.nodes)

    def to_tokens(self) -> list:
        """Canonical 1-D token sequence for the graph encoder.

        Structural markers are STRUCT-typed; particle names and X indices carry
        the content. Nothing here depends on dictionary insertion order.
        """
        seq = ["<graph>", "<in>"]
        for particle, anti in self.in_legs:
            seq += (["<anti>"] if anti else []) + [particle]
        seq.append("<out>")
        for particle, anti in self.out_legs:
            seq += (["<anti>"] if anti else []) + [particle]

        for v_id, fields in self.nodes.items():
            seq += ["<vtx>", v_id]
            for particle, anti, x_index in fields:
                seq += (["<anti>"] if anti else []) + [particle, x_index]

        for a, b, particle in self.edges:
            seq += ["<prop>", particle, a, b]

        seq.append("</graph>")
        return seq


def _vertex_index(v_id: str) -> int:
    return int(v_id.split("_")[1])


def _vertex_sort_key(item):
    return _vertex_index(item[0])


def build_graph_record(record):
    """Attach ``graph`` and ``graph_tokens`` to a Record. Raises on failure."""
    graph = FeynmanGraph(record["interaction_clean"], record["vertices_clean"])
    if not graph.is_connected():
        raise GraphError(
            f"{record.source_file}:{record.line_no} graph is disconnected")
    record["graph"] = graph
    record["graph_tokens"] = graph.to_tokens()
    return record
