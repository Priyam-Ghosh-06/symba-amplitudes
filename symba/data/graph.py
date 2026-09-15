"""Feynman diagram -> token sequence for the graph encoder.

The vertex string lists, for each vertex, its external fields and its off-shell
(internal) lines. An internal line appears at both of its vertices, so the two
ends are joined into one propagator edge by particle type.
"""

import re
from collections import defaultdict

_VERTEX = re.compile(r"(V_\d+)\s*:\s*(.*?)(?=(?:V_\d+\s*:|$))")
_OFFSHELL = re.compile(r"(AntiPart\s+)?OffShell\s+(\S+?)\s*\((V_\d+)\)")
_FIELD = re.compile(r"^(AntiPart\s+)?([A-Za-z][A-Za-z0-9]*)\s*\(\s*(X_\d+)\s*\)$")
_LEG = re.compile(r"^([A-Za-z][A-Za-z0-9]*)(?:_.*)?\(X\)(\^\(\*\))?$")


def _vertex_index(v_id):
    return int(v_id.split("_")[1])


def parse_legs(interaction):
    """``(in_legs, out_legs)``; each leg is ``(particle, is_antiparticle)``."""
    if " to " not in interaction:
        raise ValueError(f"no ' to ' in interaction: {interaction!r}")

    def side(text):
        legs, anti = [], False
        for token in text.split():
            if token == "AntiPart":
                anti = True
                continue
            match = _LEG.match(token)
            if not match:
                raise ValueError(f"unparseable leg {token!r}")
            legs.append((match.group(1), anti))
            anti = False
        return legs

    incoming, outgoing = interaction.split(" to ", 1)
    return side(incoming), side(outgoing)


class FeynmanGraph:
    """External legs, vertices with their fields, and propagator edges."""

    def __init__(self, interaction, vertices):
        self.in_legs, self.out_legs = parse_legs(interaction)
        self.nodes = {}                 # V_i -> [(particle, is_antiparticle, X_j)]
        self.edges = []                 # (V_i, V_j, particle), i < j

        ends = defaultdict(list)        # particle -> vertices its internal line touches
        found = sorted(_VERTEX.findall(vertices), key=lambda m: _vertex_index(m[0]))
        if not found:
            raise ValueError(f"no vertices in {vertices!r}")
        for v_id, body in found:
            fields = []
            for entry in (e.strip() for e in body.split(",")):
                if not entry:
                    continue
                offshell = _OFFSHELL.match(entry)
                if offshell:
                    ends[offshell.group(2)].append(v_id)
                    continue
                field = _FIELD.match(entry)
                if not field:
                    raise ValueError(f"unparseable vertex field {entry!r}")
                fields.append((field.group(2), bool(field.group(1)), field.group(3)))
            self.nodes[v_id] = fields

        for particle, vs in sorted(ends.items()):
            if len(vs) != 2 or vs[0] == vs[1]:
                raise ValueError(f"propagator {particle!r} joins {vs}, expected two vertices")
            a, b = sorted(vs, key=_vertex_index)
            self.edges.append((a, b, particle))
        self.edges.sort()

    def to_tokens(self):
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
