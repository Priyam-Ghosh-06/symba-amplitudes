r"""S4a - LALR grammar over the amplitude, flattened to prefix notation.

Unchanged in substance from the tokenisation notebook: it parses 594/594
amplitudes with zero failures. What changed is that a failure now raises
instead of writing ``[<UNK>]`` into the record (02 SS6 rule 1).
"""

import re

from lark import Lark, Token, Transformer, Tree

QFT_GRAMMAR = r"""
    ?start: expr

    ?expr:  term  (ADD_OP term)*
    ?term:  factor (MUL_OP factor)*
    ?factor: "-" factor  -> neg_node
           | "+" factor  -> pos_node
           | power

    ?power:     app ("^" power_arg)?
    ?power_arg: app
              | "(*)"    -> conj_node

    ?app:  atom (APP_OP atom)*

    ?atom: base ("_" sub_arg)?
    ?sub_arg: base

    ?comma_list: expr (COMMA expr)*

    ?base: SYMBOL            -> var
         | NUMBER            -> num
         | "(" expr ")"      -> drop_group
         | "{" comma_list "}" -> drop_group

    ADD_OP: "+" | "-"
    MUL_OP: "*" | "/"
    APP_OP: "@"
    COMMA:  ","

    SYMBOL: /%?\\?[a-zA-Z][a-zA-Z0-9]*/
    NUMBER: /\d+(\.\d+)?/

    %import common.WS
    %ignore WS
"""


class QFTPrefixTransformer(Transformer):
    """Parse tree to a flat prefix token list."""

    def _binary_flatten(self, args):
        if len(args) == 1:
            return args[0]
        res = [str(args[1])] + args[0] + args[2]
        for i in range(3, len(args), 2):
            res = [str(args[i])] + res + args[i + 1]
        return res

    def expr(self, args):       return self._binary_flatten(args)
    def term(self, args):       return self._binary_flatten(args)
    def app(self, args):        return self._binary_flatten(args)
    def comma_list(self, args): return self._binary_flatten(args)

    def power(self, args):
        return args[0] if len(args) == 1 else ["^"] + args[0] + args[1]

    def atom(self, args):
        return args[0] if len(args) == 1 else ["_"] + args[0] + args[1]

    def neg_node(self, args):   return ["NEG"] + args[0]
    def pos_node(self, args):   return ["UADD"] + args[0]
    def conj_node(self, args):  return ["CONJ"]
    def drop_group(self, args): return args[0]
    def var(self, args):        return [str(args[0])]
    def num(self, args):        return list(str(args[0]))


_parser = Lark(QFT_GRAMMAR, parser="lalr")
_transformer = QFTPrefixTransformer()


def preprocess(eq: str) -> str:
    """Make implicit structure explicit before parsing.

    Inserts ``@`` for function application and re-brackets chained subscripts
    so the grammar never has to guess a nesting.
    """
    eq = re.sub(r"([\}\)])([\(])", r"\1 @ \2", eq)
    eq = re.sub(r"([a-zA-Z0-9])([\(])", r"\1 @ \2", eq)
    eq = re.sub(r"([a-zA-Z]_\d+)_([+-]?(?:%?\\?[a-zA-Z]+)_[a-zA-Z0-9]+)",
                r"{\1}_{\2}", eq)
    eq = re.sub(r"([a-zA-Z]+)_([a-zA-Z]+)_(\d+)(_\{)",
                r"{{{\1}_{\2}}_{\3}}\4", eq)
    eq = re.sub(r"(?<![A-Za-z\}])([a-zA-Z]+)_([a-zA-Z0-9]+)(_\{)",
                r"{{\1}_{\2}}\3", eq)
    return eq


def amp_to_prefix(equation: str) -> list:
    """preprocess -> LALR parse -> prefix token list. Raises on a parse failure."""
    return _transformer.transform(_parser.parse(preprocess(equation)))


def _subtree_size(node) -> int:
    if not isinstance(node, Tree):
        return 1
    return 1 + sum(_subtree_size(c) for c in node.children)


def _largest_addition(tree):
    """The addition node with the biggest subtree, or None if there is none.

    The amplitude's sum over diagrams is not at the root - a QCD amplitude
    parses as ``/ * * * ... + + + ...``, with the sum nested under the overall
    prefactor - so the diagram structure has to be searched for rather than
    assumed.
    """
    best, stack = None, [tree]
    while stack:
        node = stack.pop()
        if not isinstance(node, Tree):
            continue
        if node.data == "expr" and len(node.children) >= 3:
            if best is None or _subtree_size(node) > _subtree_size(best):
                best = node
        stack.extend(c for c in node.children if isinstance(c, Tree))
    return best


def amp_to_segments(equation: str) -> list:
    """Split an amplitude into per-diagram prefix segments.

    The amplitude is a sum over Feynman diagrams. Encoding each diagram
    separately with shared weights turns the encoder's attention cost from
    ``(sum_d L_d)^2`` into ``sum_d L_d^2`` and injects the permutation
    invariance over diagrams that the physics already has (01 SS4.2 option 2).

    Measured on this corpus: QCD encoder attention work falls 10.7x and the
    longest segment goes from 2859 tokens to 239. QED gains 1.9x - less,
    because its amplitudes were short to begin with.

    Returns a list of token lists. The first segment carries the surrounding
    prefactor context; the rest are the individual diagrams.
    """
    tree = _parser.parse(preprocess(equation))
    addition = _largest_addition(tree)
    if addition is None:
        return [_transformer.transform(tree)]

    terms = [c for c in addition.children if isinstance(c, Tree)]
    if len(terms) < 2:
        return [_transformer.transform(tree)]

    # Replace the addition with a placeholder so the context segment keeps the
    # prefactor and the operator skeleton without duplicating the diagrams.
    segments = []
    for term in terms:
        segments.append(_transformer.transform(term))

    original = addition.children
    addition.children = [Tree("var", [Token("SYMBOL", "<diagrams>")])]
    try:
        context = _transformer.transform(tree)
    finally:
        addition.children = original

    return [context] + segments


def parse_amp_record(record, segment: bool = True):
    record["amp_tokens"] = amp_to_prefix(record["amp_std"])
    record["amp_segments"] = (amp_to_segments(record["amp_std"]) if segment
                              else [record["amp_tokens"]])
    return record
