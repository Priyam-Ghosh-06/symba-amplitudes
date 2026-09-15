r"""Amplitude -> abstract syntax tree -> prefix (Polish) token sequence.

An LALR grammar parses the amplitude; the tree is written out in prefix order,
which removes every bracket while keeping the nesting. Numbers are split into
digits so the vocabulary stays small.
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
    """Parse tree -> flat prefix token list."""

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


def preprocess(eq):
    """Make implicit structure explicit: ``@`` for function application, and
    braces around chained subscripts so the grammar never guesses a nesting."""
    eq = re.sub(r"([\}\)])([\(])", r"\1 @ \2", eq)
    eq = re.sub(r"([a-zA-Z0-9])([\(])", r"\1 @ \2", eq)
    eq = re.sub(r"([a-zA-Z]_\d+)_([+-]?(?:%?\\?[a-zA-Z]+)_[a-zA-Z0-9]+)",
                r"{\1}_{\2}", eq)
    eq = re.sub(r"([a-zA-Z]+)_([a-zA-Z]+)_(\d+)(_\{)",
                r"{{{\1}_{\2}}_{\3}}\4", eq)
    eq = re.sub(r"(?<![A-Za-z\}])([a-zA-Z]+)_([a-zA-Z0-9]+)(_\{)",
                r"{{\1}_{\2}}\3", eq)
    return eq


def amp_to_prefix(equation):
    return _transformer.transform(_parser.parse(preprocess(equation)))


def _subtree_size(node):
    if not isinstance(node, Tree):
        return 1
    return 1 + sum(_subtree_size(c) for c in node.children)


def _largest_addition(tree):
    """The sum node with the biggest subtree: the sum over Feynman diagrams.

    It is not at the root - the amplitude parses as ``prefactor * (sum)`` - so
    it is searched for.
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


def amp_to_segments(equation):
    """The amplitude split at its largest sum into one prefix sequence per term.

    For a multi-diagram amplitude that sum is the sum over Feynman diagrams, so
    each segment is one diagram. The first segment is the rest of the
    expression, with the sum replaced by a ``<diagrams>`` placeholder.
    """
    tree = _parser.parse(preprocess(equation))
    addition = _largest_addition(tree)
    terms = [c for c in addition.children if isinstance(c, Tree)] if addition else []
    if len(terms) < 2:
        return [_transformer.transform(tree)]

    segments = [_transformer.transform(term) for term in terms]
    original = addition.children
    addition.children = [Tree("var", [Token("SYMBOL", "<diagrams>")])]
    try:
        context = _transformer.transform(tree)
    finally:
        addition.children = original
    return [context] + segments
