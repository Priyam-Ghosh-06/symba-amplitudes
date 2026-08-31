r"""S4a - LALR grammar over the amplitude, flattened to prefix notation.

Unchanged in substance from the tokenisation notebook: it parses 594/594
amplitudes with zero failures. What changed is that a failure now raises
instead of writing ``[<UNK>]`` into the record (02 SS6 rule 1).
"""

import re

from lark import Lark, Transformer

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


def parse_amp_record(record):
    record["amp_tokens"] = amp_to_prefix(record["amp_std"])
    return record
