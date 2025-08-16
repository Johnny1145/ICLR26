from types import SimpleNamespace
from typing import List

import numpy as np

from symbolicregression.envs.environment import FunctionEnvironment
from symbolicregression.envs.generators import Node
from symbolicregression.parsers import get_parser

_params = None
_env = None


UNARY_OPS = {
    "abs": "abs",
    "inv": "inv",
    "sqrt": "sqrt",
    "log": "log",
    "ln": "log",
    "exp": "exp",
    "sin": "sin",
    "arcsin": "arcsin",
    "cos": "cos",
    "arccos": "arccos",
    "tan": "tan",
    "arctan": "arctan",
}


def build_params() -> SimpleNamespace:
    global _params
    if _params is None:
        _parser = get_parser()
        _params = _parser.parse_args([])
    return _params


def build_env(params: SimpleNamespace = build_params()) -> FunctionEnvironment:
    global _env
    if _env is None:
        env = FunctionEnvironment(params)
        env.rng = np.random.get_state()
    return env


def parse_expression(expr_str, params: SimpleNamespace = build_params()) -> Node:
    """Parse a mathematical expression string into a Node tree structure."""

    global UNARY_OPS

    def tokenize(expr):
        # 处理幂运算
        expr = expr.replace("pi", "3.14159")
        expr = expr.replace("**2", " pow2 ")
        expr = expr.replace("**3", " pow3 ")

        # 处理括号和运算符
        for op in ["*", "+", "-", "/", "(", ")", ","]:
            expr = expr.replace(op, f" {op} ")

        tokens = [t for t in expr.split() if t]
        return tokens

    def parse_tokens(tokens):
        def parse_power():
            # 处理基本因子和其可能的幂运算
            if not tokens:
                raise ValueError("Unexpected end of expression")

            token = tokens[0]

            # Handle unary minus
            is_negative = False
            if token == "-":
                is_negative = True
                tokens.pop(0)
                if not tokens:
                    raise ValueError("Unexpected end of expression after negative sign")
                token = tokens[0]

            # Handle unary operations
            if token in UNARY_OPS:
                op = tokens.pop(0)
                if not tokens or tokens[0] != "(":
                    raise ValueError(f"Expected ( after {op}")
                tokens.pop(0)  # Remove '('
                arg = parse_expression_internal()
                if not tokens or tokens[0] != ")":
                    raise ValueError(f"Expected ) after {op} argument")
                tokens.pop(0)  # Remove ')'
                node = Node(op, params)
                node.push_child(arg)
                if is_negative:
                    neg_node = Node("mul", params)
                    neg_node.push_child(Node("-1", params))
                    neg_node.push_child(node)
                    return neg_node
                return node

            # Handle numbers
            try:
                num = float(token)
                tokens.pop(0)
                if is_negative:
                    num = -num
                base = Node(str(num), params)
            except ValueError:
                # Handle variables or parenthesized expressions
                if token.startswith("x_"):
                    tokens.pop(0)
                    base = Node(token, params)
                    if is_negative:
                        neg_node = Node("mul", params)
                        neg_node.push_child(Node("-1", params))
                        neg_node.push_child(base)
                        base = neg_node
                elif token == "(":
                    tokens.pop(0)
                    base = parse_expression_internal()
                    if not tokens or tokens.pop(0) != ")":
                        raise ValueError("Expected )")
                    if is_negative:
                        neg_node = Node("mul", params)
                        neg_node.push_child(Node("-1", params))
                        neg_node.push_child(base)
                        base = neg_node
                else:
                    raise ValueError(f"Unexpected token: {token}")

            # Check for power operation
            if tokens and tokens[0] in ["pow2", "pow3"]:
                power_op = tokens.pop(0)
                node = Node(power_op, params)
                node.push_child(base)
                return node
            return base

        def parse_factor():
            return parse_power()

        def parse_term():
            left = parse_factor()
            while tokens and tokens[0] in ["*", "/"]:
                op = tokens.pop(0)
                right = parse_factor()
                if op == "*":
                    node = Node("mul", params)
                    node.push_child(left)
                    node.push_child(right)
                else:  # op == '/'
                    node = Node("mul", params)
                    inv_node = Node("inv", params)
                    inv_node.push_child(right)
                    node.push_child(left)
                    node.push_child(inv_node)
                left = node
            return left

        def parse_expression_internal():
            left = parse_term()
            while tokens and tokens[0] in ["+", "-"]:
                op = tokens.pop(0)
                right = parse_term()
                node = Node("add" if op == "+" else "sub", params)
                node.push_child(left)
                node.push_child(right)
                left = node
            return left

        return parse_expression_internal()

    tokens = tokenize(expr_str)
    return parse_tokens(tokens)
