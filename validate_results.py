import argparse
import ast
import json
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Tuple

from logsetting import logger

TARGET = 24.0
EPS = 1e-6


def _round_key(vals: List[float], ndigits: int = 6) -> Tuple[float, ...]:
    return tuple(sorted(round(v, ndigits) for v in vals))


@lru_cache(maxsize=200_000)
def _best_residual(key: Tuple[float, ...]) -> float:
    vals = list(key)
    n = len(vals)

    if n == 0:
        return 1e9
    if n == 1:
        v = vals[0]
        if not math.isfinite(v):
            return 1e9
        return abs(v - TARGET)

    best = 1e9
    for i in range(n):
        for j in range(i + 1, n):
            a, b = vals[i], vals[j]
            rest = [vals[k] for k in range(n) if k != i and k != j]

            candidates = [
                a + b,
                a * b,
                a - b,
                b - a,
            ]
            if abs(b) > EPS:
                candidates.append(a / b)
            if abs(a) > EPS:
                candidates.append(b / a)

            for c in candidates:
                if not math.isfinite(c):
                    continue
                if abs(c) > 1e6:
                    continue

                nxt = rest + [c]
                r = _best_residual(_round_key(nxt))
                if r < best:
                    best = r
                    if best <= 0.0 + 1e-9:
                        return 0.0

    return best


def game24_score(state: Dict) -> float:
    try:
        if state.get("invalid_move"):
            return 1e9
        items = state.get("items", [])
        if not isinstance(items, list) or len(items) == 0:
            return 1e9

        vals = []
        mag_pen = 0.0
        for it in items:
            v = float(it.get("value", 0.0))
            if not math.isfinite(v):
                return 1e9
            vals.append(v)
            mag_pen += max(0.0, abs(v) - 100.0)

        residual = _best_residual(_round_key(vals))
        depth_tiebreak = 0.05 * (len(items) - 1)

        return float(residual) + depth_tiebreak + 0.001 * float(mag_pen)
    except Exception:
        return 1e9


def _parse_numbers_from_original(original: str) -> List[float]:
    if original is None:
        return []
    s = str(original).strip()
    try:
        if s.startswith("[") and s.endswith("]"):
            arr = json.loads(s)
            return [float(x) for x in arr]
    except Exception:
        pass
    s = s.replace(",", " ")
    s = s.replace("[", " ").replace("]", " ")
    parts = [p for p in s.split() if p.strip() != ""]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            pass
    return out


def _extract_expression(text: str):
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        lowered = line.lower()
        if lowered.startswith("output"):
            if ":" in line:
                expr_line = line.split(":", 1)[1].strip()
            else:
                expr_line = line[len("output") :].strip()
            if expr_line == "" and idx + 1 < len(lines):
                expr_line = lines[idx + 1].strip()
            return _strip_expression(expr_line)
    for line in reversed(lines):
        if "=" in line:
            return _strip_expression(line)
    return None


def _strip_expression(line: str) -> str:
    expr_line = line.strip().rstrip(".")
    if "=" in expr_line:
        left, right = (part.strip() for part in expr_line.split("=", 1))
        if _looks_like_expression(right) and not _looks_like_expression(left):
            expr_line = right
        else:
            expr_line = left
    return expr_line


def _looks_like_expression(text: str) -> bool:
    return re.search(r"[+\-*/()]", text) is not None


def _extract_numbers(expression: str) -> List[float]:
    tokens = re.findall(r"\d+(?:\.\d+)?", expression)
    return [float(token) for token in tokens]


def _normalize_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _numbers_match(expression: str, original_counter: Counter) -> bool:
    expr_numbers = _extract_numbers(expression)
    expr_counter = Counter(_normalize_number(num) for num in expr_numbers)
    return expr_counter == original_counter


def _safe_eval(expression: str):
    try:
        node = ast.parse(expression, mode="eval")
    except Exception:
        return None

    def _eval_node(ast_node: ast.AST) -> float:
        if isinstance(ast_node, ast.Expression):
            return _eval_node(ast_node.body)
        if isinstance(ast_node, ast.Constant) and isinstance(ast_node.value, (int, float)):
            return float(ast_node.value)
        if isinstance(ast_node, ast.UnaryOp) and isinstance(ast_node.op, (ast.UAdd, ast.USub)):
            value = _eval_node(ast_node.operand)
            return value if isinstance(ast_node.op, ast.UAdd) else -value
        if isinstance(ast_node, ast.BinOp) and isinstance(
            ast_node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left = _eval_node(ast_node.left)
            right = _eval_node(ast_node.right)
            if isinstance(ast_node.op, ast.Add):
                return left + right
            if isinstance(ast_node.op, ast.Sub):
                return left - right
            if isinstance(ast_node.op, ast.Mult):
                return left * right
            if isinstance(ast_node.op, ast.Div):
                return left / right
        raise ValueError("Unsupported expression")

    try:
        return float(_eval_node(node))
    except Exception:
        return None


def _validate_game24_answer(original: str, result: str) -> bool:
    expression = _extract_expression(result)
    if not expression:
        return False
    original_numbers = _parse_numbers_from_original(original)
    original_counter = Counter(_normalize_number(num) for num in original_numbers)
    if not _numbers_match(expression, original_counter):
        return False
    value = _safe_eval(expression)
    if value is None:
        return False
    state = {"items": [{"value": float(value)}]}
    return game24_score(state) <= EPS


parser = argparse.ArgumentParser()
parser.add_argument(
    "--task_name",
    type=str,
    default="gameof24",
    choices=["gameof24", "checkmate", "wordsorting"],
)
parser.add_argument("--test_path", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    task = args.task_name
    test_path = args.test_path

    benchmark_path_dict = {
        "gameof24": "benchmarks/gameof24.jsonl",
        "checkmate": "benchmarks/CheckmateInOne.jsonl",
        "wordsorting": "benchmarks/word_sorting.jsonl",
    }
    test_path_dict = {
        "gameof24": "test_results/BoT_gameof24.jsonl",
        "checkmate": "test_results/BoT_checkmate.jsonl",
        "wordsorting": "test_results/BoT_wordsorting.jsonl",
    }
    if not test_path:
        test_path = test_path_dict[task]

    correct = 0
    total = 0

    if task == "gameof24":
        for line in open(test_path, "r", encoding="utf-8"):
            payload = json.loads(line)
            original = payload.get("input", "")
            result = payload.get("result", "")
            total += 1
            if _validate_game24_answer(original, result):
                correct += 1
        accuracy = correct / total if total else 0.0
        logger.info(
            "Total number:%s,Correct number:%s,Accuracy:%s",
            total,
            correct,
            accuracy,
        )
        raise SystemExit(0)

    benchmark_path = benchmark_path_dict[task]
    truth = []
    for line in open(benchmark_path, "r", encoding="utf-8"):
        answer = json.loads(line)["target"]
        truth.append(answer)

    for idx, line in enumerate(open(test_path, "r", encoding="utf-8")):
        payload = json.loads(line)
        result = str(payload.get("result", "")).splitlines()[0].strip()
        total += 1
        if idx < len(truth) and truth[idx] == result:
            correct += 1

    accuracy = correct / total if total else 0.0
    logger.info(
        "Total number:%s,Correct number:%s,Accuracy:%s",
        total,
        correct,
        accuracy,
    )
