import ast
import json
import os
import subprocess
import sys
import tempfile


ALLOWED_MODULES = {"math", "statistics", "itertools", "functools", "collections",
                   "json", "re", "string", "heapq", "bisect", "fractions", "decimal"}
FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "__import__", "input",
                   "globals", "locals", "vars", "getattr", "setattr", "delattr",
                   "breakpoint", "help", "exit", "quit"}


def tool_functions(code):
    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError) as exc:
        raise ValueError("Tool code requires valid Python syntax") from exc
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        raise ValueError("Tool code requires a function")
    if len({node.name for node in functions}) != len(functions):
        raise ValueError("Tool function names require distinct values")
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
            raise ValueError("Tool modules contain functions and imports")
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Global, ast.Nonlocal)):
            raise ValueError("Unsupported tool statement")
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            raise ValueError("Unsupported tool identifier")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Unsupported tool attribute")
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in ALLOWED_MODULES for alias in node.names):
                raise ValueError("Unsupported tool import")
        if isinstance(node, ast.ImportFrom):
            if node.level or not node.module or node.module.split(".")[0] not in ALLOWED_MODULES:
                raise ValueError("Unsupported tool import")
        if isinstance(node, ast.FunctionDef):
            if node.decorator_list or node.name.startswith("_"):
                raise ValueError("Unsupported tool declaration")
            for default in node.args.defaults + [item for item in node.args.kw_defaults if item is not None]:
                ast.literal_eval(default)
    compile(tree, "tool", "exec")
    return functions


def signature(node):
    return node.name, ast.dump(node.args, include_attributes=False)


def validate_tool(code, tests=(), timeout=3):
    functions = tool_functions(code)
    checks = json.dumps(list(tests))
    runner = (
        "import json, resource\n"
        "resource.setrlimit(resource.RLIMIT_CPU, (2, 2))\n"
        "resource.setrlimit(resource.RLIMIT_AS, (268435456, 268435456))\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))\n"
        "namespace = {}\n"
        f"exec(compile({code!r}, 'tool', 'exec'), namespace)\n"
        f"tests = json.loads({checks!r})\n"
        "for test in tests:\n"
        "    result = namespace[test['function']](*test.get('args', []), **test.get('kwargs', {}))\n"
        "    if 'expected' in test and result != test['expected']:\n"
        "        raise ValueError('Tool test output differs')\n"
    )
    with tempfile.TemporaryDirectory(prefix="tool-check-") as workspace:
        result = subprocess.run([sys.executable, "-I", "-S", "-c", runner],
                                cwd=workspace, env={"PATH": os.defpath},
                                capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise ValueError("Tool execution validation failed")
    return [node.name for node in functions]


def merge_tools(left, right, left_credit, right_credit):
    first, second = tool_functions(left), tool_functions(right)
    if {signature(node) for node in first} == {signature(node) for node in second}:
        code = left if left_credit >= right_credit else right
    else:
        if {node.name for node in first} & {node.name for node in second}:
            raise ValueError("Conflicting callable names require distinct tool targets")
        code = left + "\n\n" + right
    validate_tool(code)
    return code


def split_tools(code):
    functions = tool_functions(code)
    if len(functions) < 2:
        raise ValueError("Tool splitting requires multiple functions")
    names = {node.name for node in functions}
    for node in functions:
        if any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
               and call.func.id in names - {node.name} for call in ast.walk(node)):
            raise ValueError("Tool functions share callable dependencies")
    tree = ast.parse(code)
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    chunks = []
    for node in functions:
        chunk = ast.unparse(ast.Module(body=imports + [node], type_ignores=[]))
        validate_tool(chunk)
        chunks.append(chunk)
    return chunks
