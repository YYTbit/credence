import json
import re


def canonical_call(name, arguments):
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        raise ValueError("Reference calls require a function name and JSON arguments")
    return name, json.loads(json.dumps(arguments, allow_nan=False))


def reference_calls(parameters):
    """Read native assistant tool calls or an explicitly prepared reference path."""
    if not isinstance(parameters, dict):
        raise ValueError("ToolBench parameters require a JSON object")
    path = parameters.get("reference_path")
    result = []
    if path is not None:
        if not isinstance(path, list):
            raise ValueError("ToolBench reference_path requires a list of calls")
        for call in path:
            if call["name"] != "Finish":
                result.append(canonical_call(call["name"], call["arguments"]))
    else:
        messages = parameters.get("reference_messages", parameters.get("conversations", []))
        for message in messages:
            if message.get("role", message.get("from")) not in ("assistant", "gpt"):
                continue
            calls = message.get("tool_calls", [])
            if message.get("function_call"):
                calls = [{"function": message["function_call"]}]
            for call in calls:
                function = call.get("function", call)
                if function["name"] != "Finish":
                    result.append(canonical_call(function["name"], function["arguments"]))
            content = message.get("content", message.get("value", ""))
            if not calls and isinstance(content, str):
                match = re.search(r"(?:^|\n)\s*Action:\s*([^\r\n]+)\s*[\r\n]+\s*Action Input:\s*(.*)", content, re.DOTALL)
                if match and match[1].strip() != "Finish":
                    arguments, _ = json.JSONDecoder().raw_decode(match[2].lstrip())
                    result.append(canonical_call(match[1].strip(), arguments))
    if not result:
        raise ValueError("ToolBench training requires a nonempty released reference path")
    return result


def path_completion(steps, reference):
    if not reference:
        raise ValueError("Path completion requires reference calls")
    # Each successful call advances at most one required reference position.
    completed = 0
    for step in steps:
        if completed == len(reference):
            break
        if step["status"] != 0:
            continue
        try:
            actual = canonical_call(step["name"], step["input"])
        except (ValueError, TypeError):
            continue
        if actual == reference[completed]:
            completed += 1
    return completed / len(reference)
