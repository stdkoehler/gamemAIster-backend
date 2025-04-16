import regex


def extract_json_schema(text: str) -> str:
    pattern_json = r"(?<=```json)((?:.|\n)*)(?=```)"
    pattern = r"\{(?:[^{}]|(?R))*\}|\[(?:[^\[\]]|(?R))*\]"

    match = regex.search(pattern_json, text)
    if match is not None:
        return str(match.group(1))

    match = regex.search(pattern, text, regex.DOTALL)
    if match is not None:
        return str(match.group(0))

    match = regex.search(pattern, text.replace("'", '"'), regex.DOTALL)
    if match is not None:
        return str(match.group(0))

    raise ValueError(f"No json schema could be parsed from input: {text}")
