import regex
import unicodedata


def sanitize_json_string(s: str) -> str:

    s = unicodedata.normalize("NFKC", s)
    s = regex.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)
    return str(s)


def extract_json_schema(text: str) -> str:
    pattern_json = r"(?<=```json)((?:.|\n)*)(?=```)"
    pattern = r"\{(?:[^{}]|(?R))*\}|\[(?:[^\[\]]|(?R))*\]"

    match = regex.search(pattern_json, text)
    if match is not None:
        text_raw = str(match.group(1))
        return sanitize_json_string(text_raw)

    match = regex.search(pattern, text, regex.DOTALL)
    if match is not None:
        candidate = str(match.group(0))
        # Only attempt single-quote fix if no double quotes exist
        if '"' not in candidate and "'" in candidate:
            candidate = candidate.replace("'", '"')

        return sanitize_json_string(candidate)

    raise ValueError(f"No json schema could be parsed from input: {text}")
