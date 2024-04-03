import regex

text = """
,daöäs,daäösd

```json
[
   {
       "name": "Skye",
       "type": "Person",
       "summary": "Bayonette's fixer who provides her with a job opportunity."
   },
   {
       "name": "The Client",
       "type": "Person",
       "summary": "A metahuman rights activist who hires Bayonette to shut down a biological weapon targeting metahumans."
   },
   {
       "name": "Shutdown Target",
       "type": "Location",
       "summary": "A research facility on the outskirts of Denver that develops a biological weapon targeting metahumans. Known as the 'Twisted Tetrahedron Lab.'"
   },
   {
       "name": "Metamorphosis",
       "type": "Item",
       "summary": "The codename for the biological weapon that Bayonette needs to obtain information on."
   },
   {
       "name": "Replacement Project",
       "type": "Organization",
       "summary": "A shadowy group behind the development of the 'Metamorphosis' biological weapon. Specialized in genetic manipulation and bioweapons."
   }
]
```
"""

pattern_json = r"(?<=```json)((?:.|\n)*)(?=```)"
pattern = r"\{(?:[^{}]|(?R))*\}|\[(?:[^\[\]]|(?R))*\]"

match = regex.search(pattern_json, text)
if match is not None:
    print(match.group(1))

match = regex.search(pattern, text, regex.DOTALL)
if match is not None:
    print(match.group(0))

match = re.search(pattern, text.replace("'", '"'), regex.DOTALL)
if match is not None:
    print(match.group(0))

raise ValueError(f"No json schema could be parsed from input: {text}")
