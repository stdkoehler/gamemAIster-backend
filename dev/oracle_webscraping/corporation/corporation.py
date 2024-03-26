# We we have some missing info on corps in corporations.json.
# We can scrape https://shadowrun.fandom.com/wiki/Regulus_Joint_Industries
# div id="mw-content-text"
from bs4 import BeautifulSoup
import requests
import json
import re
from pathlib import Path
from dataclasses import asdict

from src.llmclient.types import LLMConfig

# basepath = Path(__file__).parent
# file = basepath / "corporations_raw.json"
# corporations = []
# with open(file, "r", encoding="utf-8") as fin:
#     corps = json.load(fin)
#     for corp in corps:
#         corp["active_until"] = "today"
#         corporations.append(corp)
# with open(basepath / "corporations_raw_active.json", "w", encoding="utf-8") as fout:
#     json.dump(corporations, fout, indent=4)
# print()


def scrape_summary(corp_name: str) -> str:
    corp_name = corp_name.replace(" ", "_")

    html_content = requests.get(f"https://shadowrun.fandom.com/wiki/{corp_name}").text

    soup = BeautifulSoup(html_content, "html.parser")

    # Find the div with id="mw-content-text"
    mw_content_text = soup.find("div", {"class": "mw-parser-output"})

    # Find all paragraphs within mw-content-text until the References section
    paragraphs = mw_content_text.find_all(["p", "h2", "ul"], recursive=False)
    text = ""
    for element in paragraphs:
        if element.name == "h2" and element.find("span", {"id": "References"}):
            break
        elif element.name == "p":
            text += element.get_text() + "\n"

    return text


def completion(prompt):
    headers = {"Content-Type": "application/json"}

    data = asdict(LLMConfig())
    data["top_k"] = 40
    data["repetition_penalty"] = 1
    data["prompt"] = prompt
    data["stream"] = False

    response = requests.post(
        "http://127.0.0.1:5000/v1/completions",
        headers=headers,
        json=data,
        verify=False,
        stream=False,
        timeout=60,
    )
    return json.loads(response.text)["choices"][0]["text"]


basepath = Path(__file__).parent
file = basepath / "corporations_raw.json"
with open(file, "r", encoding="utf-8") as fin:
    corps = json.load(fin)

for i, corp in enumerate(corps):
    fileout = basepath / "out" / f"{i}.json"
    if fileout.exists():
        continue

    try:
        summary = scrape_summary(corp["name"])
    except AttributeError:
        print(f"No web ressource found for {corp['name']}")
        continue

    prompt = f"""[INST] You are a summarization AI assistant that generates and corrects json data based on textual input. The output is provided by the AI assistant
    as plain text without any markdown styling in the following format:
{{
  "name": "corporation",
  "description": "short description max. 256 words",
  "class": "AAA or what?",
  "country": "country of headquarter",
  "city": "city of headquarter",
  "active_until": "today when still active or year of dissolving" 
}}

### EXAMPLE
Banana Industries was founded 1985 in Karlsruhe, Germany. With an initial budget of 1 Dollar, it was struggling to survive the first 10 years but due to the impact of their product Glixmug which is a handheld console they soon became a billion dollar company. Due to political intrigue the company was dissolved 2022.

{{
  "name": "Banana Industries",
  "description": "Unknown",
  "class": "AAA",
  "country": "Unknown",
  "city": "Unknown",
  "active_until": "today"
}}

Corrected:
{{
  "name": "Banana Industries",
  "description": "A billion dollar company which had her breakthrough with the handheld console Glixmug",
  "class": "AAA",
  "country": "Germany",
  "city": "Karlsruhe",
  "active_until": "2022"
}}
### EXAMPLE END

    
You've got the following information:
{summary}

Complete / correct the following json:

{json.dumps(corp, indent=2)}

The description is a summary of the provided information. Keep it short and concise!
Do not leave out the active_until field. If you're unsure just set it to "today".

[/INST]Corrected:
"""
    # print(prompt)

    try:
        c = completion(prompt)
        print(c)
        pattern = r'{\s*"[^"]*"\s*:\s*("[^"]*"|true|false|\d+(?:\.\d+)?)(?:\s*,\s*"[^"]*"\s*:\s*("[^"]*"|true|false|\d+(?:\.\d+)?))*\s*}'
        try:
            json_data = re.search(pattern, c, re.DOTALL).group(0)
        except AttributeError:
            json_data = re.search(pattern, c.replace("'", '"'), re.DOTALL).group(0)
        corrected = json.loads(json_data)
        with open(fileout, "w", encoding="utf-8") as fout:
            json.dump(corrected, fout)
    except:
        print(c)

        raise ValueError()

corporations = []
folder_path = basepath / "out"
for file in folder_path.glob("*"):
    if file.is_file():
        with open(file, "r", encoding="utf-8") as fin:
            corporations.append(json.load(fin))

corporations = sorted(corporations, key=lambda x: x["name"])
with open(basepath / "corporations.json", "w", encoding="utf-8") as fout:
    json.dump(corporations, fout, indent=4)
