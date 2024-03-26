import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path


# Function to scrape the document
def scrape_document(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        data = {}
        current_main_category = "uncategorized"
        current_category = "uncategorized"
        current_subcategory = "uncategorized"
        for tag in soup.find_all(["h2", "h3", "h4", "table"]):
            if tag.name == "h2":
                current_main_category = tag.text.strip()
                data[current_main_category] = {}
                current_category = "uncategorized"
                current_subcategory = "uncategorized"
            elif tag.name == "h3":
                current_category = (
                    tag.text.strip() if tag.text.strip() else "uncategorized"
                )
                data[current_main_category][current_category] = {}
                current_subcategory = "uncategorized"
            elif tag.name == "h4":
                current_subcategory = (
                    tag.text.strip() if tag.text.strip() else "uncategorized"
                )
                if current_category not in data[current_main_category]:
                    data[current_main_category][current_category] = {}

                data[current_main_category][current_category][current_subcategory] = []
            elif tag.name == "table" and "sortable" in tag.attrs.get("class"):
                weapons = []
                trs = tag.find_all("tr")
                tr_th = [
                    trth for trth in [tr.find_all("th") for tr in trs] if len(trth) > 0
                ]
                if len(tr_th) == 1:
                    headers = [th.text.strip() for th in tr_th[0]]
                    skip = 1
                else:
                    headers = []
                    skip = 2
                    column_row_2 = 0
                    for th in tr_th[0]:
                        if th.has_attr("colspan"):
                            span = int(th["colspan"])
                            th_name = th.text.strip()
                            for i in range(span):
                                headers.append(
                                    f"{th_name}_{tr_th[1][column_row_2].text.strip()}"
                                )
                                column_row_2 += 1
                        else:
                            headers.append(th.text.strip())

                for row in tag.find_all("tr")[skip:]:
                    weapon = {}
                    cells = row.find_all("td")
                    header_index = 0
                    for cell in cells:
                        if cell.has_attr("colspan"):
                            span = int(cell.attrs.get("colspan"))
                            value = cell.text.strip()
                            for j in range(span):
                                weapon[headers[header_index + j]] = value
                            header_index += span
                        else:
                            weapon[headers[header_index]] = cell.text.strip()
                            header_index += 1
                    weapons.append(weapon)

                if current_category not in data[current_main_category]:
                    data[current_main_category][current_category] = {}

                data[current_main_category][current_category][
                    current_subcategory
                ] = weapons

        # remove empty objects for headings without tables
        def remove_empty_dicts(d):
            """
            Recursively remove empty dictionaries from a dictionary.
            """
            for k, v in list(d.items()):
                if isinstance(v, dict):
                    remove_empty_dicts(v)
                    if not v:  # If the nested dictionary becomes empty after removal
                        del d[k]
                elif not v:  # If value is empty (None, "", {}, [])
                    del d[k]
            return d

        data = remove_empty_dicts(data)
        return data
    else:
        print("Failed to fetch the page")


# URL of the document to scrape
url = "http://adragon202.no-ip.org/Shadowrun/index.php/SR5:Gear_Lists:Weapons"

# Scrape the document
scraped_data = scrape_document(url)

# Store the scraped content into JSON format
basepath = Path(__file__).parent
folder_path = basepath / "out"
if scraped_data:
    with open(folder_path / "weapons.json", "w") as f:
        json.dump(scraped_data, f, indent=4)
    print("Scraped data has been stored in 'weapons.json' file.")
else:
    print("No data scraped.")
