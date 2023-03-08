import pathlib

import yaml

here = pathlib.Path(__file__).parent

# Useful papers

md = "> This file was generated from papis library using papers.py script\n\n"
md += (
    "This is a collection of papers from *ICCV2021*, *ICCV2019*, *CVPR2022*, "
    "*CVPR2021*, *CVPR2020* and  *NeurIPS2022*, *NeurIPS2021*, *NeurIPS2020* "
    "(TODO *CVPR2019* ?) "
    "regarding adversarial attack/defence.\n"
    "\n---------------------------------------------------------------------\n"
)

for paper in here.glob("**/info.yaml"):
    with open(paper, "r") as f:
        info: dict = yaml.safe_load(f)

        md += f"- **{info['title']}**, {info['year']}"
        md += f" [[url]({info['url']})]"
        if code := info.get("code"):
            md += f" [[code]({code})]"
        md += "\n\n"
        md += f"{info['abstract']}\n"
        md += "\n---\n"

with open(here / "papers.md", "w") as f:
    f.write(md)


# Selected papers

selected = {
    "1909.09481v1",
    "1912.09405v4",
    "2108.07969v1",
    "2104.12669v3",
    "2010.04925v4",
    "2011.11164v2",
    "1904.00887v4",
    "1903.09799v3",
}

md = "> This file was generated from papis library using papers.py script\n\n"
md += (
    "This is a collection of papers selected from papers.md.\n"
    "\n---------------------------------------------------------------------\n"
)

for paper in here.glob("**/info.yaml"):
    with open(paper, "r") as f:
        info: dict = yaml.safe_load(f)
        if info["ref"] not in selected:
            continue

        md += f"- **{info['title']}**, {info['year']}"
        md += f" [[url]({info['url']})]"
        if code := info.get("code"):
            md += f" [[code]({code})]"
        md += f"\n>{info['author']}\n"
        md += f"\n{info['abstract']}\n"
        md += "\n---\n"

with open(here / "selected.md", "w") as f:
    f.write(md)
