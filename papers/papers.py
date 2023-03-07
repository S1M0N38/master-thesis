import pathlib

import yaml

here = pathlib.Path(__file__).parent
md = "> This file was generated from papis library using papers.py script\n\n"
md += (
    "# Papers \n"
    "This is a collection of papers from *ICCV2021*, *ICCV2019*, *CVPR2022*, "
    "*CVPR2021*, *CVPR2020* and (TODO *CVPR2019 ?*, *NeurIPS2022*, "
    "*NeurIPS2021*, *NeurIPS2020*)\n"
    "regarding adversarial attack/defence.\n"
    "\n---\n"
)

for paper in here.glob("**/info.yaml"):
    with open(paper, "r") as f:
        info: dict = yaml.safe_load(f)

        md += f"- **{info['title']}**, {info['year']}"
        md += f" [[url]({info['url']})]"
        if code := info.get("code"):
            md += f" [[code]({code})]"
        md += "\n\n"
        try:
            md += f"{info['notes']}\n"
        except KeyError as e:
            print(info["title"])
            raise e

        md += "\n---\n"


with open(here / "papers.md", "w") as f:
    f.write(md)
