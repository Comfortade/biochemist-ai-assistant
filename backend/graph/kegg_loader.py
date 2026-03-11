"""
KEGG Pathway Loader
Pathways loaded:
- hsa00010: Glycolysis / Gluconeogenesis
- hsa00020: TCA Cycle (Citrate Cycle)
- hsa00030: Pentose Phosphate Pathway
- hsa00061: Fatty Acid Beta-Oxidation
"""

import requests
import time
import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KEGG_BASE_URL = "https://rest.kegg.jp"

PATHWAYS = {
    "glycolysis": "hsa00010",
    "tca_cycle": "hsa00020",
    "pentose_phosphate": "hsa00030",
    "beta_oxidation": "hsa00061",
}


def fetch_pathway_kgml(pathway_id: str) -> Optional[str]:
    url = f"{KEGG_BASE_URL}/get/{pathway_id}/kgml"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        logger.info(f"Fetched KGML for {pathway_id}")
        return response.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {pathway_id}: {e}")
        return None


def fetch_compound_info(compound_id: str) -> dict:
    url = f"{KEGG_BASE_URL}/get/{compound_id}"
    info = {"id": compound_id, "name": "", "formula": "", "smiles": ""}
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        for line in response.text.splitlines():
            if line.startswith("NAME"):
                info["name"] = line.split(None, 1)[1].strip().rstrip(";")
            elif line.startswith("FORMULA"):
                info["formula"] = line.split(None, 1)[1].strip()
        time.sleep(0.3)  
    except requests.RequestException as e:
        logger.warning(f"Could not fetch compound {compound_id}: {e}")
    return info


def fetch_reaction_info(reaction_id: str) -> dict:
    url = f"{KEGG_BASE_URL}/get/{reaction_id}"
    info = {"id": reaction_id, "name": "", "equation": "", "enzyme": ""}
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        for line in response.text.splitlines():
            if line.startswith("NAME"):
                info["name"] = line.split(None, 1)[1].strip().rstrip(";")
            elif line.startswith("EQUATION"):
                info["equation"] = line.split(None, 1)[1].strip()
            elif line.startswith("ENZYME"):
                info["enzyme"] = line.split(None, 1)[1].strip()
        time.sleep(0.3)
    except requests.RequestException as e:
        logger.warning(f"Could not fetch reaction {reaction_id}: {e}")
    return info


def save_raw_kgml(output_dir: str = "data/raw") -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    for name, pathway_id in PATHWAYS.items():
        kgml = fetch_pathway_kgml(pathway_id)
        if kgml:
            filepath = Path(output_dir) / f"{name}.kgml"
            filepath.write_text(kgml, encoding="utf-8")
            results[name] = kgml
            logger.info(f"Saved {name}.kgml")
        time.sleep(0.5)
    return results


if __name__ == "__main__":
    logger.info("Starting KEGG data fetch...")
    saved = save_raw_kgml()
    logger.info(f"Successfully fetched {len(saved)} pathways: {list(saved.keys())}")