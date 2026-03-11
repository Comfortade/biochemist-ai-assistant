"""
Molecule Viewer — SMILES Structure Renderer
============================================
Renders 2D chemical structures for metabolites in the knowledge graph
using the PubChem REST API to fetch SMILES and structure images.

Supports:
- Search by compound name or KEGG compound ID
- 2D structure rendering via PubChem
- Molecular formula and basic properties display
"""

import streamlit as st
import requests
from pathlib import Path

st.set_page_config(
    page_title="Molecule Viewer",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Molecule Viewer")
st.caption("Render 2D chemical structures for any biochemical compound.")

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

def get_compound_data(name: str) -> dict:
    """Fetch compound properties from PubChem by name."""
    try:
        url = f"{PUBCHEM_BASE}/compound/name/{name}/property/MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES/JSON"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        props = response.json()["PropertyTable"]["Properties"][0]
        return props
    except Exception as e:
        return {"error": str(e)}


def get_structure_image_url(cid: int) -> str:
    """Get PubChem 2D structure image URL for a compound CID."""
    return f"{PUBCHEM_BASE}/compound/cid/{cid}/PNG?record_type=2d&image_size=400x300"


# Example compounds
st.markdown("**Common metabolites:**")
examples = ["glucose", "pyruvate", "acetyl-CoA", "citrate", "ATP", "NADH", "oxaloacetate"]
cols = st.columns(len(examples))
for i, name in enumerate(examples):
    if cols[i].button(name, key=f"mol_{i}"):
        st.session_state["mol_query"] = name

compound_name = st.text_input(
    "Enter compound name:",
    value=st.session_state.get("mol_query", ""),
    placeholder="e.g. glucose, pyruvate, acetyl-CoA"
)

if compound_name:
    with st.spinner(f"Fetching structure for {compound_name}..."):
        data = get_compound_data(compound_name)

    if "error" in data:
        st.error(f"Compound not found: {data['error']}")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 2D Structure")
            cid = data.get("CID")
            if cid:
                img_url = get_structure_image_url(cid)
                st.image(img_url, caption=compound_name, use_column_width=True)

        with col2:
            st.markdown("### Properties")
            st.markdown(f"**IUPAC Name:** {data.get('IUPACName', 'N/A')}")
            st.markdown(f"**Molecular Formula:** {data.get('MolecularFormula', 'N/A')}")
            st.markdown(f"**Molecular Weight:** {data.get('MolecularWeight', 'N/A')} g/mol")
            st.markdown(f"**PubChem CID:** {data.get('CID', 'N/A')}")
            st.markdown(f"**Canonical SMILES:**")
            st.code(data.get("CanonicalSMILES", "N/A"))