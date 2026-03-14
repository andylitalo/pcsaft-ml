"""Molecule rendering utilities for the Streamlit portal."""

import io

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw


def render_molecule(smiles: str, size: tuple[int, int] = (300, 300)) -> Image.Image:
    """Render a 2D molecular structure from SMILES.

    Parameters
    ----------
    smiles : str
        Molecular SMILES string.
    size : tuple[int, int]
        Image dimensions (width, height) in pixels.

    Returns
    -------
    PIL.Image.Image
        Rendered molecule image. Returns a placeholder image if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return a blank placeholder for invalid SMILES
        return Image.new("RGB", size, color=(240, 240, 240))

    return Draw.MolToImage(mol, size=size)


def smiles_to_image_bytes(smiles: str, size: tuple[int, int] = (300, 300)) -> bytes:
    """Convert SMILES to PNG image bytes for use with st.image().

    Parameters
    ----------
    smiles : str
        Molecular SMILES string.
    size : tuple[int, int]
        Image dimensions (width, height) in pixels.

    Returns
    -------
    bytes
        PNG image data.
    """
    img = render_molecule(smiles, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
