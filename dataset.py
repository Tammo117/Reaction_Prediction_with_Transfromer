import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from IPython.display import display
from typing import List, Tuple

class ReadUsptoDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        assert "reactions" in list(self.df.columns), f"Expected ['reactions'] in columns, got {list(self.df.columns)}"

        self.df[["reactants", "products"]] = self.df["reactions"].str.split(">>", expand=True)
        self.process_canonical_smiles()

        return self.df
    
    def process_canonical_smiles(self):
        self.df['reactants'] = self.df['reactants'].apply(self.check_smile_validity_and_canonicatize)
        self.df['products'] = self.df['products'].apply(self.check_smile_validity_and_canonicatize)
        self.df.dropna(subset=['reactants', 'products'], inplace=True)
        return self.df
        
    def check_smile_validity_and_canonicatize(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
        
    def visualize_reactions(self, num_reactions=5):
        for i, string in enumerate(self.df['reactions'].head(num_reactions)):
            try:
                reaction = AllChem.ReactionFromSmarts(string)
                img = Draw.ReactionToImage(reaction, subImgSize=(200,200))
                display(img)  
            except Exception as e:
                print(f"Error parsing reaction {i}: {e}")

    def create_translation_pairs(self) -> List[Tuple[str, str]]:
        self.df['translation_pair'] = self.df.apply(lambda row: (row['reactants'], row['products']), axis=1)
        return self.df['translation_pair'].tolist()