# material_library.py
"""
Material library management system for saving/loading material presets
"""
import json
from pathlib import Path
from typing import Dict, Optional


class MaterialLibrary:
    """Manages material presets with JSON persistence"""
    
    def __init__(self, filepath: str = "material_library.json", 
                 defaults_filepath: str = "default_materials.json"):
        self.filepath = Path(filepath)
        self.defaults_filepath = Path(defaults_filepath)
        self.materials: Dict[str, dict] = {}
        self.load()
    
    def load(self):
        """Load materials from JSON file"""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    self.materials = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading material library: {e}")
                self.materials = {}
        else:
            # Initialize with default materials from defaults file
            self.materials = self._load_default_materials()
            self.save()
    
    def save(self):
        """Save materials to JSON file"""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.materials, f, indent=2)
        except IOError as e:
            print(f"Error saving material library: {e}")
    
    def add_material(self, name: str, epsilon_pre: float, bulk_modulus: float):
        """Add a new material to the library"""
        self.materials[name] = {
            'epsilon_pre': epsilon_pre,
            'bulk_modulus': bulk_modulus
        }
        self.save()
    
    def remove_material(self, name: str) -> bool:
        """Remove a material from the library"""
        if name in self.materials:
            del self.materials[name]
            self.save()
            return True
        return False
    
    def get_material(self, name: str) -> Optional[dict]:
        """Get material parameters by name"""
        return self.materials.get(name)
    
    def list_materials(self) -> list:
        """Get list of all material names"""
        return sorted(self.materials.keys())
    
    def _load_default_materials(self) -> dict:
        """Load default materials from JSON file"""
        if self.defaults_filepath.exists():
            try:
                with open(self.defaults_filepath, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading default materials: {e}")
                return self._create_default_materials()
        else:
            # Create default materials file if it doesn't exist
            defaults = self._create_default_materials()
            try:
                with open(self.defaults_filepath, 'w') as f:
                    json.dump(defaults, f, indent=2)
                print(f"Created default materials file: {self.defaults_filepath}")
            except IOError as e:
                print(f"Error creating default materials file: {e}")
            return defaults
    
    def _create_default_materials(self) -> dict:
        """Create initial default materials"""
        return {
            'Ecoflex 00-50': {
                'epsilon_pre': 0.15,
                'bulk_modulus': 50e6  # 50 MPa
            },
            'Dragon Skin 30': {
                'epsilon_pre': 0.15,
                'bulk_modulus': 100e6
            },
            'Sylgard 184': {
                'epsilon_pre': 0.15,
                'bulk_modulus': 200e6
            }
        }