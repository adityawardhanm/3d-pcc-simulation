# material_library.py
"""Material library management system for saving/loading material presets."""

import json
from pathlib import Path
from typing import Optional


class MaterialLibrary:
    """Manage material presets with JSON persistence.

    Loads and saves material properties (pre-strain, bulk modulus) to JSON files.
    Provides default materials if no saved library exists.

    Args:
        filepath: Path to the user's material library JSON file.
        defaults_filepath: Path to the default materials JSON file.

    Attributes:
        materials: Dictionary of material names to their properties.
    """

    def __init__(
        self,
        filepath: str = "material_library.json",
        defaults_filepath: str = "default_materials.json"
    ):
        self.filepath = Path(filepath)
        self.defaults_filepath = Path(defaults_filepath)
        self.materials: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        """Load materials from JSON file."""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.materials = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading material library: {e}")
                self.materials = {}
        else:
            self.materials = self._load_default_materials()
            self.save()

    def save(self) -> None:
        """Save materials to JSON file."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.materials, f, indent=2)
        except IOError as e:
            print(f"Error saving material library: {e}")

    def add_material(
        self,
        name: str,
        epsilon_pre: float,
        bulk_modulus: float
    ) -> None:
        """Add a new material to the library.

        Args:
            name: Material name (e.g., "Ecoflex 00-50").
            epsilon_pre: Pre-strain value (dimensionless).
            bulk_modulus: Bulk modulus (Pa).
        """
        self.materials[name] = {
            "epsilon_pre": epsilon_pre,
            "bulk_modulus": bulk_modulus
        }
        self.save()

    def remove_material(self, name: str) -> bool:
        """Remove a material from the library.

        Args:
            name: Material name to remove.

        Returns:
            True if material was removed, False if not found.
        """
        if name in self.materials:
            del self.materials[name]
            self.save()
            return True
        return False

    def get_material(self, name: str) -> Optional[dict]:
        """Get material parameters by name.

        Args:
            name: Material name to look up.

        Returns:
            Dictionary with 'epsilon_pre' and 'bulk_modulus', or None.
        """
        return self.materials.get(name)

    def list_materials(self) -> list[str]:
        """Get sorted list of all material names.

        Returns:
            Sorted list of material names.
        """
        return sorted(self.materials.keys())

    def _load_default_materials(self) -> dict:
        """Load default materials from JSON file.

        Returns:
            Dictionary of default materials.
        """
        if self.defaults_filepath.exists():
            try:
                with open(self.defaults_filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading default materials: {e}")
                return self._create_default_materials()

        # Create default materials file if it doesn't exist
        defaults = self._create_default_materials()
        self._save_default_materials(defaults)
        return defaults

    def _save_default_materials(self, defaults: dict) -> None:
        """Save default materials to JSON file.

        Args:
            defaults: Dictionary of default materials to save.
        """
        try:
            with open(self.defaults_filepath, "w", encoding="utf-8") as f:
                json.dump(defaults, f, indent=2)
            print(f"Created default materials file: {self.defaults_filepath}")
        except IOError as e:
            print(f"Error creating default materials file: {e}")

    @staticmethod
    def _create_default_materials() -> dict:
        """Create initial default materials.

        Returns:
            Dictionary of default soft robotics materials.
        """
        return {
            "Ecoflex 00-50": {
                "epsilon_pre": 0.15,
                "bulk_modulus": 50e6
            },
            "Dragon Skin 30": {
                "epsilon_pre": 0.15,
                "bulk_modulus": 100e6
            },
            "Sylgard 184": {
                "epsilon_pre": 0.15,
                "bulk_modulus": 200e6
            }
        }