"""
circuit_loader.py
=================
Loader dynamique pour tous les fichiers circuit_*.py du dossier tracks/.

Usage :
    from tracks.circuit_loader import load_circuits
    from circuit import Circuit

    # Tous les circuits, triés par difficulté
    circuits = load_circuits()

    # Les N plus faciles seulement
    circuits = load_circuits(n=2)

    # Sélection par nom
    circuits = load_circuits(names=["nascar", "rectangle"])

Contrat attendu dans chaque circuit_*.py :
    get_name()        → str           (obligatoire)
    get_difficulty()  → int           (obligatoire pour le tri)
    get_spawn()       → (x, y, theta) (obligatoire)
    get_walls()       → list of ((x1,y1),(x2,y2))  (obligatoire)
    get_width()       → float         (obligatoire)
    get_checkpoints() → list of (x,y) (optionnel)

Si un fichier est invalide (import raté ou fonction manquante), il est ignoré
avec un warning — les autres circuits chargent quand même.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

from circuit import Circuit


# Fonctions obligatoires que chaque module doit exposer
_REQUIRED = ("get_name", "get_difficulty", "get_spawn", "get_walls", "get_width")


def _load_module(path: Path):
    """
    Importe dynamiquement un fichier .py à partir de son chemin absolu.
    Retourne le module, ou None si l'import échoue.
    """
    module_name = f"_circuit_dyn_{path.stem}"

    # Évite de recharger si déjà importé dans la session
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec   = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"[circuit_loader] ⚠ Impossible de charger {path.name} : {e}")
        return None

    sys.modules[module_name] = module
    return module


def _validate(module, path: Path) -> bool:
    """Vérifie que le module expose toutes les fonctions obligatoires."""
    missing = [fn for fn in _REQUIRED if not hasattr(module, fn)]
    if missing:
        print(
            f"[circuit_loader] ⚠ {path.name} ignoré — "
            f"fonctions manquantes : {', '.join(missing)}"
        )
        return False
    return True


def _module_to_circuit(module) -> Circuit:
    """Construit un objet Circuit à partir d'un module valide."""
    checkpoints = (
        module.get_checkpoints()
        if hasattr(module, "get_checkpoints")
        else []
    )
    circuit = Circuit(
        walls       = module.get_walls(),
        spawn       = module.get_spawn(),
        track_width = module.get_width(),
        name        = module.get_name(),
    )
    # Stocke les checkpoints directement sur l'objet pour la fitness
    circuit.checkpoints = checkpoints
    circuit.difficulty  = module.get_difficulty()
    return circuit


def load_circuits(
    tracks_dir: str | Path = "tracks",
    n:          int | None  = None,
    names:      list | None = None,
) -> list[Circuit]:
    """
    Charge et retourne les circuits disponibles.

    Paramètres
    ----------
    tracks_dir : dossier où chercher les fichiers circuit_*.py
    n          : si fourni, retourne les n circuits les moins difficiles
    names      : si fourni, retourne uniquement les circuits dont le nom est dans la liste
                 (prioritaire sur n)

    Retour
    ------
    Liste de Circuit triés par difficulté croissante.
    """
    tracks_path = Path(tracks_dir)
    if not tracks_path.exists():
        raise FileNotFoundError(f"Dossier introuvable : {tracks_path.resolve()}")

    circuit_files = sorted(tracks_path.glob("circuit_*.py"))

    if not circuit_files:
        raise FileNotFoundError(
            f"Aucun fichier circuit_*.py trouvé dans {tracks_path.resolve()}"
        )

    # Chargement + validation
    circuits = []
    for path in circuit_files:
        module = _load_module(path)
        if module is None:
            continue
        if not _validate(module, path):
            continue
        circuits.append(_module_to_circuit(module))

    if not circuits:
        raise RuntimeError("Aucun circuit valide n'a pu être chargé.")

    # Tri par difficulté croissante
    circuits.sort(key=lambda c: c.difficulty)

    # Filtrage par noms si demandé
    if names is not None:
        name_set  = set(names)
        circuits  = [c for c in circuits if c.name in name_set]
        not_found = name_set - {c.name for c in circuits}
        if not_found:
            print(f"[circuit_loader] ⚠ Circuits introuvables : {not_found}")

    # Limitation à n circuits (les plus faciles)
    elif n is not None:
        if n > len(circuits):
            print(
                f"[circuit_loader] ⚠ n={n} demandé mais seulement "
                f"{len(circuits)} circuit(s) disponible(s) — tous chargés."
            )
        circuits = circuits[:n]

    # Résumé
    print(
        f"[circuit_loader] {len(circuits)} circuit(s) chargé(s) : "
        + ", ".join(f"{c.name}(diff={c.difficulty})" for c in circuits)
    )

    return circuits