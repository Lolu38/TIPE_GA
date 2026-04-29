"""
circuit_loader.py
=================
Loader dynamique pour tous les fichiers circuit_*.py du dossier tracks/.
Remplace build_env_from_track_config et ses if/elif.

Contrat attendu dans chaque circuit_*.py :
    get_name()        → str
    get_difficulty()  → int
    get_walls()       → list of ((x1,y1),(x2,y2))
    get_spawn()       → (x, y, theta)
    get_width()       → float
    get_checkpoints() → list of (x, y)
    get_track_geo()       → RectangularTrack | AngularTrack
                        (temporaire — sera retiré quand tout sera AngularTrack)

Usage :
    from tracks.circuit_loader import load_circuits

    configs = load_circuits()             # tous les circuits
    configs = load_circuits(n=2)          # les 2 plus faciles
    configs = load_circuits(names=["nascar", "rectangle"])

    # Chaque config est un dict :
    # {
    #   "name":        str,
    #   "difficulty":  int,
    #   "env":         VectorizedCarEnv,
    #   "checkpoints": list,
    #   "walls":       list,
    # }
"""

import importlib.util
import sys
from pathlib import Path
from envs.neuronal_env import VectorizedCarEnv


_REQUIRED = ("get_name", "get_difficulty", "get_walls", "get_spawn", "get_width", "get_checkpoints", "get_track_geo")


# -- Chargement dynamique --------------------------------------------------------

def _load_module(path: Path):
    """Importe un fichier .py dynamiquement. Retourne None si l'import échoue."""
    module_name = f"_circuit_dyn_{path.stem}"

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec   = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"[circuit_loader] ⚠  Impossible de charger {path.name} : {e}")
        return None

    sys.modules[module_name] = module
    return module


def _validate(module, path: Path):
    """Vérifie que toutes les fonctions obligatoires sont présentes."""
    missing = [fn for fn in _REQUIRED if not hasattr(module, fn)]
    if missing:
        print(
            f"[circuit_loader] ⚠  {path.name} ignoré — "
            f"fonctions manquantes : {', '.join(missing)}"
        )
        return False
    return True


# -- Construction de l'env ------------------------------------------------------

def _build_config(module, n_cars: int, n_rays: int, device: str) -> dict | None:
    """
    Appelle les 7 fonctions du module et construit le VectorizedCarEnv.
    Retourne None si la construction échoue.
    """
    name = module.get_name()

    try:
        walls = module.get_walls()
        spawn = module.get_spawn()
        track_width = module.get_width()
        checkpoints = module.get_checkpoints()
        track = module.get_track_geo()
    except Exception as e:
        print(f"[circuit_loader] ⚠  Erreur lors de la lecture de '{name}' : {e}")
        return None

    try:
        env = VectorizedCarEnv(spawn_point = spawn, walls = walls, track = track, track_width = track_width, n_cars = n_cars, n_rays = n_rays, device = device)
    except Exception as e:
        print(f"[circuit_loader] ⚠  VectorizedCarEnv a échoué pour '{name}' : {e}")
        return None

    return {"name":name, "difficulty":module.get_difficulty(), "env":env, "checkpoints":checkpoints, "walls":walls}


# -- Interface publique -----------------------------------------------------------

def load_circuits( tracks_dir:str|Path = "tracks", n:int|None = None, names:list|None = None, n_cars:int  = 100, n_rays:int = 9, device:str = "cuda"):
    """
    Charge tous les circuits valides et retourne une liste de configs.
    Paramètres
    ----------
    tracks_dir : dossier où chercher les fichiers circuit_*.py
    n          : retourne les n circuits les moins difficiles (ignoré si names est fourni)
    names      : retourne uniquement les circuits dont le nom est dans la liste
    n_cars     : nombre de voitures par env
    n_rays     : nombre de rayons lidar par voiture
    device     : 'cuda' ou 'cpu'
    Retour
    ------
    Liste de dicts triés par difficulté croissante.
    """
    tracks_path = Path(tracks_dir)
    if not tracks_path.exists():
        raise FileNotFoundError(f"Dossier introuvable : {tracks_path.resolve()}")

    circuit_files = sorted(tracks_path.glob("circuit_*.py"))
    if not circuit_files:
        raise FileNotFoundError(
            f"Aucun fichier circuit_*.py trouvé dans {tracks_path.resolve()}"
        )

    # --- Chargement + validation ---
    modules = []
    for path in circuit_files:
        module = _load_module(path)
        if module is None:
            continue
        if not _validate(module, path):
            continue
        modules.append(module)

    if not modules:
        raise RuntimeError("Aucun circuit valide n'a pu être chargé.")

    # --- Tri par difficulté croissante ---
    modules.sort(key=lambda m: m.get_difficulty())

    # --- Filtrage par noms ---
    if names is not None:
        name_set  = set(names)
        modules   = [m for m in modules if m.get_name() in name_set]
        not_found = name_set - {m.get_name() for m in modules}
        if not_found:
            print(f"[circuit_loader] Circuits introuvables : {not_found}")

    # --- Limitation à n circuits (les plus faciles) ---
    elif n is not None:
        if n > len(modules):
            print(
                f"[circuit_loader] n={n} demandé mais seulement "
                f"{len(modules)} circuit(s) disponible(s) — tous chargés."
            )
        modules = modules[:n]

    # --- Construction des envs ---
    configs = []
    for module in modules:
        config = _build_config(module, n_cars, n_rays, device)
        if config is not None:
            configs.append(config)

    if not configs:
        raise RuntimeError("Aucun environnement n'a pu être construit.")

    print(
        f"[circuit_loader] {len(configs)} circuit(s) prêt(s) : "
        + ", ".join(f"{c['name']}(diff={c['difficulty']})" for c in configs)
    )

    return configs