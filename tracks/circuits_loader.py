"""
circuits_loader_v2.py
=====================
Identique à circuits_loader.py avec deux différences :

  1. Instancie VectorizedCarEnv depuis envs.neuronal_env_v2
     (physique réaliste, systèmes pneus/fuel/pluie, collision GPU)

  2. Accepte rain_mode, initial_rain, initial_compound
     pour configurer les systèmes dès la construction de l'env.

Contrat attendu dans chaque circuit_*.py (identique à v1) :
    get_name()        -> str
    get_difficulty()  -> int
    get_walls()       -> list of ((x1,y1),(x2,y2))
    get_spawn()       -> (x, y, theta)
    get_width()       -> float
    get_checkpoints() -> list of (x, y)
    get_track_geo()   -> RectangularTrack | AngularTrack

Usage :
    from tracks.circuits_loader_v2 import load_circuits
"""

import importlib.util
import sys
from pathlib import Path
from envs.neuronal_env_physic import VectorizedCarEnv
from physics.tires_system import MEDIUM

_REQUIRED = (
    "get_name", "get_difficulty", "get_walls",
    "get_spawn", "get_width", "get_checkpoints", "get_track_geo"
)


# -- Chargement dynamique (identique à v1) ------------------------------------

def _load_module(path: Path):
    module_name = f"_circuit_dyn_{path.stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec   = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"[circuits_loader_v2] ⚠  Impossible de charger {path.name} : {e}")
        return None
    sys.modules[module_name] = module
    return module


def _validate(module, path: Path) -> bool:
    missing = [fn for fn in _REQUIRED if not hasattr(module, fn)]
    if missing:
        print(
            f"[circuits_loader_v2] ⚠  {path.name} ignoré — "
            f"fonctions manquantes : {', '.join(missing)}"
        )
        return False
    return True


# -- Construction de l'env (v2 : paramètres systèmes ajoutés) -----------------

def _build_config(
    module,
    n_cars           : int,
    n_rays           : int,
    device           : str,
    rain_mode        : str,
    initial_rain     : float,
    initial_compound : int,
) -> dict | None:
    name = module.get_name()
    try:
        walls       = module.get_walls()
        spawn       = module.get_spawn()
        track_width = module.get_width()
        checkpoints = module.get_checkpoints()
        track       = module.get_track_geo()
    except Exception as e:
        print(f"[circuits_loader_v2] ⚠  Erreur lecture '{name}' : {e}")
        return None

    try:
        env = VectorizedCarEnv(
            spawn_point      = spawn,
            walls            = walls,
            track            = track,
            track_width      = track_width,
            n_cars           = n_cars,
            n_rays           = n_rays,
            device           = device,
            rain_mode        = rain_mode,
            initial_rain     = initial_rain,
            initial_compound = initial_compound,
        )
    except Exception as e:
        print(f"[circuits_loader_v2] ⚠  VectorizedCarEnv a échoué pour '{name}' : {e}")
        return None

    return {
        "name"        : name,
        "difficulty"  : module.get_difficulty(),
        "env"         : env,
        "checkpoints" : checkpoints,
        "walls"       : walls,
    }


# -- Interface publique --------------------------------------------------------

def load_circuits(
    tracks_dir       : str | Path = "tracks",
    n                : int | None = None,
    names            : list | None = None,
    n_cars           : int   = 1000,
    n_rays           : int   = 9,
    device           : str   = 'cuda',
    rain_mode        : str   = 'fixed',
    initial_rain     : float = 0.0,
    initial_compound : int   = MEDIUM,
) -> list:
    """
    Charge tous les circuits valides et retourne une liste de configs.

    Paramètres supplémentaires par rapport à v1 :
        rain_mode        : 'fixed' | 'dynamic' | 'preset'
        initial_rain     : intensité initiale de la pluie [0, 1]
        initial_compound : composé de départ (constantes dans tire_system_gpu.py)
    """
    tracks_path   = Path(tracks_dir)
    circuit_files = sorted(tracks_path.glob("circuit_*.py"))

    if not circuit_files:
        raise FileNotFoundError(
            f"Aucun fichier circuit_*.py trouvé dans {tracks_path.resolve()}"
        )

    modules = []
    for path in circuit_files:
        module = _load_module(path)
        if module and _validate(module, path):
            modules.append(module)

    if not modules:
        raise RuntimeError("Aucun circuit valide n'a pu être chargé.")

    modules.sort(key=lambda m: m.get_difficulty())

    if names is not None:
        name_set  = set(names)
        modules   = [m for m in modules if m.get_name() in name_set]
        not_found = name_set - {m.get_name() for m in modules}
        if not_found:
            print(f"[circuits_loader_v2] Circuits introuvables : {not_found}")
    elif n is not None:
        modules = modules[:n]

    configs = []
    for module in modules:
        cfg = _build_config(
            module, n_cars, n_rays, device,
            rain_mode, initial_rain, initial_compound
        )
        if cfg is not None:
            configs.append(cfg)

    if not configs:
        raise RuntimeError("Aucun environnement n'a pu être construit.")

    print(
        f"[circuits_loader_v2] {len(configs)} circuit(s) prêt(s) : "
        + ", ".join(f"{c['name']}(diff={c['difficulty']})" for c in configs)
    )
    return configs