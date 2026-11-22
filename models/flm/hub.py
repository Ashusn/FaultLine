import shutil
import time
from pathlib import Path

from .common import dataset_repo, log, namespace, repo_id, wdir

RETRY_CODES = {429, 500, 502, 503, 504}


def api():
    from huggingface_hub import HfApi

    return HfApi()


def _retry(fn, tries=4):
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    for i in range(tries):
        try:
            return fn()
        except RepositoryNotFoundError:
            raise
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code not in RETRY_CODES or i == tries - 1:
                raise
            time.sleep(10 * (i + 1))


def _snapshot(src):
    src = Path(src)
    tmp = Path("/tmp/flm_upload") / f"{src.name}_{int(time.time() * 1000)}"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    return tmp


def push_folder(repo, folder, path_in_repo=None, repo_type="model", private=False, message="Upload", ignore=None):
    a = api()
    _retry(lambda: a.create_repo(repo, repo_type=repo_type, private=private, exist_ok=True))
    snap = _snapshot(folder)
    try:
        _retry(lambda: a.upload_folder(folder_path=str(snap), path_in_repo=path_in_repo, repo_id=repo, repo_type=repo_type, commit_message=message, ignore_patterns=ignore))
    finally:
        shutil.rmtree(snap, ignore_errors=True)
    log("uploaded", folder, "to", repo, path_in_repo or "")


def push_text(repo, path_in_repo, text, repo_type="model", message="Update"):
    a = api()
    _retry(lambda: a.upload_file(path_or_fileobj=text.encode(), path_in_repo=path_in_repo, repo_id=repo, repo_type=repo_type, commit_message=message))


def push_model(cfg, name, local_dir, adapter_dir=None, extra_dirs=None):
    repo = repo_id(cfg, name)
    private = bool(cfg["hub"].get("private", False))
    push_folder(repo, local_dir, private=private, message="Upload weights", ignore=["README.md", "checkpoint-*/**"])
    if adapter_dir and Path(adapter_dir).exists():
        push_folder(repo, adapter_dir, path_in_repo="adapter", private=private, message="Upload LoRA adapter", ignore=["README.md"])
    for sub, d in (extra_dirs or {}).items():
        if Path(d).exists():
            push_folder(repo, d, path_in_repo=sub, private=private, message=f"Upload {sub}")
    return repo


def push_dataset(cfg, key, folder):
    repo = dataset_repo(cfg, key)
    push_folder(repo, folder, repo_type="dataset", private=bool(cfg["hub"].get("private", False)), message=f"Upload {key} data")
    from .card import dataset_card

    push_text(repo, "README.md", dataset_card(cfg, key, folder), repo_type="dataset", message="Dataset card")
    return repo


def download_model(cfg, name, dest):
    from huggingface_hub import snapshot_download

    repo = repo_id(cfg, name)
    files = api().list_repo_files(repo)
    if not files:
        raise SystemExit(f"verify {name}: repo {repo} is empty")
    shutil.rmtree(dest, ignore_errors=True)
    snapshot_download(repo, local_dir=str(dest), ignore_patterns=["adapter/*", "training/*", "eval/*", "logs/*"])
    return dest, files


def add_to_collection(cfg, repos):
    title = cfg["hub"].get("collection")
    if not title:
        return
    try:
        a = api()
        ns = namespace(cfg)
        existing = [c for c in a.list_collections(owner=ns, limit=100) if c.title == title]
        col = existing[0] if existing else a.create_collection(title=title, namespace=ns, description=cfg["hub"].get("collection_description", ""), exists_ok=True)
        for r, kind in repos:
            try:
                a.add_collection_item(col.slug, item_id=r, item_type=kind, exists_ok=True)
            except Exception as e:
                log("collection item skipped", r, e)
        from .common import wpath, write_json

        write_json(wpath(cfg, "results", "collection.json"), {"slug": col.slug, "url": f"https://huggingface.co/collections/{col.slug}"})
        return col.slug
    except Exception as e:
        log("collection skipped", e)


def run_id(cfg):
    from .steps import load_state, save_state

    s = load_state(cfg)
    if not s.get("run_id"):
        s["run_id"] = time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + "_" + Path(cfg["_path"]).stem
        save_state(cfg, s)
    return s["run_id"]


def runs_url(cfg):
    return f"https://huggingface.co/datasets/{dataset_repo(cfg, 'runs')}/tree/main/runs/{run_id(cfg)}"


def upload_run(cfg, final=False):
    try:
        rid = run_id(cfg)
        snap = Path("/tmp/flm_run") / rid
        shutil.rmtree(snap, ignore_errors=True)
        snap.mkdir(parents=True)
        for sub in ("logs", "results", "report"):
            src = wdir(cfg, sub)
            if any(src.iterdir()):
                shutil.copytree(src, snap / sub, ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.pt", "*.joblib"))
        for f in ("state.json",):
            p = Path(cfg["_work"]) / f
            if p.exists():
                shutil.copy(p, snap / f)
        cfg_copy = Path(cfg["_path"])
        shutil.copy(cfg_copy, snap / cfg_copy.name)
        repo = dataset_repo(cfg, "runs")
        a = api()
        _retry(lambda: a.create_repo(repo, repo_type="dataset", private=bool(cfg["hub"].get("runs_private", False)), exist_ok=True))
        _retry(lambda: a.upload_folder(folder_path=str(snap), path_in_repo=f"runs/{rid}", repo_id=repo, repo_type="dataset", commit_message=f"{'Final' if final else 'Live'} logs for {rid}"))
        shutil.rmtree(snap, ignore_errors=True)
        log("run uploaded", runs_url(cfg))
        return True
    except Exception as e:
        log("run upload failed, will retry after the next step:", e)
        return False


def model_dir(cfg, name):
    return wdir(cfg, "models", name)
