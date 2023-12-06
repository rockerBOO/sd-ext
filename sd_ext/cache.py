from platformdirs import user_cache_dir
from pathlib import Path


def ensure_model(
    model: Path, url: str, appname="sd_ext", appauthor="rockerBOO"
):
    cache_dir = Path(user_cache_dir(appname, appauthor))
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not cache_dir / model:
        import requests

        if url.startswith("http") or url.startswith("https"):
            print(f"Downloading {url}...")
            r = requests.get(url)
            with open(cache_dir / model, "wb") as f:
                f.write(r.content)
                print(f"Saved to {cache_dir / model}")
