#!.venv/bin/python
"""
Command-Line Interface definition
Type `zk --help` to get available commands
"""

import os
from subprocess import call

import typer

app = typer.Typer(name="zk cli", help="zkml g6 CLI tool")

TOKEN_DEFAULT = "b6189a4b5d049f31fc5581e958d95e1bb27dbd3d5c173866"


@app.command()
def fastapi():
    """Start the fastapi service"""
    print(f"Starting fastapi ...")
    call(["uv", "run", "fastapi", "run", "--port=8888"])


@app.command()
def jupyter(
    browser: bool = True,
    token: str = TOKEN_DEFAULT,
):
    """Start the jupyter service"""
    token = os.environ.get("ZK_NOTEBOOK_TOKEN", token)
    # if token is None:
    #    print("ZK_NOTEBOOK_TOKEN envvar is not set, using default !!")

    print(f"Starting jupyter with token {token} ...")
    call(
        [
            "uv",
            "run",
            "jupyter",
            "lab",
            "--config=jupyter_notebook_config.json",
            "--allow-root",
            # "--ServerApp.ip=*",
            # "--ServerApp.password=''",
            f"--IdentityProvider.token=${token}",
            "" if browser else "--no-browser",
            "-y",
        ]
    )


@app.command()
def update():
    """Update git repo"""
    git_repo = os.environ["GIT_REPO"]
    git_branch = os.environ["GIT_BRANCH"]
    git_token = os.environ["GIT_TOKEN"]

    print(f"Updating from git repo ...")
    call(
        rf"""
        echo "git clone ...\n" \
        uv sync \        
        echo "git clone ...\n" \
        uv sync \        
        echo "git clone ...\n" \
        git clone --depth=1 --branch {git_branch} https://xyz:{git_token}@{git_repo}.git ../app_tmp \
        echo "mv /app /app_old \n" \
        echo "mv /app_tmp /app \n" \
        pwd \
        # echo "cd /app \n" \
        # pwd \
        uv sync \
        # echo "rm -Rf /app_old' \n" \
        """,
        shell=True,
    )


@app.command()
def service():
    """Start a service fastapi or jupyter depending on ZK_SERVICE envar"""
    git_checkout = os.environ.get("GIT_CHECKOUT")
    if git_checkout is not None:
        update()

    service_envvar = os.environ.get("ZK_SERVICE")
    if service_envvar is None:
        print("ZK_SERVICE envvar is not set, default to 'fastapi' !!")

    if service_envvar == "jupyter":
        jupyter(browser=False)
    else:
        fastapi()


if __name__ == "__main__":
    app()
