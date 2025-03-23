# see https://fastapi.tiangolo.com/deployment/docker/
# see https://docs.astral.sh/uv/guides/integration/fastapi/
# see https://github.com/astral-sh/uv-docker-example/tree/main

FROM python:3.12-slim

EXPOSE 8888
WORKDIR /app

# https://github.com/PhE/zkml-bootcamp2025Q1-g6
# git@github.com:PhE/zkml-bootcamp2025Q1-g6.git
# https://github.com/PhE/zkml-bootcamp2025Q1-g6.git
ENV GIT_CHECKOUT=no
ENV GIT_REPO=github.com/PhE/zkml-bootcamp2025Q1-g6b
ENV GIT_BRANCH=main
# set GIT_TOKEN in deploy env
ENV GIT_TOKEN=xxxx

# Run the application.
# start fastapi or jupyter upon env var ZK_SERVICE
#CMD ["/app/.venv/bin/fastapi", "run", "app/main.py", "--port", "8000", "--host", "0.0.0.0"]
# set ZK_SERVICE=fastapi or ZK_SERVICE=jupyter
CMD ["./zk", "service"]

RUN apt update \
    && apt install --yes git

# Install the application dependencies.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-cache

# Copy the application into the container.
COPY . /app/




## # Stage 1: Build Stage
## FROM node:22.14.0-alpine AS build
## WORKDIR /app
## COPY pnpm-lock.yaml package.json ./
## COPY . .
## RUN npm install -g corepack@latest
## RUN corepack enable
## RUN pnpm install --frozen-lockfile --prod
## RUN pnpm run build
## 
## # Stage 2: Final Stage
## FROM node:22.14.0-alpine AS final
## WORKDIR /app
## COPY --from=build /app/.output .output
## RUN apk update && apk add --no-cache curl
## EXPOSE 3000
## CMD ["node", ".output/server/index.mjs"]
