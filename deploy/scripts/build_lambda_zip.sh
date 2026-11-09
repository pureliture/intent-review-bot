#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output_dir="${1:-${repo_root}/deploy/terraform/dist}"
mkdir -p "${output_dir}"

docker run --rm --platform linux/amd64 \
  --volume "${repo_root}:/src:ro" \
  --volume "${output_dir}:/out:rw" \
  --workdir /src \
  --entrypoint /bin/bash \
  public.ecr.aws/lambda/python:3.11 \
  -euo pipefail -c '
    rm -rf /out/package /out/wheels /out/lambda.zip /out/review_bot.whl
    mkdir -p /out/package /out/wheels
    python -m pip wheel --no-deps --wheel-dir /out/wheels /src
    wheel="$(find /out/wheels -maxdepth 1 -type f -name "*.whl" -print -quit)"
    test -n "${wheel}"
    cp "${wheel}" /out/review_bot.whl
    python -m pip install --no-cache-dir --no-deps --target /out/package "${wheel}"
    python -m pip install --no-cache-dir --target /out/package \
      "boto3>=1.34,<2" "pydantic>=2.13,<3"
    cd /out
    PYTHONPATH=/out/package python -c "import review_bot.runtime.aws_lambda; print(\"aws_lambda import OK\")"
    python -c "import zipfile; from pathlib import Path; root=Path(\"/out/package\"); archive=zipfile.ZipFile(\"/out/lambda.zip\", \"w\", zipfile.ZIP_DEFLATED); [archive.write(path, path.relative_to(root).as_posix()) for path in sorted(root.rglob(\"*\")) if path.is_file() and \"__pycache__\" not in path.parts and path.suffix != \".pyc\"]; archive.close()"
  '

printf 'Lambda ZIP: %s\n' "${output_dir}/lambda.zip"
printf 'Application wheel: %s\n' "${output_dir}/review_bot.whl"
