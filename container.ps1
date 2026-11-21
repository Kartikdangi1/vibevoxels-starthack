# container.ps1 - Windows twin of container.bash: the ONE entrypoint (Docker Desktop, WSL2 backend).
# Differometor needs Python 3.12 + JAX with Linux CPU-affinity pinning (benchmark.py --cpus 1), which native Windows
# does not support - this container gives everyone the same Linux environment regardless of host OS.
#
#   .\container.ps1                  build (if needed) + open a shell in the container
#   .\container.ps1 -b | --build     force rebuild the image (after editing requirements.txt or the Dockerfile)
#   .\container.ps1 exec CMD...      run one command in the container and exit, e.g.:
#                                       .\container.ps1 exec python start_here.py
#                                       .\container.ps1 exec python benchmark.py --cpus 1 --output benchmark-baseline.json
#   .\container.ps1 -s | --stop      stop the container (keeps the image/volumes)
#   .\container.ps1 -d | --delete    remove the container
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
  if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Write-Host "XX docker not found" -ForegroundColor Red; exit 1 }
  docker compose version *> $null
  if ($LASTEXITCODE) { Write-Host "XX docker compose v2 not found" -ForegroundColor Red; exit 1 }

  $env:HOST_UID = '1000'; $env:HOST_GID = '1000'   # Docker Desktop does not map host IDs
  $Compose = @('compose', '-f', 'docker/docker-compose.yaml')

  $a = [System.Collections.Generic.List[string]]::new(); foreach ($x in $args) { $a.Add($x) }
  $Mode = 'shell'
  if ($a.Count -gt 0) {
    switch -Regex ($a[0]) {
      '^(-b|--build)$' { docker @Compose build; if ($LASTEXITCODE) { exit 1 }; $a.RemoveAt(0) }
      '^(-s|--stop)$' { docker @Compose stop; exit 0 }
      '^(-d|--delete)$' { docker @Compose down; exit 0 }
      '^exec$' { $Mode = 'exec'; $a.RemoveAt(0) }
      '^(-h|--help)$' { foreach ($l in (Get-Content $PSCommandPath)[1..18]) { $l }; exit 0 }
    }
  }

  $have = $null
  try { $have = docker image inspect -f '{{.Id}}' 'feyer-hackathon:dev' 2>$null } catch {}
  if (-not $have) {
    Write-Host "-> building feyer-hackathon:dev (first build is large, pulls jax/jaxlib etc.)" -ForegroundColor Cyan
    docker @Compose build
    if ($LASTEXITCODE) { Write-Host "XX build failed" -ForegroundColor Red; exit 1 }
  }

  if ($Mode -eq 'exec') {
    docker @Compose run --rm hack @a
  } else {
    docker @Compose run --rm hack bash
  }
} finally { Pop-Location }
