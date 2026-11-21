# container.ps1 - Windows twin of container.bash: the ONE entrypoint (Docker Desktop, WSL2 backend).
# Differometor needs Python 3.12 + JAX with Linux CPU-affinity pinning (benchmark.py --cpus 1), which native Windows
# does not support - this container gives everyone the same Linux environment regardless of host OS.
#
#   .\container.ps1                  env up + VS Code attached to the container (needs the 'code' CLI)
#   .\container.ps1 -u | --up        env up, no shell, no VS Code
#   .\container.ps1 -t | --terminal  shell only (start the container first if needed)
#   .\container.ps1 exec CMD...      run one command in the running container and exit, e.g.:
#                                       .\container.ps1 exec python start_here.py
#                                       .\container.ps1 exec python benchmark.py --cpus 1 --output benchmark-baseline.json
#   .\container.ps1 -b | --build     force rebuild the image (after editing requirements.txt or the Dockerfile)
#   .\container.ps1 -s | --stop      stop the container (keeps the image/volumes)
#   .\container.ps1 -d | --delete    remove the container
#
# A running, up-to-date container is reused as is (no `docker compose up` on repeat calls).
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
  function Info($m) { Write-Host "-> $m" -ForegroundColor Cyan }
  function Ok($m) { Write-Host "OK $m" -ForegroundColor Green }
  function Warn($m) { Write-Host "!! $m" -ForegroundColor Yellow }
  function Die($m) { Write-Host "XX $m" -ForegroundColor Red; exit 1 }

  if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Die 'docker not found' }
  docker compose version *> $null
  if ($LASTEXITCODE) { Die 'docker compose v2 not found' }

  $env:HOST_UID = '1000'; $env:HOST_GID = '1000'   # Docker Desktop does not map host IDs
  $Compose = @('compose', '-f', 'docker/docker-compose.yaml')
  $CName = 'feyer_hackathon'

  $a = [System.Collections.Generic.List[string]]::new(); foreach ($x in $args) { $a.Add($x) }
  $Mode = 'attach'; $Force = $false
  :argloop while ($a.Count -gt 0) {
    $t = $a[0]; $a.RemoveAt(0)
    switch -Regex ($t) {
      '^(-h|--help)$' { foreach ($l in (Get-Content $PSCommandPath)[1..14]) { $l -replace '^# ?', '' }; exit 0 }
      '^(-u|--up)$' { $Mode = 'up' }
      '^(-t|--terminal)$' { $Mode = 'shell' }
      '^exec$' { $Mode = 'exec'; break argloop }
      '^(-b|--build)$' { $Force = $true }
      '^(-s|--stop)$' { docker @Compose stop; exit 0 }
      '^(-d|--delete)$' { docker @Compose down; exit 0 }
      default { Die "unknown option $t (see -h)" }
    }
  }

  $have = $null
  try { $have = docker image inspect -f '{{.Id}}' 'feyer-hackathon:dev' 2>$null } catch {}
  if ($Force -or -not $have) {
    Info "building image feyer-hackathon:dev (first build is large, pulls jax/jaxlib etc.)"
    docker @Compose build
    if ($LASTEXITCODE) { Die 'image build failed' }
  }
  $running = $null
  try { $running = (docker inspect -f '{{.State.Running}}' $CName 2>$null) } catch {}
  if ($running -ne 'true') {
    Info "starting container"
    docker @Compose up -d | Out-Null
    if ($LASTEXITCODE) { Die 'docker compose up failed' }
  }

  switch ($Mode) {
    'up' { Ok "container is up; shell: .\container.ps1 -t" }
    'attach' {
      if (Get-Command code -ErrorAction SilentlyContinue) {
        $cid = (docker inspect -f '{{.Id}}' $CName 2>$null | Select-Object -First 1)
        $hex = -join ([System.Text.Encoding]::UTF8.GetBytes($cid) | ForEach-Object { $_.ToString('x2') })
        Info "opening VS Code attached..."
        code --folder-uri "vscode-remote://attached-container+$hex/work"
      } else {
        Warn "'code' CLI not found (install the VS Code 'code' shell command) - opening a shell instead"
        docker exec -it $CName bash
      }
    }
    'shell' { docker exec -it $CName bash }
    'exec' { docker exec -it $CName @a }
  }
} finally { Pop-Location }
