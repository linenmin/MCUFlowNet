# Usage from PowerShell: .\tools\setup\run-local.ps1 python tools/setup/check_tensorflow.py --device gpu --output /runs/gpu-check.json
$ErrorActionPreference = 'Stop'
$repoPath = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$linuxRepo = (& wsl.exe -d Ubuntu-24.04 -u root --exec wslpath -a $repoPath.Replace('\', '/'))
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve repository path in WSL.' }
$linuxRepo = $linuxRepo.Trim()
$runsPath = 'C:/00Work/Runs/MCUFlowNet'
New-Item -ItemType Directory -Force -Path $runsPath | Out-Null
$commandArgs = @($args)
if ($commandArgs.Count -eq 0) { $commandArgs = @('python', '--version') }
& wsl.exe -d Ubuntu-24.04 -u root -- docker run --rm --gpus all --shm-size 1g `
    -v "${linuxRepo}:/workspace" `
    -v /mnt/c/00Work/Datasets:/datasets:ro `
    -v /mnt/c/00Work/Runs/MCUFlowNet:/runs `
    -w /workspace mcuflownet-local:20260917 @commandArgs
exit $LASTEXITCODE
