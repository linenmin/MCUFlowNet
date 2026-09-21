param([ValidateSet('torch','tf')][string]$Framework, [string]$RunName = 'formal')
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location $repo
$commit = (git rev-parse HEAD).Trim()
$runRoot = 'C:/00Work/Runs/MCUFlowNet'
$outRoot = "$runRoot/GPU-FPS-01/$RunName"
New-Item -ItemType Directory -Force $outRoot | Out-Null
$torchRuns = @('raft-things','spynet-chairs','pwc-things','raft-small','fastflow-things','neuflow2-things','rapidflow-things','gmflow-things','sea-raft-tartan-things','raft-sintel','pwc-sintel','spynet-sintel')
$tfRuns = @('edge-full','edge-chunks','nano-native-provisional','MCUFlowNet-S','MCUFlowNet-L')
$runs = if ($Framework -eq 'torch') { $torchRuns } else { $tfRuns }
foreach ($run in $runs) {
    if (Test-Path "$outRoot/$run") { throw "Output exists: $outRoot/$run" }
    Write-Output "START $run $(Get-Date -Format o)"
    if ($Framework -eq 'torch') {
        $manifest = Get-Content "$runRoot/SINTEL-BASE-01/$run/manifest.json" -Raw | ConvertFrom-Json
        & C:/00Work/Envs/sintel-torch/python.exe tools/baselines/benchmark_fps.py --model $manifest.model --weights $manifest.weights --upstream C:/00Work/Code/optical-flow-upstream --dataset C:/00Work/Datasets/Sintel --output "$outRoot/$run" --reference "$runRoot/SINTEL-BASE-01/$run" --code-commit $commit *> "$outRoot/$run.log"
    } else {
        $extraArgs = @()
        if ($run -like 'MCUFlowNet-*') {
            $model = $run
            $weights = "/runs/pretrained/published-20260917/MCUFlowNet_checkpoint/$run/sintel_best.ckpt"
            $reference = '/runs/20260917-sl-sintel01'
        } else {
            $manifest = Get-Content "$runRoot/SINTEL-BASE-01/$run/manifest.json" -Raw | ConvertFrom-Json
            $model = $manifest.model
            $weights = $manifest.weights.Replace('\','/').Replace('C:/00Work/Code/optical-flow-upstream','/upstream')
            $reference = "/runs/SINTEL-BASE-01/$run"
            if ($model -eq 'nano') { $extraArgs = @('--nano-native') }
        }
        & wsl -d Ubuntu-24.04 -- docker run --rm --gpus all --shm-size 1g -v /mnt/c/00Work/Code/MCUFlowNet-baselines:/workspace:ro -v /mnt/c/00Work/Code/optical-flow-upstream:/upstream:ro -v /mnt/c/00Work/Datasets:/datasets:ro -v /mnt/c/00Work/Runs/MCUFlowNet:/runs -w /workspace mcuflownet-local:20260917 python tools/baselines/benchmark_fps.py --model $model --weights $weights --upstream /upstream --dataset /datasets/Sintel --output "/runs/GPU-FPS-01/$RunName/$run" --reference $reference --code-commit $commit @extraArgs *> "$outRoot/$run.log"
    }
    if ($LASTEXITCODE -ne 0) {
        Get-Content "$outRoot/$run.log" -Tail 20
        throw "Benchmark failed for $run"
    }
    $result = Get-Content "$outRoot/$run/result.json" -Raw | ConvertFrom-Json
    Write-Output "DONE $run FPS=$($result.fps)"
}
