param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
$workspaceDirectory = Split-Path (Split-Path $projectDirectory -Parent) -Parent
$runtimeDirectory = Join-Path $workspaceDirectory 'work/runtime'
New-Item -ItemType Directory -Force $runtimeDirectory | Out-Null
& $Python -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12"'
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }
& $Python -m venv (Join-Path $runtimeDirectory 'venv')
if ($LASTEXITCODE -ne 0) { throw 'Could not create environment.' }
$basePrefix = & $Python -c 'import sys; print(sys.base_prefix)'
$tclSource = Join-Path $basePrefix 'tcl'
if (Test-Path -LiteralPath $tclSource) {
    Copy-Item -LiteralPath $tclSource -Destination (Join-Path $runtimeDirectory 'tcl') -Recurse -Force
}
$venvPython = Join-Path $runtimeDirectory 'venv/Scripts/python.exe'
& $venvPython -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $venvPython -m pip install -r (Join-Path $projectDirectory 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Audio dependencies installation failed.' }
& $venvPython (Join-Path $projectDirectory 'install_singing_model.py')
if ($LASTEXITCODE -ne 0) { throw 'Vocal activity model installation failed.' }
Write-Host 'Setup complete. FFmpeg and ffprobe must be available on PATH. Run launch.ps1.'
