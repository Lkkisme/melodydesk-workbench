$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
$workspaceDirectory = Split-Path (Split-Path $projectDirectory -Parent) -Parent
$pythonExecutable = Join-Path $workspaceDirectory 'work/runtime/venv/Scripts/pythonw.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw 'Runtime missing. Run setup.ps1 with Python 3.12 first.'
}
Start-Process -FilePath $pythonExecutable -ArgumentList ('"' + (Join-Path $projectDirectory 'app.py') + '"') -WorkingDirectory $projectDirectory -WindowStyle Hidden
