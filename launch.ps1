$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
$candidates = @(
    (Join-Path $projectDirectory '.venv/Scripts/pythonw.exe'),
    (Join-Path $projectDirectory 'work/runtime/venv/Scripts/pythonw.exe')
)
$parent2 = Split-Path (Split-Path $projectDirectory -Parent) -Parent
if ($parent2) {
    $candidates += (Join-Path $parent2 'work/runtime/venv/Scripts/pythonw.exe')
}
$pythonExecutable = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $pythonExecutable) {
    $cmd = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if ($cmd) { $pythonExecutable = $cmd.Source }
}
if (-not $pythonExecutable) {
    throw 'Python runtime missing. Download the standalone MelodyDesk.exe release from GitHub Releases or create .venv first.'
}
Start-Process -FilePath $pythonExecutable -ArgumentList ('"' + (Join-Path $projectDirectory 'melodydesk_desktop.py') + '"') -WorkingDirectory $projectDirectory -WindowStyle Hidden
