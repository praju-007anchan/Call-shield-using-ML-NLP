Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Starting CallShield Web Application..." -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Cyan
Set-Location -Path $PSScriptRoot
python app.py
