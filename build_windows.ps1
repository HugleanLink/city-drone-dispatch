param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& $Python -m PyInstaller --noconfirm --onedir --name DroneDispatch `
    --collect-all streamlit --collect-all streamlit_folium --copy-metadata streamlit `
    --exclude-module torch --exclude-module tensorflow --exclude-module IPython `
    --add-data 'streamlit_app.py:.' --add-data 'data/demo_orders.csv.gz:data' `
    --add-data 'data/demo_network.json:data' --add-data 'data/demo_planning.json.gz:data' launch_windows.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Copy-Item -LiteralPath 'README.md' -Destination 'dist/DroneDispatch/README.md'
New-Item -ItemType Directory -Path 'dist/DroneDispatch/.streamlit' -Force | Out-Null
Copy-Item -LiteralPath '.streamlit/config.toml' -Destination 'dist/DroneDispatch/.streamlit/config.toml'
Write-Output 'dist/DroneDispatch/DroneDispatch.exe'
