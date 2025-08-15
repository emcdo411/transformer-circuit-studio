param(
  [string]$RepoDir = "C:\Users\Veteran\Documents\transformer-circuit-studio"
)
Set-Location $RepoDir
git status
git add -A
git commit -m "chore: initial scaffold (folders + non-empty files)" | Out-Null
git branch -M main
git push -u origin main
Write-Host "Pushed to 'main'."
