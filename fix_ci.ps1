Set-Location D:\Git\empirlab

# 1. Remove all lock files
Remove-Item .git\index.lock  -Force -ErrorAction SilentlyContinue
Remove-Item .git\HEAD.lock   -Force -ErrorAction SilentlyContinue
Remove-Item .git\MERGE_HEAD  -Force -ErrorAction SilentlyContinue

# 2. Remove accidentally committed pytest cache from git tracking
git rm -r --cached "pytest-cache-files-3ypepjka/" 2>$null

# 3. Stage all changes
git add -A

# 4. Commit
git commit -m "chore: add .gitignore, remove pytest cache dir from tracking"

# 5. Push
git push

Write-Host "`n[DONE] All fixes pushed." -ForegroundColor Green
