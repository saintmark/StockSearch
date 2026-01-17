# Git 仓库初始化脚本
# 使用方法：在 PowerShell 中运行：.\init_git.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  StockSearch Git 初始化脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查是否已初始化
if (Test-Path .git) {
    Write-Host "[INFO] Git 仓库已存在" -ForegroundColor Yellow
    git status
    Write-Host ""
} else {
    Write-Host "[1/5] 初始化 Git 仓库..." -ForegroundColor Green
    git init
    Write-Host "[OK] Git 仓库初始化完成" -ForegroundColor Green
    Write-Host ""
}

# 检查 .gitignore
if (Test-Path .gitignore) {
    Write-Host "[2/5] 检查 .gitignore 文件..." -ForegroundColor Green
    Write-Host "[OK] .gitignore 文件存在" -ForegroundColor Green
} else {
    Write-Host "[WARN] .gitignore 文件不存在" -ForegroundColor Yellow
}
Write-Host ""

# 添加文件到暂存区
Write-Host "[3/5] 添加文件到暂存区..." -ForegroundColor Green
git add .
Write-Host "[OK] 文件已添加到暂存区" -ForegroundColor Green
Write-Host ""

# 显示将要提交的文件
Write-Host "将要提交的文件：" -ForegroundColor Cyan
git status --short
Write-Host ""

# 询问是否提交
$commit = Read-Host "是否现在提交代码？(Y/N)"
if ($commit -eq "Y" -or $commit -eq "y") {
    $message = Read-Host "请输入提交信息（直接回车使用默认信息）"
    if ([string]::IsNullOrWhiteSpace($message)) {
        $message = "Initial commit: StockSearch project"
    }
    Write-Host "[4/5] 提交代码..." -ForegroundColor Green
    git commit -m $message
    Write-Host "[OK] 代码提交完成" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "[SKIP] 跳过提交，您可以稍后手动执行：git commit -m '提交信息'" -ForegroundColor Yellow
    Write-Host ""
}

# 检查远程仓库
Write-Host "[5/5] 检查远程仓库配置..." -ForegroundColor Green
$remote = git remote -v
if ($remote) {
    Write-Host "当前远程仓库：" -ForegroundColor Cyan
    Write-Host $remote
    Write-Host ""
} else {
    Write-Host "[INFO] 尚未配置远程仓库" -ForegroundColor Yellow
    Write-Host ""
    $addRemote = Read-Host "是否现在添加远程仓库？(Y/N)"
    if ($addRemote -eq "Y" -or $addRemote -eq "y") {
        $remoteUrl = Read-Host "请输入远程仓库地址（例如：https://github.com/username/StockSearch.git）"
        if ($remoteUrl) {
            git remote add origin $remoteUrl
            Write-Host "[OK] 远程仓库已添加" -ForegroundColor Green
            Write-Host ""
            
            $push = Read-Host "是否现在推送到远程仓库？(Y/N)"
            if ($push -eq "Y" -or $push -eq "y") {
                Write-Host "正在推送到远程仓库..." -ForegroundColor Green
                git branch -M main
                git push -u origin main
                Write-Host "[OK] 代码已推送到远程仓库" -ForegroundColor Green
            }
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  初始化完成！" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "后续操作：" -ForegroundColor Yellow
Write-Host "  1. 添加远程仓库: git remote add origin <仓库地址>" -ForegroundColor White
Write-Host "  2. 推送代码: git push -u origin main" -ForegroundColor White
Write-Host "  3. 查看状态: git status" -ForegroundColor White
Write-Host ""
Write-Host "详细文档请查看: readme/GIT_SYNC_GUIDE.md" -ForegroundColor Cyan

