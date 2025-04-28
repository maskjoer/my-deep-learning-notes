# 添加远程仓库

```python
git remote add origin 仓库名称
例子 git remote add origin https://github.com/maskjoer/my-deep-learning-notes.git
error: remote origin already exists.
```

- 本地已存在名为 `origin` 的远程配置

# 修改现有仓库URL

```python
git remote set-url origin https://github.com/maskjoer/my-deep-learning-notes.git
```

# 强制命名当前空间分支

```python
git branch -M  分支名称(test2025.4.28)
```

* `-M` 是 `--move --force` 的缩写，表示强制重命名（即使有未提交的更改或冲突）
* 假设原分支名为 `main` 或 `master`，执行后原分支名被覆盖为 `test2025.4.28`。
* 若目的是创建新分支而非重命名，应使用 `git checkout -b test2025.4.28`。

# 推送当前分支到仓库

```
git push -u origin test2025.4.28
```

# 如何推送当前分支

```

git checkout main              # 切换到 main
git pull origin main           # 更新本地 main
git merge your-current-branch  # 合并你的分支
git status                     # 检查状态
git add .                      # 添加所有更改
git commit -m "更新文档和工具"   # 提交
git pull origin main           # 拉取远程最新代码
git push origin main           # 推送到远程

```
