# SETUP.md

Owen Studio 博客项目的本地开发环境搭建说明。**重装系统后照此文档操作即可继续维护**。

---

## 项目概览

- **生成器**：Hugo Extended `0.161.1`
- **主题**：[Blowfish](https://github.com/nunocoracao/blowfish)（作为 Git submodule 挂载在 `themes/blowfish`）
- **部署**：GitHub Actions 自动构建 → GitHub Pages（`https://everglow01.github.io/Owen-Studio/`）
- **评论**：Giscus（基于 GitHub Discussions）
- **统计**：GoatCounter（外部账户，配置在 `layouts/partials/extend-head.html`）
- **数学公式**：KaTeX（按需启用，文章顶部加 `{{< katex >}}` 触发）

---

## 一、首次环境搭建（重装系统后从这里开始）

### 1. 安装基础工具

```bash
sudo apt update
sudo apt install -y git curl
```

### 2. 安装 Hugo Extended `0.161.1`

⚠️ **必须装 extended 版**（Blowfish 用了 SCSS，普通版会报错）。
⚠️ **版本要和 GitHub Actions 里一致**（见 `.github/workflows/hugo.yml` 的 `HUGO_VERSION`），否则本地构建结果可能和线上不一致。

```bash
wget -O /tmp/hugo.deb \
  https://github.com/gohugoio/hugo/releases/download/v0.161.1/hugo_extended_0.161.1_linux-amd64.deb
sudo dpkg -i /tmp/hugo.deb

# 验证
hugo version
# 应输出: hugo v0.161.1+extended ...
```

### 3. 配置 GitHub SSH key

```bash
ssh-keygen -t ed25519 -C "everglow818@outlook.com"
cat ~/.ssh/id_ed25519.pub
# 复制公钥，粘贴到 GitHub: Settings → SSH and GPG keys → New SSH key

# 测试
ssh -T git@github.com
# 应看到: Hi everglow01! You've successfully authenticated...
```

### 4. 克隆仓库（**注意 submodule**）

```bash
cd ~
git clone --recurse-submodules git@github.com:everglow01/Owen-Studio.git my-first-blog
cd my-first-blog
```

如果忘了 `--recurse-submodules`，事后修复：

```bash
git submodule update --init --recursive
```

验证主题已拉下来：

```bash
ls themes/blowfish/   # 不应为空
```

### 5. 启动本地预览

```bash
hugo server -D
```

浏览器访问 `http://localhost:1313`，能看到首页就 OK。`-D` 表示包含草稿（`draft: true` 的文章）。

---

## 二、日常维护命令

### 写新文章

```bash
hugo new posts/<english-slug>/index.md
```

会基于 `archetypes/posts.md` 生成新文章模板（已预置 frontmatter 和 katex shortcode）。

### 草稿管理

- 文章 frontmatter 里 `draft: true` → 本地能看，构建到 GitHub Pages 时被跳过
- 写完后改成 `draft: false`，push 即上线

### 图片放置

每篇文章的图片放在该文章自己的目录里（页面 bundle 模式）：

```
content/posts/my-post/
├── index.md
├── cover.png        ← 文章卡片封面（约 1200x800）
├── background.png   ← 文章页 hero 大背景（约 1920x800）
└── 正文图片.png     ← Markdown 里 ![alt](正文图片.png) 引用
```

### 提交与部署

```bash
git add -A
git commit -m "新增博客 XXX"
git push
```

push 到 `main` 分支后 GitHub Actions 自动构建并部署到 GitHub Pages，**1-2 分钟生效**。

构建状态查看：`https://github.com/everglow01/Owen-Studio/actions`

---

## 三、常见坑与排查

### 坑 1：`themes/blowfish` 是空目录

submodule 没拉下来。执行：

```bash
git submodule update --init --recursive
```

### 坑 2：`hugo` 命令报错 SCSS 相关

装的不是 extended 版。重新下载带 `extended` 字样的 deb 包安装。

### 坑 3：本地能看到草稿，部署后看不到

正常行为。`hugo server -D` 包含草稿，但 GitHub Actions 里的 `hugo --gc --minify` 默认不包含 `draft: true` 的文章。

### 坑 4：数学公式不渲染

文章顶部漏写了 `{{< katex >}}` shortcode。Blowfish 是按需加载 KaTeX 的，没这一行不会加载库。

### 坑 5：本地预览样式异常

清缓存重启：

```bash
rm -rf resources/ public/ .hugo_build.lock
hugo server -D
```

### 坑 6：评论或访问数没显示

- Giscus：检查 `layouts/partials/comments.html` 里的 `data-repo`、`data-category-id` 是否正确
- GoatCounter：登录 `https://owen-studio.goatcounter.com/` 看 Settings → Site settings → Allow access 是否启用

---

## 四、重装系统前的备份清单

Git 只能恢复**已经 push 上去**的内容。重装前必查：

```bash
# 1. 看有没有未提交的改动
git status

# 2. 看有没有已 commit 但没 push 的
git log origin/main..HEAD

# 3. 全部 push 上去
git add -A && git commit -m "重装前保存进度" && git push
```

任何"还在本地、没 push"的内容（草稿文章、未保存的图片、本地修改的配置）**重装后全部丢失**。

可选备份（丢了也能重新生成）：
- `~/.ssh/`（SSH key，重装后也可以重新生成并加到 GitHub）
- 当前 `HUGO_VERSION`（已记录在本文件和 `.github/workflows/hugo.yml`）

---

## 五、关键路径速查

| 场景 | 文件 |
|---|---|
| 改站点配置（标题、菜单、作者） | `config/_default/*.toml` |
| 改主题外观参数 | `config/_default/params.toml` |
| 改 markdown 渲染（KaTeX 等） | `config/_default/markup.toml` |
| 自定义 CSS | `assets/css/custom.css` |
| 代码块顶部 header（语言名/复制按钮） | `layouts/_default/_markup/render-codeblock.html` |
| 评论组件 | `layouts/partials/comments.html` |
| 阅读量统计组件 | `layouts/partials/meta/views.html` |
| 文章页 footer 注入 JS | `layouts/partials/extend-footer.html` |
| 头部注入（GoatCounter 等） | `layouts/partials/extend-head.html` |
| 主页 SVG 背景 | `layouts/partials/extend-head-uncached.html` |
| 新文章模板 | `archetypes/posts.md` |
| 部署工作流 | `.github/workflows/hugo.yml` |
