# Owen Studio

个人技术博客，聚焦计算机视觉与深度学习。在线地址：<https://everglow01.github.io/Owen-Studio/>

## 技术栈

- **静态站点生成器**：[Hugo Extended](https://gohugo.io/) `0.161.1`（SCSS 编译必须 extended 版）
- **主题**：[Blowfish](https://github.com/nunocoracao/blowfish)（作为 Git submodule 挂载在 `themes/blowfish`）
- **托管 / 部署**：GitHub Pages + GitHub Actions 自动构建
- **评论系统**：Giscus（基于 GitHub Discussions）
- **访问统计**：GoatCounter
- **数学公式**：KaTeX（按需启用，文章顶加 `{{< katex >}}`）
- **图片放大**：medium-zoom

## 方案设计

- **内容**：Markdown 写在 `content/posts/<slug>/index.md`，图片放同目录用相对路径引用，方便整篇文章作为目录搬迁
- **配置分层**：所有站点参数集中在 `config/_default/`（`hugo.toml` / `languages.zh-cn.toml` / `params.toml` / `menus.zh-cn.toml` / `markup.toml`），多语言只需新增 `languages.<lang>.toml`
- **主题不动**：所有定制走 Hugo 的 **local override** 机制 —— 在根目录 `layouts/` 下放同路径同名文件覆盖 `themes/blowfish/layouts/` 对应模板，永远不修改 submodule 内部，方便升级
- **自定义视觉**：`assets/css/custom.css` 集中维护全站 CSS 变量与首页 / 正文页特效；`assets/js/site-fx.js` 集中维护交互脚本（粒子、星空、月亮轨迹、图片缩放等）
- **CI/CD**：push 到 `main` → GitHub Actions 跑 `hugo --gc --minify` → 产物上传到 Pages

## 个人定制

绝大多数定制只动以下几个文件，互不耦合：

| 想改什么 | 改哪里 |
|---|---|
| 站点标题、作者信息、社交链接 | `config/_default/languages.zh-cn.toml` |
| 导航菜单 | `config/_default/menus.zh-cn.toml` |
| 主题行为开关（TOC、目录、首页布局等） | `config/_default/params.toml` |
| 颜色 / 字体 / 全站样式 | `assets/css/custom.css` |
| 首页特效 / 交互脚本 | `assets/js/site-fx.js` |
| 注入到 `<head>` 的统计、第三方脚本 | `layouts/partials/extend-head.html` |
| 覆盖主题模板（如正文页 single） | 在 `layouts/` 下按 `themes/blowfish/layouts/` 同路径创建文件 |
| 头像 / favicon | `assets/img/` 与 `static/` |

写新文章：

```bash
hugo new posts/my-new-post/index.md
hugo server -D   # 本地预览（含 draft），打开 http://localhost:1313
```

## 复刻 / 迁移

详细步骤见 [SETUP.md](./SETUP.md)。简版流程：

1. Fork 或克隆本仓库（**带 submodule**）：`git clone --recurse-submodules <url>`
2. 安装 Hugo Extended `0.161.1`（版本必须与 `.github/workflows/hugo.yml` 中的 `HUGO_VERSION` 一致）
3. 改 `config/_default/` 下的站点信息与 `baseURL`
4. 替换 `assets/img/touxiang.png` 等个人素材
5. `hugo server -D` 本地跑通后 push 到自己仓库

## GitHub Pages 部署

工作流在 `.github/workflows/hugo.yml`，开箱即用，只需要：

1. GitHub 仓库 **Settings → Pages → Source** 选择 `GitHub Actions`
2. **Settings → Actions → General → Workflow permissions** 勾选 `Read and write permissions`
3. push 到 `main` 自动触发；构建产物部署到 `https://<user>.github.io/<repo>/`
4. `config/_default/hugo.toml` 里的 `baseURL` 改成自己的 Pages 地址（CI 也会用 `actions/configure-pages` 自动覆盖，但本地构建仍依赖这里）

绑定自定义域名：在 `static/CNAME` 写入域名，并把 `baseURL` 改成 `https://your-domain.com/`。

## 许可

- 文章内容：版权归作者所有，部分图片来自网络，若侵权请联系作者删除。
- 主题：Blowfish 遵循其原仓库 License
