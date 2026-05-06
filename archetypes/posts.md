---
title: "{{ replace .Name "-" " " | title }}"
date: {{ .Date }}
draft: false
categories: ["tech-talk"]   # 必须是英文 slug，可选：2d-vision / 3d-vision / multimodal / notes / linux / tech-talk
tags: []                    # 中文/英文随意，例如 ["ROS", "Ubuntu"]
description: ""             # 一句话简介，会显示在卡片下和搜索结果
showAuthor: true
showDateUpdated: false
---


<!-- 写正文。在这个目录下放：
     cover.<ext>      → 卡片封面（约 1200x800，1.5:1）
     background.<ext> → 文章页 hero 大背景（约 1920x800，宽幅）
     正文里插图也放在这个目录，markdown 用 ![alt](图片名.png) 即可
-->
