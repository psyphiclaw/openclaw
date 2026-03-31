# PsyPhiClaw Dashboard 部署指南

## 快速启动

```bash
# 安装依赖
pip install dash plotly pandas numpy

# 启动
python scripts/app.py --project-dir /path/to/project --port 8050

# 浏览器访问
open http://127.0.0.1:8050
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--project-dir` | 必填 | 项目数据目录路径 |
| `--host` | 127.0.0.1 | 绑定地址 |
| `--port` | 8050 | 端口号 |
| `--debug` | false | 开启调试模式 |

## 配置选项

### 主题切换
在 CSS 文件中通过 `data-theme="dark"` 属性切换暗色主题。

### 自定义配色
修改 `assets/dashboard.css` 中的 CSS 变量：
```css
:root {
  --primary: #4A90D9;    /* 主色 */
  --danger: #E74C3C;     /* 警告/错误色 */
  --success: #27ae60;    /* 成功色 */
}
```

### 添加自定义页面
1. 在 `scripts/pages/` 下创建新模块
2. 实现 `create_layout(project_dir)` 函数
3. 在 `app.py` 中注册导航项

## 生产部署

### Gunicorn
```bash
pip install gunicorn
gunicorn scripts.app:server --workers 4 --bind 0.0.0.0:8050
```

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY skills/psyphiclaw-web-dashboard /app
RUN pip install -r requirements.txt
EXPOSE 8050
CMD ["gunicorn", "scripts.app:server", "--workers", "2", "--bind", "0.0.0.0:8050"]
```

### Nginx 反向代理
```nginx
location /dash/ {
    proxy_pass http://127.0.0.1:8050/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```
