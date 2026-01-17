# Railway 部署步骤（快速指南）

## 一、准备工作

1. **注册 Railway 账号**
   - 访问 https://railway.app/
   - 使用 GitHub 账号登录

2. **准备代码仓库**
   - 确保代码已推送到 GitHub
   - 确保包含所有必要文件（已创建）

## 二、部署后端服务

### 步骤 1：创建新项目

1. 登录 Railway 后，点击 **"New Project"**
2. 选择 **"Deploy from GitHub repo"**
3. 选择你的代码仓库（StockSearch）

### 步骤 2：配置服务

Railway 会自动检测到 `Procfile`，如果检测失败，手动设置：

- **Root Directory**: 留空（项目根目录）
- **Build Command**: `cd server && pip install -r requirements.txt`
- **Start Command**: `cd server && uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`

### 步骤 3：添加环境变量

在项目设置 → **Variables** 中添加：

| 变量名 | 值 | 说明 |
|--------|-----|------|
| `LLM_API_KEY` | `sk-xxx...` | **必需** - 你的 SiliconFlow API Key |
| `SQLITE_DB_PATH` | `/data/stock_logic.db` | 数据库路径（使用 Volume 时） |

### 步骤 4：创建持久化存储（Volume）

1. 在 Railway 项目中，点击 **"New"** → **"Volume"**
2. 命名：`database`
3. 挂载路径：`/data`
4. 在服务设置中，将 Volume 挂载到服务

### 步骤 5：部署

1. Railway 会自动开始构建
2. 等待构建完成（3-5分钟）
3. 查看日志确认服务启动成功

### 步骤 6：获取服务地址

1. 在服务设置中，找到 **"Settings"** → **"Networking"**
2. 点击 **"Generate Domain"** 生成公网域名
3. 记录下域名（例如：`stocksearch-production.up.railway.app`）

## 三、部署前端（可选）

### 方案 A：使用 Vercel/Netlify（推荐）

1. **构建前端**
   ```bash
   cd frontend
   npm install
   npm run build
   ```

2. **部署到 Vercel**
   - 访问 https://vercel.com/
   - 导入 GitHub 仓库
   - 设置：
     - **Root Directory**: `frontend`
     - **Build Command**: `npm run build`
     - **Output Directory**: `dist`
     - **环境变量**：
       ```
       VITE_API_URL=https://your-backend-url.railway.app/api
       ```

### 方案 B：使用 Railway（同一平台）

1. 在 Railway 项目中，点击 **"New"** → **"Service"**
2. 选择 **"Deploy from GitHub repo"**，选择同一仓库
3. 设置：
   - **Root Directory**: `frontend`
   - **Build Command**: `npm install && npm run build`
   - **Start Command**: `npx serve -s dist -l $PORT`
   - **环境变量**：
     ```
     VITE_API_URL=https://your-backend-url.railway.app/api
     ```

## 四、验证部署

### 检查后端

1. 访问：`https://your-backend-url.railway.app/docs`
2. 应该看到 FastAPI 的 Swagger 文档界面

### 检查 API

```bash
# 测试健康检查
curl https://your-backend-url.railway.app/api/stocks/market_recommendations

# 应该返回 JSON 数据
```

### 检查前端

1. 访问前端部署地址
2. 确认可以正常加载
3. 测试 API 连接是否正常

## 五、常见问题解决

### 问题 1：构建失败

**错误信息**：`ModuleNotFoundError` 或依赖安装失败

**解决方案**：
- 检查 `server/requirements.txt` 是否完整
- 查看 Railway 构建日志获取详细错误
- 确保 Python 版本为 3.12

### 问题 2：服务无法启动

**错误信息**：`Port already in use` 或服务崩溃

**解决方案**：
- 确认启动命令使用 `$PORT` 环境变量
- 检查 `Procfile` 格式是否正确
- 查看 Railway 运行日志

### 问题 3：数据库文件丢失

**错误信息**：重启后数据消失

**解决方案**：
- 确认已创建并挂载 Volume
- 检查 `SQLITE_DB_PATH` 环境变量指向 Volume 路径
- 验证 Volume 挂载状态

### 问题 4：API 调用失败

**错误信息**：前端无法连接后端

**解决方案**：
- 检查 `VITE_API_URL` 环境变量是否正确
- 确认后端服务已成功部署
- 检查 CORS 配置（代码中已允许所有来源）

## 六、后续优化

1. **设置自定义域名**
   - 在 Railway 服务设置中添加自定义域名

2. **配置监控告警**
   - 设置服务健康检查
   - 配置错误通知

3. **优化性能**
   - 使用 Redis 缓存（Railway 提供 Redis 服务）
   - 优化数据库查询
   - 使用 CDN 加速前端资源

## 七、部署检查清单

- [ ] Railway 账号已创建
- [ ] 代码已推送到 GitHub
- [ ] 后端服务已创建并配置
- [ ] 环境变量已设置（`LLM_API_KEY`）
- [ ] Volume 已创建并挂载
- [ ] 构建成功完成
- [ ] 服务可以正常访问
- [ ] API 端点可以正常响应
- [ ] 前端已部署（如需要）
- [ ] 前端可以连接到后端

## 详细文档

更多详细信息请参考：[Railway 部署完整文档](readme/RAILWAY_DEPLOYMENT.md)

