# Dockerfile — 七个服务共用一个镜像（L5）
# 【学徒注】镜像 = 镜头里的布景（代码+依赖，只读）；容器 = 一场演出。
# 同一个镜像起七个容器，各自 command 不同——复用镜像层，构建快、省空间。
# 密钥不进镜像（.env 在 .dockerignore 里挡掉）：镜像会被推送/分发，
# 钥匙跟着镜像走 = 见者有份；密钥走 compose environment 注入（运行时才给）。
FROM python:3.11-slim

WORKDIR /app

# 先拷 requirements 再装依赖：这层只在你改依赖时才重建，
# 改业务代码重新 build 时 pip 层直接吃缓存（Docker 分层缓存的经典用法）。
#
# 【学徒注】三步装法（复刻宿主机的共存态）：干净解析天生不可能——
# fastapi 0.121.3 钉死 starlette<0.51，fastmcp 4.2（经 fastmcp-slim
# [client,server]）要 >=1.0.1，两头对撞 ResolutionImpossible。
# 宿主机当初是"先装齐再降级"救回来的（L2~L4 全实测能跑），容器里
# 照方抓药：①摘掉 fastapi/starlette 两行让 pip 装齐全家（starlette
# 落 1.x）②强制压回 0.50.0 ③最后装 fastapi（starlette 0.50 正好满足
# 它）。pip 中途会警告依赖冲突——警告不是错误，运行时真不兼容才会
# 炸，而这个组合的运行时 L2~L4 已全部实测过。
COPY requirements.txt .
RUN grep -v -e '^starlette' -e '^fastapi' requirements.txt > /tmp/reqs.txt \
 && pip install --no-cache-dir -r /tmp/reqs.txt \
 && pip install --no-cache-dir --force-reinstall "starlette==0.50.0" \
 && pip install --no-cache-dir "fastapi==0.121.3" \
 && python -c "import starlette, fastapi, fastmcp, langchain; print('依赖冒烟OK：starlette', starlette.__version__)"

# 代码和种子账本（data/ 的 JSON 是种子，首次起卷时被拷进空卷——
# 之后账本活在实际挂载的卷里，升级容器不丢账）
COPY config.py agent.py web.py eval.py ./
COPY servers/ servers/
COPY static/ static/
COPY data/ data/

# 默认门脸是 web；六个 MCP Server 用 command 覆盖成 --http 模式
CMD ["python", "web.py"]
