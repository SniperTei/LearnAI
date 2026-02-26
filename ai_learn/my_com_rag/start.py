#!/usr/bin/env python3
"""
启动脚本
"""
import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)

# 加载环境变量
load_dotenv()


def check_env():
    """检查环境配置"""
    env_file = SCRIPT_DIR / ".env"
    if not env_file.exists():
        print("⚠️  警告: .env 文件不存在")
        print("请复制 .env.example 为 .env 并配置你的API密钥")
        print("\n运行命令:")
        print("  cp .env.example .env")
        print("  然后编辑 .env 文件填入你的 ZHIPUAI_API_KEY\n")
        return False
    return True


def install_dependencies():
    """安装依赖"""
    print("📦 检查依赖...")

    try:
        import fastapi
        import langchain
        print("✅ 依赖已安装")
        return True
    except ImportError:
        print("📥 正在安装依赖...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依赖安装完成")
        return True


def create_directories():
    """创建必要的目录"""
    directories = [
        "data/documents",
        "data/uploads",
        "data/vector_db",
        "logs"
    ]

    for directory in directories:
        (SCRIPT_DIR / directory).mkdir(parents=True, exist_ok=True)

    print("📁 目录结构已创建")


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 企业知识库系统 - 启动脚本")
    print("=" * 60)
    print()

    # 检查环境
    if not check_env():
        sys.exit(1)

    # 创建目录
    create_directories()

    # 安装依赖
    try:
        install_dependencies()
    except Exception as e:
        print(f"❌ 安装依赖失败: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("✅ 准备完成!")
    print("=" * 60)
    print()

    # 从环境变量读取配置
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print("🌐 API服务地址: http://localhost:{}".format(port))
    print("📖 API文档地址: http://localhost:{}/docs".format(port))
    print("🎨 Web界面地址: 请在浏览器打开 frontend/index.html")
    print()
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    print()

    # 启动服务
    try:
        import uvicorn
        uvicorn.run(
            "backend.api.main:app",
            host=host,
            port=port,
            reload=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
