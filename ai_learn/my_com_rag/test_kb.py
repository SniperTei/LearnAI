#!/usr/bin/env python3
"""
测试脚本 - 用于快速测试知识库功能
"""
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from backend.core.document_processor import DocumentProcessor
from backend.core.vector_store import VectorStoreManager
from backend.core.rag_chain import KnowledgeBase


def test_document_processing():
    """测试文档处理"""
    print("=" * 60)
    print("📄 测试文档处理")
    print("=" * 60)

    processor = DocumentProcessor()

    # 测试加载目录中的文档
    documents_path = "data/documents"
    if Path(documents_path).exists():
        print(f"\n📂 加载目录: {documents_path}")
        documents = processor.load_documents_from_directory(documents_path)
        print(f"✅ 加载了 {len(documents)} 个文档")

        if documents:
            print(f"\n📝 第一个文档预览:")
            print(f"内容: {documents[0].page_content[:200]}...")
            print(f"元数据: {documents[0].metadata}")
    else:
        print(f"⚠️  目录不存在: {documents_path}")
        print(f"请在 {documents_path} 中放入一些文档进行测试")

    print()


def test_vector_store():
    """测试向量数据库"""
    print("=" * 60)
    print("🗄️  测试向量数据库")
    print("=" * 60)

    manager = VectorStoreManager()
    info = manager.get_collection_info()

    print(f"\n📊 向量数据库信息:")
    print(f"类型: {info.get('type')}")
    print(f"集合名称: {info.get('collection_name')}")
    print(f"初始化状态: {info.get('initialized')}")

    if 'count' in info:
        print(f"文档数量: {info.get('count')}")

    print()


def test_knowledge_base():
    """测试知识库"""
    print("=" * 60)
    print("🤖 测试知识库问答")
    print("=" * 60)

    kb = KnowledgeBase()

    # 获取知识库信息
    info = kb.get_info()
    print(f"\n📊 知识库状态:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 测试搜索
    test_query = "人工智能"
    print(f"\n🔍 测试搜索: '{test_query}'")

    try:
        results = kb.search(test_query, k=3)
        print(f"✅ 找到 {len(results)} 个相关文档")

        for i, result in enumerate(results, 1):
            print(f"\n  [{i}] 相似度: {result.get('score', 'N/A')}")
            print(f"  内容: {result.get('content', '')[:150]}...")
            if result.get('metadata'):
                print(f"  来源: {result['metadata'].get('file_name', 'N/A')}")
    except Exception as e:
        print(f"⚠️  搜索失败: {e}")

    print()


def test_api_imports():
    """测试API导入"""
    print("=" * 60)
    print("🔌 测试API模块")
    print("=" * 60)

    try:
        from backend.api.main import app
        print("✅ FastAPI应用导入成功")

        routes = []
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                for method in route.methods:
                    routes.append(f"{method} {route.path}")

        print(f"\n📋 可用的API端点 ({len(routes)}):")
        for route in sorted(routes):
            print(f"  {route}")

    except Exception as e:
        print(f"❌ API导入失败: {e}")

    print()


def main():
    """主测试函数"""
    print("\n")
    print("🧪 企业知识库 - 测试脚本")
    print("=" * 60)
    print()

    try:
        # 测试各个模块
        test_api_imports()
        test_document_processing()
        test_vector_store()
        test_knowledge_base()

        print("=" * 60)
        print("✅ 测试完成!")
        print("=" * 60)
        print("\n💡 提示:")
        print("  1. 在 data/documents/ 目录中放入文档进行测试")
        print("  2. 运行 'python start.py' 启动服务")
        print("  3. 访问 http://localhost:8000/docs 查看API文档")
        print()

    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
