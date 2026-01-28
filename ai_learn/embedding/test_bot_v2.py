"""
测试三国问答机器人 v2
"""

# 导入机器人模块
import sys
sys.path.append('.')
from three_kingdoms_bot import handle_question

print("=" * 70)
print("三国问答机器人 - 功能演示")
print("=" * 70)

# 测试问题
test_questions = [
    "和刘备相似的人有哪些？",
    "典韦是哪个国家的？",
    "张飞和关羽是什么关系？",
    "诸葛亮是谁？",
    "刘备加曹操减张飞等于什么？",
    "孙权阵营有哪些人？",
    "赵云",
]

for i, question in enumerate(test_questions, 1):
    print("\n" + "=" * 70)
    print(f"问题 {i}: {question}")
    print("=" * 70)
    answer = handle_question(question)
    print(f"\n回答:\n{answer}")

print("\n" + "=" * 70)
print("演示完成！")
print("=" * 70)
print("\n要使用交互式机器人，请运行：")
print("  python three_kingdoms_bot.py")
