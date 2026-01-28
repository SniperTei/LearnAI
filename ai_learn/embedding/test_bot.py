"""
测试三国问答机器人
"""

# 导入机器人模块
import sys
sys.path.append('.')
from three_kingdoms_bot import ask_question

print("=" * 70)
print("三国问答机器人 - 测试")
print("=" * 70)

# 测试问题
test_questions = [
    "和刘备相似的人有哪些？",
    "典韦是蜀国还是魏国的？",
    "张飞和关羽是什么关系？",
    "诸葛亮是谁？",
    "刘备加曹操减张飞等于什么？",
    "曹操最擅长什么？",
]

for i, question in enumerate(test_questions, 1):
    print("\n" + "=" * 70)
    print(f"问题 {i}: {question}")
    print("=" * 70)
    answer = ask_question(question)
    print(f"\n回答:\n{answer}")
    print()

print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)
