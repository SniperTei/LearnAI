"""
测试改进后的三国问答机器人
"""

import sys
sys.path.append('.')
from three_kingdoms_bot import handle_question

print("=" * 70)
print("三国问答机器人 v3 - 改进测试")
print("=" * 70)

test_questions = [
    "和刘备相似的人有哪些？",           # 混合显示
    "和刘备相似的人物人名有哪些？",     # 只显示人名
    "和诸葛亮相似的人有哪些？",         # 混合显示
    "典韦是哪个国家的？",
    "关羽和张飞是什么关系？",
]

for i, question in enumerate(test_questions, 1):
    print("\n" + "=" * 70)
    print(f"问题 {i}: {question}")
    print("=" * 70)
    answer = handle_question(question)
    print(f"\n回答:\n{answer}")

print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)
