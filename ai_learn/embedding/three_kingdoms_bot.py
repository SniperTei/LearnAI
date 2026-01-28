"""
三国问答机器人 v2
基于 Word2Vec + 文本检索
"""

from gensim.models import Word2Vec
import jieba
import re

print("=" * 70)
print("三国问答机器人")
print("=" * 70)

# ===== 加载模型和文本 =====
print("\n加载模型和文本...")
model = Word2Vec.load("word2vec.model")

with open("three_kingdoms.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("✓ 加载完成\n")

# ===== 三国人名列表 =====
PEOPLE_LIST = [
    "刘备", "关羽", "张飞", "诸葛亮", "赵云", "黄忠", "马超", "魏延", "姜维", "庞统", "法正",
    "曹操", "司马懿", "夏侯惇", "张辽", "许褚", "典韦", "郭嘉", "夏侯渊", "曹仁", "曹洪", "荀彧",
    "孙权", "周瑜", "鲁肃", "吕蒙", "陆逊", "甘宁", "太史慈", "黄盖", "程普", "孙策",
    "吕布", "董卓", "袁绍", "公孙瓒", "刘表", "刘璋", "刘禅"
]

# ===== 阵营信息 =====
FACTION_INFO = {
    "蜀": ["刘备", "关羽", "张飞", "诸葛亮", "赵云", "黄忠", "马超", "魏延", "姜维", "庞统", "法正", "刘禅"],
    "魏": ["曹操", "司马懿", "夏侯惇", "张辽", "许褚", "典韦", "郭嘉", "夏侯渊", "曹仁", "曹洪", "荀彧"],
    "吴": ["孙权", "孙策", "周瑜", "鲁肃", "吕蒙", "陆逊", "甘宁", "太史慈", "黄盖", "程普"],
    "其他": ["吕布", "董卓", "袁绍", "公孙瓒", "刘表", "刘璋"]
}

# 创建反向映射：人物 -> 阵营
PERSON_TO_FACTION = {}
for faction, people in FACTION_INFO.items():
    for person in people:
        PERSON_TO_FACTION[person] = faction


# ===== 提取问题中的人名 =====
def extract_person_names(question):
    """从问题中提取人名"""
    found = []
    for name in PEOPLE_LIST:
        if name in question:
            found.append(name)
    return found


# ===== 功能函数 =====

def get_similar_words(target_name, topn=10, people_only=False):
    """获取相似的词"""
    if target_name not in model.wv:
        return []

    # 获取更多相似词（人名可能排名靠后，需要搜索更多）
    search_size = topn * 100 if people_only else topn * 10
    similar = model.wv.most_similar(target_name, topn=search_size)

    # 分类
    people_result = []
    other_result = []

    for word, score in similar:
        if word in PEOPLE_LIST:
            people_result.append((word, score))
        else:
            # 过滤掉单字和明显不是词的内容
            if len(word) >= 2:
                other_result.append((word, score))

    if people_only:
        # 只要人名
        return people_result[:topn]
    else:
        # 混合：优先显示人名
        if len(people_result) >= topn:
            return people_result[:topn]
        else:
            # 人名不够，补充其他词
            return people_result + other_result[:topn - len(people_result)]


def search_in_text(keyword, max_results=5):
    """在文本中搜索关键词"""
    sentences = text.split("。")
    results = []

    for sent in sentences:
        if keyword in sent and len(sent.strip()) > 5:
            results.append(sent.strip())
            if len(results) >= max_results:
                break

    return results


# ===== 问题处理 =====

def handle_question(question):
    """处理用户问题"""
    question = question.strip()
    names = extract_person_names(question)

    # 类型1: 相似度查询
    if any(kw in question for kw in ["相似", "像", "接近", "类似"]):
        if names:
            target = names[0]

            # 判断是否只要人名（问"人"、"人物"、"人名"都算）
            people_only = any(kw in question for kw in ["人名", "人物", "是谁", "有谁", "哪些人", "人"])

            similar = get_similar_words(target, topn=10, people_only=people_only)

            if similar:
                if people_only:
                    answer = f"与 '{target}' 相似的人物有：\n"
                else:
                    answer = f"与 '{target}' 相似的词有：\n"
                    people_count = sum(1 for w, s in similar if w in PEOPLE_LIST)
                    if people_count > 0:
                        answer += f"（其中包括 {people_count} 个人名，标注★）\n"

                for i, (word, score) in enumerate(similar[:10], 1):
                    bar = "█" * int(score * 30)
                    is_person = "★" if word in PEOPLE_LIST else " "
                    answer += f"  {i:2}. {is_person} {word:<8} (相似度: {score:.3f}) {bar}\n"

                return answer.strip()
            else:
                return f"找不到与 '{target}' 相似的词"
        else:
            return "请问您想查询哪个人物的相似人物？"

    # 类型2: 阵营查询
    elif any(kw in question for kw in ["哪国", "哪个国家", "是蜀", "是魏", "是吴", "阵营"]):
        if names:
            target = names[0]
            if target in PERSON_TO_FACTION:
                faction = PERSON_TO_FACTION[target]
                faction_name = {"蜀": "蜀国", "魏": "魏国", "吴": "吴国"}.get(faction, faction)

                # 找出同阵营的人
                same_faction = [p for p in FACTION_INFO[faction] if p != target][:5]

                answer = f"{target} 属于 {faction_name}\n"
                if same_faction:
                    answer += f"\n同阵营的人物：{', '.join(same_faction)}"
                return answer
            else:
                # 搜索文本
                results = search_in_text(target, max_results=3)
                if results:
                    return f"关于 '{target}'：\n\n" + "\n".join(f"  {r}。" for r in results)
                else:
                    return f"找不到 '{target}' 的阵营信息"
        else:
            return "请问您想查询哪位人物的阵营？"

    # 类型3: 关系查询
    elif "关系" in question:
        if len(names) >= 2:
            name1, name2 = names[0], names[1]

            if name1 in model.wv and name2 in model.wv:
                sim = model.wv.similarity(name1, name2)

                answer = f"'{name1}' 和 '{name2}' 的分析：\n\n"
                answer += f"  词向量相似度: {sim:.3f}\n"

                if sim > 0.2:
                    answer += f"  评价: 关系较为密切（经常一起出现）"
                elif sim > 0.1:
                    answer += f"  评价: 有一定关联"
                else:
                    answer += f"  评价: 关联度较低"

                # 搜索文本中的描述
                search_text = f"{name1}.*?{name2}|{name2}.*?{name1}"
                matches = re.findall(search_text, text[:50000])
                if matches:
                    answer += f"\n\n  文本提及: {matches[0][:80]}..."

                return answer
            else:
                return f"模型中缺少 '{name1}' 或 '{name2}' 的信息"
        else:
            return "请问您想查询哪两个人物的关系？"

    # 类型4: 描述/是谁
    elif any(kw in question for kw in ["是谁", "是什么人", "介绍", "描述"]):
        if names:
            target = names[0]
            results = search_in_text(target, max_results=5)

            if results:
                answer = f"关于 '{target}' 的信息：\n\n"
                for i, r in enumerate(results[:5], 1):
                    answer += f"  {i}. {r}。\n"

                # 添加阵营信息
                if target in PERSON_TO_FACTION:
                    faction = PERSON_TO_FACTION[target]
                    faction_name = {"蜀": "蜀国", "魏": "魏国", "吴": "吴国"}.get(faction, faction)
                    answer += f"\n  所属阵营: {faction_name}"

                return answer.strip()
            else:
                return f"找不到关于 '{target}' 的详细信息"
        else:
            return "请问您想了解哪位人物？"

    # 类型5: 词向量运算
    elif "加" in question or "减" in question or "+" in question or "-" in question:
        # 解析人名
        positive = []
        negative = []

        for name in names:
            # 检查前面是否有"减"或"-"
            name_pos = question.find(name)
            if name_pos > 0:
                before = question[max(0, name_pos-2):name_pos]
                if "减" in before or "-" in before or "不" in before:
                    negative.append(name)
                else:
                    positive.append(name)
            else:
                positive.append(name)

        if positive:
            try:
                result = model.wv.most_similar(
                    positive=positive if positive else None,
                    negative=negative if negative else None,
                    topn=5
                )

                # 过滤出人名
                result_people = [(w, s) for w, s in result if w in PEOPLE_LIST][:5]

                expr = " + ".join(positive)
                if negative:
                    expr += " - " + " - ".join(negative)

                answer = f"词向量运算: {expr} = ?\n\n"

                if result_people:
                    answer += "结果（人物）：\n"
                    for i, (word, score) in enumerate(result_people, 1):
                        bar = "█" * int(score * 30)
                        answer += f"  {i}. {word:<6} (相似度: {score:.3f}) {bar}\n"
                else:
                    answer += "结果（所有词）：\n"
                    for i, (word, score) in enumerate(result[:5], 1):
                        bar = "█" * int(score * 30)
                        answer += f"  {i}. {word:<6} (相似度: {score:.3f}) {bar}\n"

                return answer.strip()
            except Exception as e:
                return f"运算失败: {e}"
        else:
            return "请使用格式如：刘备加曹操减张飞"

    # 类型6: 通用搜索
    else:
        if names:
            # 搜索人名相关信息
            target = names[0]
            results = search_in_text(target, max_results=5)

            if results:
                answer = f"关于 '{target}'：\n\n"
                for i, r in enumerate(results[:5], 1):
                    answer += f"  {i}. {r}。\n"
                return answer.strip()

        # 关键词搜索
        keywords = [w for w in jieba.cut(question) if len(w) > 1]
        if keywords:
            # 搜索包含关键词的句子
            sentences = text.split("。")
            scored = []

            for sent in sentences[:1000]:
                score = sum(1 for kw in keywords if kw in sent)
                if score > 0 and len(sent.strip()) > 10:
                    scored.append((sent.strip(), score))

            scored.sort(key=lambda x: x[1], reverse=True)

            if scored:
                answer = f"找到以下相关信息：\n\n"
                for i, (sent, score) in enumerate(scored[:5], 1):
                    answer += f"  {i}. {sent}。\n"
                return answer.strip()

        return "抱歉，找不到相关信息。请尝试更具体的问题。"


# ===== 交互界面 =====
def main():
    print("\n" + "=" * 70)
    print("欢迎使用三国问答机器人！")
    print("=" * 70)
    print("\n你可以问以下类型的问题：")
    print("  1. 相似度：'和刘备相似的人有哪些？'")
    print("  2. 阵营：'典韦是哪个国家的？'")
    print("  3. 关系：'张飞和关羽是什么关系？'")
    print("  4. 描述：'诸葛亮是谁？'")
    print("  5. 运算：'刘备加曹操减张飞等于什么？'")
    print("  6. 搜索：任意三国相关内容")
    print("\n输入 'quit' 退出\n")

    while True:
        try:
            print("\n" + "-" * 70)
            question = input("请输入问题: ").strip()

            if question.lower() in ['quit', 'exit', '退出', 'q']:
                print("\n再见！")
                break

            if not question:
                continue

            answer = handle_question(question)
            print(f"\n{answer}")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n出错: {e}")


if __name__ == "__main__":
    main()
