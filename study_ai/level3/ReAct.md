# ReAct：在语言模型中协同推理与行动

> 发表于 ICLR 2023 会议论文  
> 原文标题：**REACT: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS**  
> 作者：Shunyu Yao、Jeffrey Zhao、Dian Yu、Nan Du、Izhak Shafran、Karthik Narasimhan、Yuan Cao  
> 单位：普林斯顿大学计算机科学系；Google Research, Brain team

---

## 目录

- [摘要](#摘要)
- [1 引言](#1-引言)
- [2 ReAct：协同推理 + 行动](#2-react协同推理--行动)
- [3 知识密集型推理任务](#3-知识密集型推理任务)
- [4 决策任务](#4-决策任务)
- [5 相关工作](#5-相关工作)
- [6 结论](#6-结论)
- [致谢](#致谢)
- [可复现性声明](#可复现性声明)
- [伦理声明](#伦理声明)
- [附录](#附录)

---

## 摘要

尽管大语言模型（LLMs）在语言理解和交互式决策等任务中表现出令人印象深刻的性能，但它们的推理能力（例如思维链提示，chain-of-thought prompting）和行动能力（例如动作计划生成）主要被作为两个独立主题研究。本文探索使用 LLMs 以交错方式同时生成推理轨迹和任务特定动作，从而让二者产生更强的协同：推理轨迹帮助模型诱导、跟踪和更新行动计划，并处理异常；动作则允许模型与外部来源（如知识库或环境）交互并收集额外信息。我们将该方法称为 **ReAct**，并将其应用于多种语言和决策任务，证明其相较最先进基线更有效，同时提升了人类可解释性和可信度。

具体而言，在问答（HotpotQA）和事实核查（Fever）任务上，ReAct 通过与一个简单的 Wikipedia API 交互，克服了思维链推理中常见的幻觉和错误传播问题，并生成比无推理轨迹基线更可解释的、类人的任务解决轨迹。此外，在两个交互式决策基准（ALFWorld 和 WebShop）上，ReAct 仅使用一个或两个上下文示例进行提示，就以绝对成功率分别超过模仿学习和强化学习方法 34% 和 10%。

---

## 1 引言

人类智能的一个独特特征是能够将任务导向的行动与语言推理（或内部言语）无缝结合。这种能力被认为在人类认知中发挥重要作用，支持自我调节或策略化，并维持工作记忆。以在厨房做菜为例：在两个具体动作之间，我们可能用语言推理来跟踪进度（“现在所有东西都切好了，我应该把锅里的水加热”），处理异常或根据情况调整计划（“我没有盐，所以用酱油和胡椒代替”），以及意识到何时需要外部信息（“我怎么做面团？让我上网搜一下”）。我们也可能采取行动（打开食谱阅读、打开冰箱、检查食材）来支持推理并回答问题（“我现在能做什么菜？”）。这种“行动”和“推理”之间的紧密协同使人类能够快速学习新任务，并在先前未见的情况或信息不确定时进行稳健决策或推理。

近期结果暗示了在自主系统中结合语言推理与交互式决策的可能性。一方面，经过适当提示的大语言模型已展现出涌现能力，可在算术、常识和符号推理任务中执行多步推理轨迹以从问题推导答案。然而，这种“思维链”推理是一个静态黑箱：模型使用自身内部表示生成思考，并未扎根于外部世界，这限制了其进行反应式推理或更新知识的能力。这可能导致事实幻觉和推理过程中的错误传播（图 1(1b)）。另一方面，近期工作探索使用预训练语言模型在交互环境中进行规划和行动，重点是通过语言先验预测动作。这些方法通常将多模态观察转换为文本，使用语言模型生成领域特定动作或计划，然后使用控制器选择或执行它们。然而，除了 Huang 等人（2022b）执行有限形式的语言推理以重述当前状态的空间事实外，它们并未使用语言模型对高层目标进行抽象推理或维护支持行动的工作记忆。除了与少量方块交互的简单具身任务外，尚无研究探讨如何以协同方式结合推理和行动进行通用任务解决，以及这种结合是否能相较单独推理或单独行动带来系统性收益。

在本工作中，我们提出 **ReAct**，一种结合语言模型推理和行动的通用范式，用于解决多样化的语言推理和决策任务（图 1）。ReAct 提示 LLMs 以交错方式生成与任务相关的语言推理轨迹和动作，使模型能够执行动态推理来创建、维护和调整高层行动计划（推理以行动），同时与外部环境（如 Wikipedia）交互，将额外信息纳入推理（行动以推理）。

我们在四个多样化基准上对 ReAct 和最先进基线进行实证评估：问答（HotPotQA）、事实核查（Fever）、文本游戏（ALFWorld）和网页导航（WebShop）。对于 HotpotQA 和 Fever，在模型可交互的 Wikipedia API 访问下，ReAct 优于普通动作生成模型，并与思维链推理（CoT）具有竞争力。总体最佳方法是 ReAct 与 CoT 的结合，允许推理过程中同时使用内部知识和外部获取的信息。在 ALFWorld 和 WebShop 上，即使只有两次甚至一次 ReAct 提示，也能超过用 \(10^3 \sim 10^5\) 个任务实例训练的模仿学习或强化学习方法，成功率绝对提升分别为 34% 和 10%。我们还展示了稀疏、多样推理在决策中的重要性，证明其相较仅动作的受控基线具有一致优势。除通用适用性和性能提升外，推理与行动的结合还在所有领域中促进模型可解释性、可信度和可诊断性：人类可以容易地区分模型内部知识与外部环境信息，并检查推理轨迹以理解模型动作的决策依据。

总结而言，我们的主要贡献如下：

1. 我们引入 ReAct，一种新的基于提示的范式，用于在语言模型中协同推理和行动以进行通用任务解决；
2. 我们在多样基准上进行大量实验，展示 ReAct 在少样本学习设置中相较仅执行推理或仅生成动作的先前方法的优势；
3. 我们提供系统消融和分析，以理解行动在推理任务中的重要性，以及推理在交互任务中的重要性；
4. 我们分析 ReAct 在提示设置下的局限性（即对推理和行动行为的有限支持），并进行初步微调实验，显示 ReAct 可通过额外训练数据改进的潜力。

将 ReAct 扩展到更多任务的训练和运行，并与强化学习等互补范式结合，可进一步释放大语言模型的潜力。

---

## 2 ReAct：协同推理 + 行动

考虑一个智能体与环境交互进行任务解决的一般设置。在时间步 \(t\)，智能体从环境接收观察 \(o_t \in O\)，并根据某个策略 \(\pi(a_t|c_t)\) 采取动作 \(a_t \in A\)，其中 \(c_t = (o_1,a_1,\cdots,o_{t-1},a_{t-1},o_t)\) 是智能体的上下文。当映射 \(c_t \rightarrow a_t\) 高度隐式且需要大量计算时，学习策略具有挑战性。例如，图 1(1c) 中的智能体无法生成正确的最终动作（Act 4）来完成问答任务，因为它需要对轨迹上下文（问题、Act 1-3、Obs 1-3）进行复杂推理。类似地，图 1(2a) 中的智能体无法从上下文理解 sinkbasin 1 不包含 peppershaker 1，因此不断产生幻觉动作。

ReAct 的思想很简单：我们扩展智能体的动作空间为 \(\hat{A}=A\cup L\)，其中 \(L\) 是语言空间。语言空间中的动作 \(\hat{a}_t \in L\) 被称为思考或推理轨迹，它不影响外部环境，因此不产生观察反馈。相反，思考 \(\hat{a}_t\) 旨在通过对当前上下文 \(c_t\) 进行推理来组合有用信息，并更新上下文 \(c_{t+1}=(c_t,\hat{a}_t)\)，以支持未来推理或行动。如图 1 所示，有用的思考有多种类型，例如分解任务目标并创建行动计划、注入与任务解决相关的常识知识、从观察中提取重要部分、跟踪进度并转换行动计划、处理异常并调整行动计划等。

然而，由于语言空间 \(L\) 是无限的，在这个增强动作空间中学习很困难，并且需要强语言先验。本文主要关注这样的设置：冻结的大语言模型 PaLM-540B 通过少样本上下文示例进行提示，生成领域特定动作和自由形式语言思考以解决任务（图 1(1d)、(2b)）。每个上下文示例都是人类解决任务实例的动作、思考和環境观察轨迹（见附录 C）。对于推理至关重要的任务（图 1(1)），我们交替生成思考和动作，使任务解决轨迹由多个思考-动作-观察步骤组成。相反，对于可能涉及大量动作的决策任务（图 1(2)），思考只需在轨迹中最相关的位置稀疏出现，因此我们让语言模型自行决定思考和动作的异步发生。

由于决策和推理能力集成在大语言模型中，ReAct 具有若干独特特性：

- **直观且易于设计**：设计 ReAct 提示很直接，人类标注者只需在采取的动作之上用语言写下思考。本文未使用临时格式选择、思考设计或示例选择。
- **通用且灵活**：由于灵活的思考空间和思考-动作发生格式，ReAct 适用于具有不同动作空间和推理需求的各种任务，包括但不限于问答、事实核查、文本游戏和网页导航。
- **高性能且稳健**：ReAct 仅从一到六个上下文示例学习，就对新任务实例表现出强泛化能力，并在不同领域持续优于仅推理或仅行动的基线。
- **与人类对齐且可控**：ReAct 提供可解释的序列决策和推理过程，人类可以轻松检查推理和事实正确性。此外，人类还可以通过编辑思考在过程中控制或纠正智能体行为（第 4 节图 5）。

---

## 3 知识密集型推理任务

我们从知识密集型推理任务开始，如多跳问答和事实核查。如图 1(1d) 所示，通过与 Wikipedia API 交互，ReAct 能够检索信息以支持推理，同时使用推理来定位下一步检索什么，展示了推理和行动的协同。

### 3.1 设置

**领域** 我们考虑两个挑战知识检索和推理的数据集：

1. **HotpotQA**：多跳问答基准，需要对两个或更多 Wikipedia 段落进行推理。
2. **FEVER**：事实核查基准，每条声明被标注为 SUPPORTS、REFUTES 或 NOT ENOUGH INFO，取决于是否存在 Wikipedia 段落可验证该声明。

在本工作中，我们在这两个任务上采用仅问题设置：模型只接收问题/声明作为输入，无法访问支持段落，必须依赖内部知识或通过与外部环境交互检索知识来支持推理。

**动作空间** 我们设计一个简单的 Wikipedia Web API，支持三类交互式信息检索动作：

1. `search[entity]`：如果对应实体维基页面存在，返回该页面前 5 句话；否则从 Wikipedia 搜索引擎建议前 5 个相似实体。
2. `lookup[string]`：返回页面中包含字符串的下一句话，模拟浏览器 Ctrl+F 功能。
3. `finish[answer]`：以答案结束当前任务。

我们指出，该动作空间大多只能基于精确段落名称检索段落的一小部分，显著弱于最先进的词汇或神经检索器。目的是模拟人类如何与 Wikipedia 交互，并迫使模型通过显式语言推理进行检索。

### 3.2 方法

**ReAct 提示** 对于 HotpotQA 和 Fever，我们分别从训练集中随机选择 6 个和 3 个案例，并手动编写 ReAct 格式轨迹作为提示中的少样本示例。与图 1(d) 类似，每条轨迹由多个思考-动作-观察步骤组成（即密集思考），自由形式思考用于多种目的。具体而言，我们组合使用以下思考：分解问题（“我需要搜索 x，找到 y，然后找到 z”）、从 Wikipedia 观察中提取信息（“x 始于 1844 年”，“该段落没有说明 x”）、进行常识推理（“x 不是 y，所以 z 必须是……”）或算术推理（“1844 < 1989”）、指导搜索重述（“也许我可以改为搜索/查找 x”）、以及综合最终答案（“……所以答案是 x”）。详见附录 C。

**基线** 我们系统消融 ReAct 轨迹以构建多个基线的提示（格式如图 1(1a-1c)）：

- **Standard prompting（Standard）**：移除 ReAct 轨迹中的所有思考、动作、观察。
- **Chain-of-thought prompting（CoT）**：移除动作和观察，作为仅推理基线。
- **自一致性基线（CoT-SC）**：在推理时以解码温度 0.7 采样 21 条 CoT 轨迹并采用多数答案，发现其持续优于 CoT。
- **仅行动提示（Act）**：移除 ReAct 轨迹中的思考，大致类似 WebGPT 与互联网交互回答问题的方式，尽管其任务和动作空间不同，并使用模仿和强化学习而非提示。

**结合内部和外部知识** 如第 3.3 节详述，我们观察到 ReAct 展示的问题解决过程更事实性和扎根，而 CoT 在构建推理结构方面更准确，但容易遭受幻觉事实或思考。因此，我们提出结合 ReAct 和 CoT-SC，并让模型根据以下启发式决定何时切换到另一方法：

- **ReAct → CoT-SC**：当 ReAct 在给定步数内未能返回答案时，回退到 CoT-SC。HotpotQA 和 FEVER 分别设置 7 步和 5 步。
- **CoT-SC → ReAct**：当 \(n\) 个 CoT-SC 样本中的多数答案出现少于 \(n/2\) 次（即内部知识可能不支持任务）时，回退到 ReAct。

**微调** 由于大规模手动标注推理轨迹和动作具有挑战性，我们考虑类似 Zelikman 等人（2022）的自举方法，使用 ReAct 生成的 3,000 条正确答案轨迹（也用于其他基线）微调较小语言模型（PaLM-8/62B），以根据输入问题/声明解码轨迹（所有思考、动作、观察）。更多细节见附录 B.1。

### 3.3 结果与观察

**ReAct 持续优于 Act** 表 1 显示以 PaLM 540B 为基础模型时 HotpotQA 和 Fever 的提示结果。我们注意到 ReAct 在两个任务上都优于 Act，证明推理指导行动的价值，尤其是综合最终答案时（图 1(1c-d)）。微调结果也确认推理轨迹对更有信息量的行动有益。

**表 1：PaLM-540B 在 HotpotQA 和 Fever 上的提示结果**

| 提示方法       | HotpotQA (EM) | Fever (Acc) |
| -------------- | ------------: | ----------: |
| Standard       |          28.7 |        57.1 |
| CoT            |          29.4 |        56.3 |
| CoT-SC         |          33.4 |        60.4 |
| Act            |          25.7 |        58.9 |
| ReAct          |          27.4 |        60.9 |
| CoT-SC → ReAct |          34.2 |        64.6 |
| ReAct → CoT-SC |          35.1 |        62.0 |
| 监督式 SoTA    |          67.5 |        89.5 |

**ReAct 与 CoT 对比** 另一方面，ReAct 在 Fever 上优于 CoT（60.9 vs. 56.3），在 HotpotQA 上略落后于 CoT（27.4 vs. 29.4）。Fever 中 SUPPORTS/REFUTES 声明可能只差很小程度（见附录 D.1），因此采取行动检索准确且最新的知识至关重要。为更好理解 ReAct 和 CoT 在 HotpotQA 上的行为差异，我们分别从 ReAct 和 CoT 中随机采样 50 条正确和错误答案轨迹（共 200 例），并手动标注其成功和失败模式（表 2）。关键观察如下：

- **A）幻觉是 CoT 的严重问题**，导致其成功模式中的假阳性率远高于 ReAct（14% vs. 6%），并构成其主要失败模式（56%）。相反，ReAct 的问题解决轨迹更扎根、事实驱动且可信，得益于外部知识库访问。
- **B）** 虽然交错推理、动作和观察步骤提高了 ReAct 的扎根性和可信度，但这种结构约束也降低了其制定推理步骤的灵活性，导致推理错误率高于 CoT。我们注意到 ReAct 有一个频繁错误模式：模型重复生成之前的思考和动作，我们将其归类为“推理错误”的一部分，因为模型未能推理出正确的下一步动作并跳出循环。
- **C）** 对 ReAct 而言，通过搜索成功检索信息性知识至关重要。无信息搜索占错误案例的 23%，会使模型推理脱轨，并难以恢复和重新制定思考。这可能是事实性与灵活性之间的预期权衡，促使我们提出结合两种方法的策略。

**表 2：ReAct 和 CoT 在 HotpotQA 上的成功与失败模式类型及百分比**

| 类型               | 定义                                   | ReAct |  CoT |
| ------------------ | -------------------------------------- | ----: | ---: |
| 成功：真阳性       | 正确推理轨迹和事实                     |   94% |  86% |
| 成功：假阳性       | 幻觉推理轨迹或事实                     |    6% |  14% |
| 失败：推理错误     | 错误推理轨迹（包括未能从重复步骤恢复） |   47% |  16% |
| 失败：搜索结果错误 | 搜索返回空或不含有用信息               |   23% |  56% |
| 失败：幻觉         | 幻觉推理轨迹或事实                     |    0% |  28% |
| 失败：标签歧义     | 预测正确但未精确匹配标签               |   29% |    — |

**ReAct+CoT-SC 在提示 LLMs 时表现最佳** 表 1 还显示，HotpotQA 和 Fever 上最佳提示方法分别是 ReAct → CoT-SC 和 CoT-SC → ReAct。此外，图 2 显示不同方法随 CoT-SC 样本数变化的表现。虽然两种 ReAct + CoT-SC 方法各在一个任务上占优，但它们在不同样本数下都显著且持续优于 CoT-SC，仅用 3-5 个样本就达到 21 个样本的 CoT-SC 性能。这些结果表明适当结合模型内部知识和外部知识对推理任务的价值。

**ReAct 在微调中表现最佳** 图 3 显示在 HotpotQA 上提示/微调四种方法（Standard、CoT、Act、ReAct）的规模效应。对于 PaLM-8/62B，提示 ReAct 在四种方法中表现最差，因为从上下文示例学习推理和行动都很难。然而，仅用 3,000 个示例微调后，ReAct 成为四种方法中最佳：PaLM-8B 微调 ReAct 超过所有 PaLM-62B 提示方法，PaLM-62B 微调 ReAct 超过所有 540B 提示方法。相反，微调 Standard 或 CoT 对 PaLM 8/62B 都显著差于微调 ReAct 或 Act，因为前者本质上教模型记忆（可能幻觉的）知识事实，后者教模型如何（推理并）行动以从 Wikipedia 访问信息，这是更具泛化性的知识推理技能。由于所有提示方法仍显著落后于领域特定最先进方法（表 1），我们认为用更多人工编写数据微调可能是释放 ReAct 潜力的更好方式。

---

## 4 决策任务

我们还在两个基于语言的交互式决策任务 ALFWorld 和 WebShop 上测试 ReAct，这两个任务都具有复杂环境，要求智能体在长时程和稀疏奖励下行动，因此需要推理以有效行动和探索。

**ALFWorld** ALFWorld 是一个合成文本游戏，设计用于与具身 ALFRED 基准对齐。它包括 6 类任务，智能体需要通过导航和与模拟家庭交互（文本动作，如 `go to coffeetable 1`、`take paper 2`、`use desklamp 1`）来实现高层目标（例如在台灯下检查纸张）。一个任务实例可能有超过 50 个位置，专家策略需要超过 50 步才能解决，因此挑战智能体规划并跟踪子目标，以及系统探索（例如逐个检查所有桌子寻找台灯）。特别是，ALFWorld 内置一个挑战：需要确定常见家居物品的可能位置（例如台灯可能在桌子、架子或梳妆台上），使该环境适合 LLMs 利用预训练常识知识。为提示 ReAct，我们为每种任务类型从训练集中随机标注三条轨迹，每条轨迹包含稀疏思考：(1) 分解目标，(2) 跟踪子目标完成，(3) 确定下一个子目标，(4) 通过常识推理在哪里找到物体以及如何处理它。ALFWorld 提示见附录 C.4。遵循 Shridhar 等人（2020b），我们在 134 个未见评估游戏上以任务特定设置评估。为稳健性，我们通过从标注的 3 条轨迹中取 2 条的所有排列，为每种任务类型构建 6 个提示。Act 提示使用相同轨迹但无思考——由于任务实例从训练集随机选择，这对 ReAct 和 Act 都不偏袒，并提供公平受控比较以测试稀疏思考的重要性。基线使用 BUTLER，一种为每种任务类型在 \(10^5\) 条专家轨迹上训练的模仿学习智能体。

**WebShop** ReAct 能否与噪声真实世界语言环境交互用于实际应用？我们研究 WebShop，一个最近提出的在线购物网站环境，包含 118 万真实产品和 1.2 万人类指令。与 ALFWorld 不同，WebShop 包含高度多样化的结构化和非结构化文本（例如从 Amazon 爬取的产品标题、描述和选项），要求智能体根据用户指令（例如“我在找带抽屉的床头柜。它应有镍色饰面，价格低于 140 美元”）通过网页交互（例如搜索“nightstand drawers”、选择“color: modern-nickel-white”或“back to search”等按钮）购买产品。该任务在 500 条测试指令上按平均分数（所选产品覆盖期望属性的百分比）和成功率（所选产品满足所有要求的回合百分比）评估。我们为 Act 提示制定搜索、选择产品、选择选项和购买动作，ReAct 提示额外推理以确定探索什么、何时购买以及哪些产品选项与指令相关。示例提示见表 6，模型预测见附录表 10。我们与模仿学习（IL）方法比较，该方法用 1,012 条人工标注轨迹训练；以及模仿+强化学习（IL+RL）方法，额外用 10,587 条训练指令训练。

**结果** ReAct 在 ALFWorld（表 3）和 WebShop（表 4）上都优于 Act。在 ALFWorld 上，最佳 ReAct 试验达到平均成功率 71%，显著超过最佳 Act（45%）和 BUTLER（37%）试验。事实上，即使较差的 ReAct 试验（48%）也超过两种方法的最佳试验。此外，ReAct 相对 Act 的优势在六次受控试验中一致，相对性能增益从 33% 到 90%，平均 62%。定性上，我们看到完全没有思考时，Act 无法正确将目标分解为更小子目标，或失去对当前环境状态的跟踪。ReAct 和 Act 的示例轨迹见附录 D.2.1 和 D.2.2。

在 WebShop 上，一次性 Act 提示已经与 IL 和 IL+RL 方法相当。加入稀疏推理后，ReAct 取得显著更好性能，成功率绝对提升 10%。检查示例发现，ReAct 更可能通过推理弥合噪声观察和动作之间的差距，识别与指令相关的产品和选项（例如“对于‘space-saving ottoman bench for living room’，该商品有‘39x18x18inch’和‘blue’选项，似乎不错，可以购买。”）。然而，现有方法仍远低于专家人类性能（表 4），专家人类进行显著更多产品探索和查询重述，这对基于提示的方法仍具挑战性。

**表 3：ALFWorld 任务特定成功率（%）**

| 方法                  | Pick | Clean | Heat | Cool | Look | Pick2 |  All |
| --------------------- | ---: | ----: | ---: | ---: | ---: | ----: | ---: |
| Act（best of 6）      |   88 |    42 |   74 |   67 |   72 |    41 |   45 |
| ReAct（avg）          |   65 |    39 |   83 |   76 |   55 |    24 |   57 |
| ReAct（best of 6）    |   92 |    58 |   96 |   86 |   78 |    41 |   71 |
| ReAct-IM（avg）       |   55 |    59 |   60 |   55 |   23 |    24 |   48 |
| ReAct-IM（best of 6） |   62 |    68 |   87 |   57 |   39 |    33 |   53 |
| BUTLERg（best of 8）  |   33 |    26 |   70 |   76 |   17 |    12 |   22 |
| BUTLER（best of 8）   |   46 |    39 |   74 |  100 |   22 |    24 |   37 |

**表 4：WebShop 上的分数和成功率（SR）**

| 方法  | Score |   SR |
| ----- | ----: | ---: |
| Act   |  62.3 | 30.1 |
| ReAct |  66.6 | 40.0 |
| IL    |  59.9 | 29.1 |
| IL+RL |  62.4 | 28.7 |
| Human |  82.1 | 59.6 |

**内部推理与外部反馈的价值** 据我们所知，ReAct 是首次展示在闭环系统中使用 LLM 将推理和行动结合应用于交互环境。最接近的先前工作可能是 Inner Monologue（IM），其中具身智能体的动作由同名“内心独白”驱动。然而，IM 的“内心独白”仅限于环境状态观察和智能体为实现目标需要完成的内容。相反，ReAct 在决策中的推理轨迹灵活且稀疏，允许为不同任务诱导多种推理类型（见第 2 节）。

为展示 ReAct 与 IM 的差异，并强调内部推理与简单对外部反馈反应的重要性，我们使用由 IM 式密集外部反馈组成的思考模式进行消融实验。如表 3 所示，ReAct 显著优于 IM 式提示（ReAct-IM）（总体成功率 71 vs. 53），在六项任务中的五项上具有一致优势。定性上，我们观察到 ReAct-IM 经常错误判断子目标何时完成，或下一个子目标应是什么，因为缺乏高层目标分解。此外，许多 ReAct-IM 轨迹难以确定物品在 ALFWorld 环境中可能在哪里，因为缺乏常识推理。这两个缺点都可以在 ReAct 范式中解决。ReAct-IM 更多细节见附录 B.2，示例提示见附录 C.4，示例轨迹见附录 D.2.3。

---

## 5 相关工作

**用于推理的语言模型** 最著名的使用 LLMs 进行推理的工作可能是思维链（CoT），它揭示了 LLMs 制定自己的“思考过程”以解决问题的能力。此后有多项后续工作，包括用于解决复杂任务的最小到最多提示、零样本 CoT、以及带自一致性的推理。最近，Madaan & Yazdanbakhsh 系统研究了 CoT 的表述和结构，并观察到符号、模式和文本的存在对 CoT 有效性至关重要。其他工作还扩展到简单提示之外的更复杂推理架构。例如 Selection-Inference 将推理过程分为“选择”和“推断”两步。STaR 通过在模型自身生成的正确理由上微调模型来自举推理过程。Faithful reasoning 将多步推理分解为三步，每步分别由专用 LM 执行。类似方法 Scratchpad 在中间计算步骤上微调 LM，也在多步计算问题上表现出改进。与这些方法相比，ReAct 不仅执行孤立、固定的推理，还将模型动作及其对应观察集成到连贯输入流中，使模型更准确推理并处理推理之外的任务（例如交互式决策）。

**用于决策的语言模型** LLMs 的强大能力使其能够执行语言生成之外的任务，利用 LLMs 作为决策策略模型尤其在交互环境中越来越流行。WebGPT 使用 LM 与网页浏览器交互、浏览网页并从 ELI5 中推断复杂问题答案。与 ReAct 相比，WebGPT 不显式建模思考和推理过程，而是依赖昂贵的人类反馈进行强化学习。在对话建模中，BlenderBot 和 Sparrow 等聊天机器人以及 SimpleTOD 等任务导向对话系统也训练 LM 做出 API 调用决策。与 ReAct 不同，它们也不显式考虑推理过程，并依赖昂贵数据集和人类反馈收集进行策略学习。相反，ReAct 以更便宜的方式学习策略，因为决策过程只需要推理过程的语言描述。

LLMs 也越来越多地被用于交互和具身环境中的规划和决策。与此最相关的可能是 SayCan 和 Inner Monologue，它们使用 LLMs 进行机器人动作规划和决策。在 SayCan 中，LLMs 被提示直接预测机器人可采取的可能动作，然后由基于视觉环境的可供性模型重新排序以进行最终预测。Inner Monologue 通过添加同名“内心独白”进一步改进，其实现为从环境注入反馈。据我们所知，Inner Monologue 是首个展示此类闭环系统的工作，ReAct 建立在其上。然而，我们认为 Inner Monologue 并不真正包含内部思考——第 4 节对此进行了详述。我们还注意到，在交互决策过程中利用语言作为语义丰富输入已在其他设置中成功。越来越明显的是，借助 LLMs，语言作为基本认知机制将在交互和决策中发挥关键作用。此外，LLMs 的进展也启发了通用智能体的发展。

---

## 6 结论

我们提出了 ReAct——一种简单而有效的方法，用于在大语言模型中协同推理和行动。通过对多跳问答、事实核查和交互式决策任务的多样实验，我们表明 ReAct 带来更优性能和可解释的决策轨迹。尽管方法简单，具有大动作空间的复杂任务需要更多演示才能学好，但这很容易超出上下文学习的输入长度限制。我们在 HotpotQA 上探索了微调方法并取得初步有希望的结果，但从更多高质量人工标注中学习将是进一步提高性能的期望。将 ReAct 扩展到多任务训练，并与强化学习等互补范式结合，可能产生更强的智能体，进一步释放 LLMs 在更多应用中的潜力。

---

## 致谢

我们感谢 Google Brain 团队和普林斯顿 NLP 组许多人的支持和反馈。本工作部分得到美国国家科学基金会 Grant No. 2107048 支持。本材料中表达的任何意见、发现、结论或建议均为作者观点，不一定反映美国国家科学基金会的观点。

## 可复现性声明

我们的主要实验在 PaLM 上进行，该模型尚未公开可用。为提高可复现性，我们在附录 C 中包含所有使用的提示，在附录 A.1 中包含使用 GPT-3 的额外实验，并在 https://anonymous.4open.science/r/ReAct-2268/ 提供相关 GPT-3 ReAct 提示代码。

## 伦理声明

ReAct 提示大语言模型生成比先前方法更人类可解释、可诊断和可控的任务解决轨迹。然而，将大语言模型与动作空间连接以与外部环境（例如网络、物理环境）交互存在潜在危险，例如查找不当或私人信息，或在环境中采取有害行动。我们的实验通过将交互限制在特定网站（Wikipedia 或 WebShop）来最小化此类风险，这些网站不含私人信息，动作空间设计中没有危险动作（即模型无法在 WebShop 研究基准中真正购买产品，也无法编辑 Wikipedia）。我们认为研究人员在未来设计更广泛实验之前应意识到此类风险。

---

## 附录

### A 额外结果

#### A.1 GPT-3 实验

**表 5：使用 PaLM-540B 与 GPT-3 的 ReAct 提示结果**

| 任务                 | PaLM-540B | GPT-3 |
| -------------------- | --------: | ----: |
| HotpotQA（精确匹配） |      29.4 |  30.8 |
| ALFWorld（成功率%）  |      70.9 |  78.4 |

我们在 HotpotQA 上随机采样 500 个验证问题的子集。在 ALFWorld 上，我们使用全部 134 个未见验证任务实例，并使用根据 PaLM-540B 的最佳提示集。

我们运行额外 GPT-3 实验以确认 ReAct 提示性能在不同大语言模型上具有通用性。如表 5 所示，GPT-3（text-davinci-002，贪婪解码）在 HotpotQA 和 ALFWorld 上持续优于 PaLM-540B，可能是因为它用人类指令跟随进行了微调。这表明 ReAct 提示在不同大语言模型和不同任务上有效。这些实验代码见 https://react-lm.github.io/。

#### A.2 ReAct 在 HotpotQA 上获得最新知识

在轨迹检查中，我们还发现有时 ReAct 与数据集标签不一致，因为标签本身可能过时。例如，图 4 中问题询问酒店规模，该规模自 HotpotQA 构建以来已增加。Standard 和 CoT 因幻觉给出错误答案，Act 尽管可访问真实世界网络交互，但因缺乏指导如何与互联网交互进行问答的推理而失败。只有 ReAct 能够从互联网检索最新信息并提供合理答案。因此，更好地结合推理能力可能有益于最近的互联网增强语言模型进行最新任务解决。

#### A.3 ALFWorld 中的人类在环行为纠正

我们还探索了与 ReAct 的人类在环交互，允许人类检查和编辑 ReAct 的推理轨迹。图 5 显示，通过简单删除 Act 17 中的幻觉句子并在 Act 23 中添加一些提示，ReAct 可以被引导大幅改变行为以对齐这些人类思考编辑并成功完成任务。从人类角度看，解决此类任务变得显著更容易：从输入几十个动作变为仅编辑几个思考，这开启了新形式的人机协作。我们指出，这种即时策略编辑对 Act 和先前 RL 方法很困难，因为人类不能改变模型参数，改变几个动作可能无法编辑模型其余行为。该范式也不仅仅是像 Huang 等人（2022b）中通过人类对话更新目标或子目标——编辑 ReAct 思考可以做到这些，还可以修改模型内部信念、推理风格或灵活思考空间支持的任何内容，以更好地解决任务。我们认为这是人类对齐的一个激动方向，并将更系统研究留作未来工作。

### B 实验细节

#### B.1 HotpotQA 微调细节

所有微调使用批量大小 64。在 PaLM-8B 上，我们微调 ReAct 和 Act 方法 4,000 步，Standard 和 CoT 方法 2,000 步。在 PaLM-62B 上，我们微调 ReAct 和 Act 方法 4,000 步，Standard 和 CoT 方法 1,000 步。我们发现 ReAct 和 Act 方法通常受益于更多训练步数（和更多训练数据），而 Standard 和 CoT 方法在微调后很快退化。

#### B.2 ALFWorld IM 式细节

对于 IM 式消融，使用与 ReAct 相同的专家轨迹重新标注密集外部反馈思考，这些思考限制 ReAct-IM 只能思考：(1) 分解当前目标；(2) 需要完成的当前子目标。特别是，ReAct-IM 缺少以下思考：(1) 确定子目标何时完成；(2) 确定下一个子目标应是什么；(3) 诱导 LLM 参考其内部预训练知识以识别物品在环境中可能的位置。

### C 提示

#### C.1 HotpotQA

原始问题示例：

- 问题：科罗拉多造山带东段延伸区域的海拔范围是多少？答案：1,800 到 7,000 英尺
- 问题：音乐家和讽刺作家 Allie Goertz 写了一首关于《辛普森一家》角色 Milhouse 的歌，Matt Groening 以谁的名字命名 Milhouse？答案：Richard Nixon
- 问题：哪部纪录片关于芬兰摇滚乐队，是 Adam Clayton Powell 还是 The Saimaa Gesture？答案：The Saimaa Gesture
- 问题：Nicholas Ray 和 Elia Kazan 共同从事什么职业？答案：导演、编剧、演员
- 问题：哪本杂志先创办，Arthur’s Magazine 还是 First for Women？答案：Arthur’s Magazine
- 问题：Pavel Urysohn 和 Leonid Levin 是否以同一类型工作闻名？答案：是

**Act 提示示例（节选）**

```text
问题：科罗拉多造山带东段延伸区域的海拔范围是多少？
动作 1：Search[Colorado orogeny]
观察 1：科罗拉多造山带是科罗拉多及周边地区的一次造山事件。
动作 2：Lookup[eastern sector]
观察 2：（结果 1/1）东段延伸到高平原，被称为中央平原造山带。
动作 3：Search[High Plains]
观察 3：High Plains 指两个不同陆地区域之一：
动作 4：Search[High Plains (United States)]
观察 4：高平原是大平原的子区域。从东到西，高平原海拔从约 1,800 到 7,000 英尺（550 到 2,130 米）。
动作 5：Finish[1,800 to 7,000 ft]
```

**CoT 提示示例（节选）**

```text
问题：科罗拉多造山带东段延伸区域的海拔范围是多少？
思考：让我们逐步思考。科罗拉多造山带东段延伸到高平原。高平原海拔从约 1,800 到 7,000 英尺，所以答案是 1,800 到 7,000 英尺。
答案：1,800 到 7,000 英尺
```

**ReAct 提示示例（节选）**

```text
问题：科罗拉多造山带东段延伸区域的海拔范围是多少？
思考 1：我需要搜索 Colorado orogeny，找到科罗拉多造山带东段延伸的区域，然后找到该区域的海拔范围。
动作 1：Search[Colorado orogeny]
观察 1：科罗拉多造山带是科罗拉多及周边地区的一次造山事件。
思考 2：它没有提到东段。所以我需要查找 eastern sector。
动作 2：Lookup[eastern sector]
观察 2：（结果 1/1）东段延伸到高平原，被称为中央平原造山带。
思考 3：科罗拉多造山带东段延伸到高平原。所以我需要搜索 High Plains 并找到其海拔范围。
动作 3：Search[High Plains]
观察 3：High Plains 指两个不同陆地区域之一。
思考 4：我需要改为搜索 High Plains (United States)。
动作 4：Search[High Plains (United States)]
观察 4：高平原是大平原的子区域。从东到西，高平原海拔从约 1,800 到 7,000 英尺（550 到 2,130 米）。
思考 5：高平原海拔从约 1,800 到 7,000 英尺，所以答案是 1,800 到 7,000 英尺。
动作 5：Finish[1,800 to 7,000 ft]
```

#### C.2 FEVER

**Act 提示示例（节选）**

```text
判断是否存在 Observation 支持或反驳 Claim，或信息不足。
Claim：Nikolaj Coster-Waldau 曾与 Fox Broadcasting Company 合作。
动作 1：Search[Nikolaj Coster-Waldau]
观察 1：Nikolaj William Coster-Waldau（生于 1970 年 7 月 27 日）是丹麦演员和制片人。他毕业于丹麦国家表演艺术学院……他在 Fox 电视系列剧 New Amsterdam（2008）中扮演侦探，并出现在 2009 年 Fox 电视电影 Virtuality 中。
动作 2：Finish[SUPPORTS]

Claim：Stranger Things 设定在印第安纳州 Bloomington。
动作 1：Search[Stranger Things]
观察 1：Stranger Things 是一部美国科幻恐怖剧情电视剧，由 Duffer Brothers 创作。故事设定在 1980 年代，主要发生在虚构的印第安纳州 Hawkins 镇……
动作 2：Finish[REFUTES]
```

**CoT 提示示例（节选）**

```text
Claim：Nikolaj Coster-Waldau 曾与 Fox Broadcasting Company 合作。
思考：Nikolaj William Coster-Waldau 出现在 2009 年 Fox 电视电影 Virtuality 中，所以他曾与 Fox Broadcasting Company 合作。
答案：SUPPORTS

Claim：Stranger Things 设定在印第安纳州 Bloomington。
思考：Stranger Things 设定在虚构的印第安纳州 Hawkins 镇，而不是 Bloomington。
答案：REFUTES
```

**ReAct 提示示例（节选）**

```text
Claim：Nikolaj Coster-Waldau 曾与 Fox Broadcasting Company 合作。
思考 1：我需要搜索 Nikolaj Coster-Waldau，并查找他是否曾与 Fox Broadcasting Company 合作。
动作 1：Search[Nikolaj Coster-Waldau]
观察 1：……他出现在 2009 年 Fox 电视电影 Virtuality 中……
思考 2：因为他“出现在 2009 年 Fox 电视电影 Virtuality 中”，所以他应该曾与 Fox Broadcasting Company 合作。
动作 2：Finish[SUPPORTS]

Claim：Stranger Things 设定在印第安纳州 Bloomington。
思考 1：我应该搜索 Stranger Things，看看它是否设定在印第安纳州 Bloomington。
动作 1：Search[Stranger Things]
观察 1：……故事设定在虚构的印第安纳州 Hawkins 镇……
思考 2：观察说它设定在“虚构的印第安纳州 Hawkins 镇”，所以它不是设定在 Bloomington。
动作 2：Finish[REFUTES]
```

#### C.3 WebShop

**表 6：第 4 节 WebShop 实验使用的提示（左侧 Act，右侧 ReAct 加入推理）**

指令：我想要一瓶 3 盎司的明亮柑橘味敏感肌除臭剂，价格低于 50.00 美元。

Act 提示示例：

```text
Action: search 3 ounce bright citrus deodorant sensitive skin
Observation: Back to Search; Page 1 (Total results: 50); ...
B078GWRC1J Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce $10.99
...
Action: click B078GWRC1J
Observation: ... scent [assorted scents][bright citrus]... size [travel set (4-pack)][3 ounce (pack of 1)]...
Action: click[bright citrus]
Observation: You have clicked bright citrus.
Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).
Action: click[Buy Now]
```

ReAct 提示示例：

```text
Action: search 3 ounce bright citrus deodorant sensitive skin
Observation: ...
Action: think[B078GWRC1J 和 B078GTKVXY 是价格低于 50 美元的明亮柑橘味除臭剂。我可以先检查 B078GWRC1J。]
Observation: OK.
Action: click B078GWRC1J
Observation: ...
Action: think[对于 3 盎司瓶装明亮柑橘味敏感肌除臭剂，该商品有选项 'bright citrus' 和 '3 ounce (pack of 1)'，似乎不错，可以购买。]
Observation: OK.
Action: click[bright citrus]
Observation: You have clicked bright citrus.
Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).
Action: click[Buy Now]
```

#### C.4 ALFWorld

**表 7：ALFWorld clean 任务上的 Act 提示（无思考）**

```text
你在房间中央。快速环顾四周，你看到 cabinet 13, cabinet 12, ..., diningtable 1, ..., fridge 1, ..., sinkbasin 1, ...
你的任务是：put a clean lettuce in diningtable.
> go to fridge 1
The fridge 1 is closed.
> open fridge 1
You open the fridge 1. The fridge 1 is open. In it, you see a cup 3, a egg 2, a potato 3, and a potato 2.
> go to diningtable 1
On the diningtable 1, you see ... a lettuce 1, ...
> take lettuce 1 from diningtable 1
You pick up the lettuce 1 from the diningtable 1.
> go to sinkbasin 1
On the sinkbasin 1, you see a apple 2, a ladle 2, a spoon 1, and a tomato 3.
> clean lettuce 1 with sinkbasin 1
You clean the lettuce 1 using the sinkbasin 1.
> think: Now I clean a lettuce (1). Next, I need to put it in/on diningtable 1.
OK.
> go to diningtable 1
...
> put lettuce 1 in/on diningtable 1
You put the lettuce 1 in/on the diningtable 1.
```

**表 8：ALFWorld clean 任务上的 ReAct 提示**

```text
你在房间中央。...
你的任务是：put a clean lettuce in diningtable.
> think: 要解决任务，我需要找到并拿起一个生菜，然后用 sinkbasin 清洗，再放入 diningtable。
OK.
> think: 首先我需要找到生菜。生菜更可能出现于 fridge (1), diningtable (1), sinkbasin (1), stoveburner (1-3), cabinet (1-13)。我可以逐个检查，从 fridge 1 开始。
OK.
> go to fridge 1
The fridge 1 is closed.
> open fridge 1
You open the fridge 1. ...
> go to diningtable 1
On the diningtable 1, you see ... a lettuce 1, ...
> think: 现在我找到生菜 (1)。接下来，我需要拿起它。
OK.
> take lettuce 1 from diningtable 1
You pick up the lettuce 1 from the diningtable 1.
> think: 现在我拿起了生菜 (1)。接下来，我需要去 sinkbasin (1) 并清洗它。
OK.
> go to sinkbasin 1
...
> clean lettuce 1 with sinkbasin 1
You clean the lettuce 1 using the sinkbasin 1.
> think: 现在我清洗了生菜 (1)。接下来，我需要把它放入 diningtable 1。
OK.
> go to diningtable 1
...
> put lettuce 1 in/on diningtable 1
You put the lettuce 1 in/on the diningtable 1.
```

**表 9：ALFWorld clean 任务上的 ReAct-IM 提示**

```text
你在房间中央。...
你的任务是：put a clean lettuce in diningtable.
> think: 要解决任务，我需要找到并拿起一个生菜，然后用 sinkbasin 清洗，再放入 diningtable。
OK.
> think: 首先我需要找到一个生菜。
OK.
> go to fridge 1
The fridge 1 is closed.
> think: 首先我需要找到一个生菜。
OK.
> open fridge 1
...
> think: 首先我需要找到一个生菜。
OK.
> go to diningtable 1
On the diningtable 1, you see ... a lettuce 1, ...
> think: 首先我需要找到一个生菜。
OK.
> take lettuce 1 from diningtable 1
You pick up the lettuce 1 from the diningtable 1.
> think: 我需要用 sinkbasin 1 清洗这个生菜 (1)。
OK.
> go to sinkbasin 1
...
> clean lettuce 1 with sinkbasin 1
You clean the lettuce 1 using the sinkbasin 1.
> think: 我需要把这个清洗过的生菜 (1) 放入 diningtable 1。
OK.
> go to diningtable 1
...
> put lettuce 1 in/on diningtable 1
You put the lettuce 1 in/on the diningtable 1.
```

### D 轨迹

#### D.1 FEVER 轨迹

以下为 ReAct、CoT 和 Act 在 FEVER 开发集随机样本上的部分轨迹。搜索结果省略。

**示例 2491（真实标签：REFUTES）**

ReAct：

```text
Claim：Bermuda Triangle 是太平洋中一个定义宽松的区域。
思考 1：我需要搜索 Bermuda Triangle，并查找它是否在太平洋。
动作 1：Search[Bermuda Triangle]
观察 1：据报道闹鬼地点……
思考 2：观察说它位于北大西洋西部，所以它不在太平洋。
动作 2：Finish[REFUTES]
```

Act：

```text
Claim：Bermuda Triangle 是太平洋中一个定义宽松的区域。
动作 1：Search[Bermuda Triangle]
观察 1：据报道闹鬼地点……
动作 2：Finish[REFUTES]
```

CoT：

```text
Claim：Bermuda Triangle 是太平洋中一个定义宽松的区域。
思考：让我们逐步思考。Bermuda Triangle 是大西洋中一个定义宽松的区域。太平洋不是大西洋。
答案：REFUTES
```

**示例 5908（真实标签：SUPPORTS）**

ReAct：

```text
Claim：Princess Mononoke 是一部电影。
思考 1：我需要搜索 Princess Mononoke，并查找它是否是一部电影。
动作 1：Search[Princess Mononoke]
观察 1：Princess Mononoke……
思考 2：从观察看，它说 Princess Mononoke 是一部电影。
动作 2：Finish[SUPPORTS]
```

Act：

```text
Claim：Princess Mononoke 是一部电影。
动作 1：Search[Princess Mononoke]
观察 1：Princess Mononoke……
动作 2：Finish[SUPPORTS]
```

CoT：

```text
Claim：Princess Mononoke 是一部电影。
思考：让我们逐步思考。Princess Mononoke 是一部电影，而电影就是 film。
答案：SUPPORTS
```

**示例 1951（真实标签：REFUTES）**

ReAct：

```text
Claim：Soyuz 是美国太空计划的一部分。
思考 1：我需要搜索 Soyuz，并查找它是否是美国太空计划的一部分。
动作 1：Search[Soyuz]
观察 1：Soyuz 是音译……
思考 2：它没有提到美国太空计划，所以我需要搜索“American space program”以查找它是否是其一部分。
动作 2：Search[American space program]
观察 2：太空政策……
思考 3：它没有提到 Soyuz，所以我不确定该声明是否真实。
动作 3：Finish[NOT ENOUGH INFO]
```

Act：

```text
Claim：Soyuz 是美国太空计划的一部分。
动作 1：Search[Soyuz]
观察 1：Soyuz 是音译……
动作 2：Finish[NOT ENOUGH INFO]
```

CoT：

```text
Claim：Soyuz 是美国太空计划的一部分。
思考：让我们逐步思考。Soyuz 是俄罗斯飞船。美国太空计划是 NASA。NASA 和俄罗斯在国际空间站上合作过。
答案：SUPPORTS
```

**示例 3208（真实标签：REFUTES）**

ReAct：

```text
Claim：Reign Over Me 是一部 2010 年制作的美国电影。
思考 1：我需要搜索 Reign Over Me，并查找它是否是一部 2010 年制作的美国电影。
动作 1：Search[Reign Over Me]
观察 1：Reign Over Me……
思考 2：观察说它是一部 2007 年制作的美国电影，所以它不是 2010 年制作。
动作 2：Finish[REFUTES]
```

Act：

```text
Claim：Reign Over Me 是一部 2010 年制作的美国电影。
动作 1：Search[Reign Over Me]
动作 2：Finish[SUPPORTS]
```

CoT：

```text
Claim：Reign Over Me 是一部 2010 年制作的美国电影。
思考：让我们逐步思考。首先，Reign Over Me 是一部美国电影吗？是的。其次，它是 2010 年制作的吗？是的。
答案：SUPPORTS
```

#### D.2 ALFWorld 轨迹

以下为 ReAct、Act 和 ReAct-IM 在同一 ALFWorld 游戏 `pick_clean_then_place_in_recep-Knife-None-CounterTop-10/trial_T20190909_110445_675754` 上的示例轨迹。

##### D.2.1 ReAct 轨迹

ReAct 能正确找到刀、清洗它，然后放到柜台上。

```text
你在房间中央。快速环顾四周，你看到 cabinet 6, cabinet 5, ..., countertop 3, countertop 2, countertop 1, ..., sinkbasin 1, ...
你的任务是：put a clean knife in countertop.
> think: 要解决任务，我需要找到并拿起一把刀，然后用 sinkbasin 清洗，再放到 countertop。
OK.
> think: 首先我需要找到一把刀。刀更可能出现于 cabinet (1-6), drawer (1-3), countertop (1-3), fridge (1), garbagecan (1), shelf (1-3), sinkbasin (1), stoveburner (1-4), toaster (1)。我可以逐个检查，从 cabinet 1 开始。
OK.
> go to cabinet 1
On the cabinet 1, you see a bowl 1.
> go to cabinet 2
The cabinet 2 is closed.
> go to cabinet 3
On the cabinet 3, you see a glassbottle 1.
> go to cabinet 4
On the cabinet 4, you see a mug 1.
> go to cabinet 5
The cabinet 5 is closed.
> go to cabinet 6
The cabinet 6 is closed.
> go to drawer 1
The drawer 1 is closed.
> go to drawer 2
The drawer 2 is closed.
> go to drawer 3
The drawer 3 is closed.
> go to countertop 1
On the countertop 1, you see a lettuce 2, a mug 2, a peppershaker 1, and a spoon 2.
> go to countertop 2
On the countertop 2, you see a cup 1, a dishsponge 1, a glassbottle 3, a knife 1, a plate 2, a potato 3, and a statue 1.
> think: 现在我找到了一把刀 (1)。接下来，我需要拿起它。
OK.
> take knife 1 from countertop 2
You pick up the knife 1 from the countertop 2.
> think: 现在我拿起了刀 (1)。接下来，我需要去 sinkbasin (1) 并清洗它。
OK.
> go to sinkbasin 1
On the sinkbasin 1, you see a fork 3, a lettuce 3, and a spatula 2.
> clean knife 1 with sinkbasin 1
You clean the knife 1 using the sinkbasin 1.
> think: 现在我清洗了刀 (1)。接下来，我需要把它放到 countertop 1。
OK.
> go to countertop 1
On the countertop 1, you see a lettuce 2, a mug 2, a peppershaker 1, and a spoon 2.
> put knife 1 in/on countertop 1
You put the knife 1 in/on the countertop 1.
```

##### D.2.2 Act 轨迹

Act 能正确找到刀，但在实际去 sinkbasin 之前就尝试用 sinkbasin 清洗它。此后模型无法恢复，并卡在重复相同命令序列中。使用 ReAct 时，已有思考会具体识别刀已被拿起，以及下一个子目标是去 sinkbasin 清洗它。

```text
你在房间中央。...
你的任务是：put a clean knife in countertop.
> go to countertop 1
On the countertop 1, you see a lettuce 2, a mug 2, a peppershaker 1, and a spoon 2.
> go to countertop 2
On the countertop 2, you see a cup 1, a dishsponge 1, a glassbottle 3, a knife 1, a plate 2, a potato 3, and a statue 1.
> take knife 1 from countertop 2
You pick up the knife 1 from the countertop 2.
> go to countertop 1
On the countertop 1, you see a lettuce 2, a mug 2, a peppershaker 1, and a spoon 2.
> clean knife 1 with sinkbasin 1
Nothing happens.
> go to countertop 3
...
> take knife 2 from countertop 3
Nothing happens.
> go to countertop 2
...
> take knife 1 from countertop 2
Nothing happens.
> go to countertop 1
...
> clean knife 1 with sinkbasin 1
Nothing happens.
（重复）
```

##### D.2.3 ReAct-IM 轨迹

ReAct-IM 能正确找到刀，但也无法清洗它。LLM 随后卡在尝试把刀放到柜台上。一个可能问题是错误思考 `think: I need to find a clean knife.` 似乎欺骗 LLM 相信刀已经清洗。ReAct 中这一系列子目标会被每个思考具体捕获。

```text
你在房间中央。...
你的任务是：put a clean knife in countertop.
> think: 要解决任务，我需要找到并拿起一把干净的刀，然后放到 countertop。
OK.
> think: 我需要找到一把干净的刀。
OK.
> go to countertop 1
...
> think: 我需要找到一把干净的刀。
OK.
> go to countertop 2
On the countertop 2, you see ... a knife 1, ...
> think: 我需要找到一把干净的刀。
OK.
> take knife 1 from countertop 2
You pick up the knife 1 from the countertop 2.
> think: 我需要把这把刀 (1) 放到 countertop 1。
OK.
> go to countertop 1
...
> think: 我需要把这把刀 (1) 放到 countertop 1。
OK.
> put knife 1 in/on countertop 1
You put the knife 1 in/on the countertop 1.
> think: 我需要把这把刀 (1) 放到 countertop 1。
OK.
> put knife 1 in/on countertop 1
Nothing happens.
（重复）
```

#### D.3 WebShop 轨迹

**表 10：Act 和 ReAct 预测的 WebShop 示例轨迹。与 Act 相比，ReAct 使用推理找到满足所有目标属性的产品。**

指令：给我一包 16 包装的苹果肉桂冻干香蕉片，价格低于 50.00 美元。

Act：

```text
Action: search sixteen pack apple cinnamon freeze dried banana chips
Observation: ...
Action: click B0061IVFZE
Observation: ... flavor name [asian pear][banana][fuji apple & cinnamon][strawberry banana] ... Price: $85.0 ...
Action: click[Buy Now]
Score: 0.125
```

ReAct：

```text
Action: search sixteen pack apple cinnamon freeze dried banana chips
Observation: ...
Action: think[B0061IVFZE 是草莓香蕉，不是苹果肉桂。B096H2P6G2 是水果零食，不是冻干香蕉片。B092JLLYK6 是香蕉脆片，不是苹果肉桂。我可以先检查 B092JLLYK6。]
Observation: OK.
Action: click B092JLLYK6
Observation: ... flavor name [apple][apple cinnamon][banana]... size [0.53 ounce (pack of 6)][0.53 ounce (pack of 8)][0.53 ounce (pack of 16)] ...
Action: think[对于 16 包装苹果肉桂冻干香蕉片，该商品有选项 'apple cinnamon' 和 '0.53 ounce (pack of 16)'，似乎不错，可以购买。]
Observation: OK.
Action: click[apple cinnamon]
Observation: You have clicked apple cinnamon.
Action: click[0.53 ounce (pack of 16)]
Observation: You have clicked 0.53 ounce (pack of 16).
Action: click[Buy Now]
Score: 1.0
```

### E 更多分析

#### E.1 成功与失败模式分析

以下提供第 3.3 节成功和错误模式分析对应的示例。为节省空间，省略搜索结果和非代表性步骤。

**成功：真阳性**

ReAct：

```text
问题：作者 David Chanoff 曾与一位美国海军上将合作，该上将曾在哪位总统任内担任驻英国大使？
思考 1：我需要搜索 David Chanoff，找到他合作过的美国海军上将。
动作 1：Search[David Chanoff]
观察 1：David Chanoff 是著名作者……
动作 2：Search[William J. Crowe]
观察 2：William J. Crowe 是……
思考 3：William J. Crowe 在 Bill Clinton 总统任内担任驻英国大使。所以答案是 Bill Clinton。
动作 3：Finish[Bill Clinton]
```

CoT：

```text
问题：代表亚利桑那州一所大学的球队，在覆盖美国西部广大地区（包括亚利桑那、加利福尼亚和新墨西哥）的联盟中，球队名称是什么？
思考：让我们逐步思考。代表亚利桑那州一所大学的球队，在覆盖美国西部广大地区（包括亚利桑那、加利福尼亚和新墨西哥）的联盟中，名称是 Arizona State Sun Devils。
答案：Arizona State Sun Devils
```

**成功：假阳性**

ReAct：

```text
问题：桌游 San Marco 还是 About Time 先推出？
思考 1：我需要搜索 San Marco 和 About Time，并找出哪个先推出。
动作 1：Search[San Marco]
观察 1：San Marco 是……建于威尼斯。
思考 2：San Marco 是威尼斯的一个区。我接下来需要搜索 About Time。
动作 2：Search[About Time]
观察 2：找不到 [About Time]。相似：[...]。
思考 3：About Time 是 2013 年电影。所以 San Marco 先推出。
动作 3：Finish[San Marco]
```

CoT：

```text
问题：Vanderbilt University 还是 Emory University 先成立？
思考：让我们逐步思考。Vanderbilt University 成立于 1873 年。Emory University 成立于 1836 年。1873 > 1836，所以 Vanderbilt University 先成立。
答案：Emory University
```

**失败：推理错误**

ReAct：

```text
问题：与 Harvey Fierstein、Linda Hart、Dick Latessa、Clarke Thorell、Mary Bond Davis、Laura Bell Bundy、Matthew Morrison、Corey Reynolds 和 Marissa Jaret Winokur 一起出演 Hairspray 的女演员中间名是什么？
思考 1：我需要搜索 Harvey Fierstein、Linda Hart、Dick Latessa、Clarke Thorell、Mary Bond Davis、Laura Bell Bundy、Matthew Morrison、Corey Reynolds 和 Marissa Jaret Winokur，找到与他们一起出演 Hairspray 的女演员，然后找到她的中间名。
动作 1：Search[Harvey Fierstein]
观察 1：Harvey Forbes Fierstein……
思考 2：Harvey Fierstein 是演员、剧作家和编剧。我接下来需要搜索 Linda Hart……
动作 2：Search[Linda Hart]……
```

CoT：

```text
问题：Danny Green 还是 James Worthy 年龄更大？
思考：让我们逐步思考。Danny Green 出生于 1987 年。James Worthy 出生于 1961 年。1987 > 1961，所以 Danny Green 年龄更大。
答案：Danny Green
```

**失败：搜索错误**

ReAct：

```text
问题：与女神 frigg 相关的女神由哪些故事组成？
思考 1：我需要搜索 goddess frigg，并找到其相关女神。
动作 1：Search[goddess frigg]
观察 1：找不到 goddess frigg。相似：['Frigg', 'Frigg and Freyja common origin hypothesis', ...]。
思考 2：我需要改为搜索 Frigg……
```

**失败：幻觉**

CoT：

```text
问题：由位于 Bronx 的组织出版的科学期刊首次出版年份是哪一年？
思考：让我们逐步思考。由位于 Bronx 的组织出版的科学期刊首次出版年份是 1916 年。
答案：1916（标签：1909）
```

**失败：标签歧义**

ReAct：

```text
问题：开发独裁者游戏的人是什么国籍？
思考 1：我需要搜索 dictator game，找到开发它的人，然后找到他的国籍。
动作 1：Search[dictator game]
观察 1：独裁者游戏是社交心理学和经济学中流行的实验工具……
动作 3：Search[Amos Tversky]
观察 3：Amos Nathan Tversky……
思考 4：Amos Tversky 是以色列认知和数学心理学家。所以开发独裁者游戏的人是以色列人。
动作 4：Finish[Israeli]（标签：Israel-American）
```

CoT：

```text
问题：来自加州 Stockton 的 Kurt Fuller 出现在哪部由 Steve Franks 创作的美国侦探喜剧电视剧中？
思考：让我们逐步思考。来自加州 Stockton 的 Kurt Fuller 出现在 Psych 中。Psych 是由 Steve Franks 创作的美国侦探喜剧电视剧。
答案：Psych（标签：Psych is an American detective comedy-drama）
```

---

> 注：参考文献部分在原文中为英文条目列表，通常保持原文以便检索；本 Markdown 文档已翻译正文、表格、图注和附录说明。附录中的提示词与轨迹为保持可执行性和格式一致性，保留了动作/观察的英文形式，并对关键思考进行了中文翻译。