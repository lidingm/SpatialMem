# SpatialMem: Trustworthy Self-Evolving Skills and Memory for Training-Free Spatial Reasoning Agents


## Abstract

3D 场景中的空间智能仍是视觉语言模型（VLM）面临的一大挑战。近期一些 Spatial agent 通过引入外部工具增强空间感知和推理能力，但大多局限于单个样本内部，缺乏跨样本长期经验积累。已有工作尝试引入 SKILL 机制以支持跨样本经验积累，但该方法依赖模型自我判断从轨迹中提炼纯文本 SKILL，复用时仍需由 LLM 重新解释为具体工具调用，在感知噪声密集的空间任务中，这类未经验证的纯文本 SKILL 难以保证可靠性与可复用性。为此，本文提出 **SpatialMem**，一个 training-free 的自进化空间推理 agent 框架，将跨样本经验知识组织为陈述性 Memory 与过程性 SKILL，并通过五条自进化路径持续扩充。其中，相比于纯文本 SKILL 方法，SpatialMem 有两类优化设计：（1）确定性执行，将 SKILL 封装为可直接执行的代码单元，被选中即复现确定性的工具调用路径；（2）验证式知识准入，要求失败驱动的修正方案必须在原样本上重新执行并匹配真值才被允许记入知识库。二者分别从复用执行与入库准入两个阶段约束经验积累，从而提升长期知识的稳定性与可扩展性。框架全程无需修改 VLM 权重，知识库可审计、可跨模型复用。实验表明，在 VSI-Bench 基准上，SpatialMem 的框架与知识库使 【占位】 的性能提升【xx%】，超越代表性的智能体式方法 【占位】【xx%】。

Spatial intelligence in 3D scenes remains a major challenge for vision-language models (VLMs). Recent spatial agents incorporate external tools for stronger spatial perception and reasoning, but they are mostly confined to single-sample inference and lack long-term experience accumulation across samples. A recent work introduces a SKILL mechanism to support cross-sample experience reuse; however, it relies on the model's self-judgment to distill plain-text SKILLs from trajectories, which still need to be translated into concrete tool calls during reuse. In spatial tasks with dense perception noise, such unverified plain-text SKILLs are difficult to guarantee in terms of reliability and reusability. To address this issue, we propose **SpatialMem**, a training-free self-evolving spatial reasoning agent that organizes cross-sample experiential knowledge into declarative Memory and procedural SKILL, and continuously expands them through five self-evolution paths. Compared with plain-text SKILL methods, SpatialMem introduces two key designs: (1) deterministic execution, where each SKILL is encapsulated as directly executable code and reproduces a deterministic tool-calling path once selected; and (2) validated knowledge admission, where failure-driven repair plans must be rerun on their source samples and match the ground-truth answers before being persisted into the knowledge base. These designs constrain experience accumulation at the reuse-execution and admission stages, respectively, improving the stability and scalability of long-term knowledge. SpatialMem keeps the VLM weights frozen throughout the process, while maintaining an auditable and cross-model reusable knowledge base. Experiments on VSI-Bench show that SpatialMem improves 【placeholder】 by 【xx%】 and outperforms representative agentic baselines 【placeholder】 by 【xx%】.

---

## 1 Introduction

3D 场景中的空间智能要求模型理解物体间的距离、方位、数量与场景尺度，是具身智能与机器人应用的基础能力，却始终是视觉语言模型（VLM）的显著短板 [SpatialVLM; VSI-Bench]。在空间数据上后训练模型的路线成本高昂，且被证明容易过拟合基准分布 [ViSRA]；因此，以外部工具增强冻结 VLM 的智能体式方法成为主流的 training-free 替代：VLM 负责语义规划，深度估计、目标分割与三维定位等专家工具提供显式几何证据。

近期的智能体式方法已能显著提升单个样本内的空间推理。S-Agent [Dai et al.] 以分层空间工具与双记忆实现跨帧证据累积；SpatialClaw [NVIDIA] 以持久化代码内核实现灵活的工具组合。然而，这些方法的记忆与内核状态都随样本重置，智能体每次面对新问题仍需重新规划工具调用，难以把既往同类问题中的求解经验带入未来。与此同时，通用智能体领域已发展出成熟的应对范式：自进化技能库 [Voyager; Agent Skills; EvoSkill]，将成功经验沉淀为可复用的技能包。空间推理域的最新工作 Skill-3D [Li et al.] 正是沿这一方向，尝试引入 SKILL 机制以支持跨样本经验积累。

然而，把技能库范式直接落到空间推理上，会暴露两个此前不突出的障碍。首先，空间任务中感知噪声密集，深度在反光材质上失效、分割混淆相邻实例等情况频繁发生，工具可以给出精确几何量，却可能对应了错误物体或错误实例。因此，SpatialMem 在常规规划与反思角色之外保留一个 Checker 角色，用于结合可视证据与几何证据核查候选目标和最终定位。其次，这些噪声会进一步污染跨样本经验：现有方法依赖模型自我判断从轨迹中提炼纯文本 SKILL，Reflexion 式反思不经检验直接注入上下文 [Reflexion]，一致性校验也只检查候选知识与既往案例是否表面一致 [Skill-3D]。由此蒸馏出的知识不一定具有可行性，可能无效甚至引入污染；这类知识一旦入库，便以经验之名持续误导后续相似样本。此外，纯文本 SKILL 在复用时仍需由 LLM 重新解释为具体工具调用，同一技能的两次复用可能产生不同的工具顺序与参数，在空间工具链中造成额外执行波动。

为此，我们提出 **SpatialMem**，将自进化技能库范式引入空间推理，并针对上述问题做出系统适配（[Figure 1(c)]）。在单样本层面，SpatialMem 采用 Planner、Reasoner、Checker 与 Reflector 的多角色架构编排空间工具，使 VLM 可在规划、视觉检查与最终归纳中使用工具产生的视觉和几何证据。在跨样本层面，SpatialMem 将经验分为陈述性 Memory 与过程性 SKILL，并通过五条自进化路径持续扩充。相比于纯文本 SKILL 方法，SpatialMem 有两类优化设计：（1）**确定性执行**，即将 SKILL 封装为可直接执行的代码单元，被选中后即可复现确定性的工具调用路径，从而减少文本重新解释带来的执行波动；（2）**验证式知识准入**，即要求失败驱动的修正方案必须在原样本上重新执行并匹配真值，才被允许记入知识库。由此，SpatialMem 将空间经验的"如何存储、如何执行、如何准入"统一到一个 training-free 框架中，全程保持 VLM 权重冻结。

我们以 VSIBench 为主评测集，并在 MMSI-Bench、ViewSpatial-Bench、EmbSpatial-Bench、MindCube 与 All-Angles-Bench 上检验跨基准泛化。SpatialMem 超越了 S-Agent、SpatialClaw 与 Skill-3D 等代表性方法【占位】；针对性分析进一步表明：Checker 降低了目标绑定错误【占位】，验证式知识准入将入库知识中无效条目的比例从【占位】降至【占位】，并消除了无验证变体在训练后期的性能震荡；确定性执行使可执行 SKILL 的工具调用路径保持一致，而纯文本 SKILL 基线存在更高的执行波动【占位】。

本文的贡献如下：

1. 我们将通用智能体领域的自进化技能库范式系统地引入空间推理，并采用陈述性 Memory 与过程性可执行 SKILL 的双知识表征，使数值先验可统计更新、工具流程可确定性复现。
2. 我们提出验证式知识准入机制：五条路径持续扩充 Memory 与 SKILL，其中失败驱动路径必须经过原样本重跑验证方可写入长期知识。该机制区别于依赖 LLM 自我判断或一致性校验的纯文本技能方法，从准入阶段控制知识污染。
3. 我们在多个空间推理基准上验证了框架的有效性，并通过消融分离了手工先验、各进化机制、验证准入与可执行表征各自带来的收益。

---

## 2 Related Work

**空间推理的多模态大模型。** 一类工作通过空间监督数据或结构修改增强 VLM 的空间能力 [Cambrian-S; Spatial-MLLM; VST; SpatialVLM]。这些方法需修改权重、承担高昂训练成本，且其提升可能源于基准过拟合而非可迁移的空间理解 [ViSRA]。SpatialMem 采取正交的 training-free 路线，依靠外部工具与外置知识库实现能力增长。

**智能体式空间推理。** 该方向沿工具编排、动作接口、经验积累的脉络演进：VADAR 合成 3D 推理 API [Marsili et al.]，ViSRA 以多角色智能体编排专家模型 [Mou et al.]，S-Agent 引入分层空间工具与样本内双记忆 [Dai et al.]，SpatialClaw 论证了动作接口设计的决定性影响 [NVIDIA]，AlloSpatial 以固定的三阶段推理规程约束工具调用并将其蒸馏进权重 [AlloSpatial]。上述方法的知识均不跨样本保留。Skill-3D [Li et al.] 迈出跨样本积累的一步，其技能以 JSON 文本注入提示词、准入依赖一致性判断，并最终经 SFT 与 GRPO 固化进特定模型权重；SpatialMem 与之互补：采用可执行 SKILL 与验证式知识准入，且保持基础模型冻结。

**智能体的技能库与经验记忆。** Voyager 开创了以可执行代码构建自增长技能库的范式 [Wang et al.]，Anthropic 的 Agent Skills 规范进一步将技能标准化为含 SKILL.md 与辅助脚本的文件夹格式 [Anthropic]，本文的技能表征即遵循这一规范。近期 Formal Skill 进一步指出，Markdown 或提示词形式的技能虽然便于书写，但执行顺序、状态恢复与完成条件主要停留在自然语言约束中，难以由运行时强制保证；因此它将技能形式化为带结构化接口、执行器、状态与控制钩子的可编程运行时模块 [Formal Skill]。文本形式的经验积累包括反思 [Reflexion]、跨任务洞见 [ExpeL]、规则手册 [AutoManual] 与过程性记忆 [Memp]。与本文最相关的是近期的技能进化工作：EvoSkill 通过失败分析提出技能、以 held-out 验证集的性能提升决定保留 [EvoSkill]；CoEvoSkills 以协同进化的代理验证器为技能包合成测试 [CoEvoSkills]；SkillBrew 与 SkillOps 研究技能库的策展与维护 [SkillBrew; SkillOps]。SpatialMem 与这些工作共享诸多设计，包括文件夹级可执行 SKILL、失败驱动的发现与验证把关。本文的贡献在于将该范式适配到感知噪声密集的空间推理：进化在训练流中逐样本在线进行，技能与陈述性数值先验协同，且每条知识在入库前均经真实执行的验证，即检验其对应的修正方案是否确实修复了产生它的失败样本。

---

## 3 Method

### 3.1 Agent 架构与问题形式化

**问题设定。** 给定训练集 $\mathcal{D}_{train}=\{(s_i, a_i^\star)\}$，其中每个样本 $s$ 包含一段室内场景视频（若干帧）与一个空间问题，$a^\star$ 为真值答案。智能体由一个冻结的 VLM、一组确定性空间工具 $\mathcal{T}$、以及两类可持久化的知识资产构成：陈述性记忆 $M$ 与过程性技能库 $\Sigma$。训练阶段的目标不是更新 VLM 参数，而是更新知识状态：处理第 $t$ 个样本后，

$$(M_{t+1}, \Sigma_{t+1}) = \mathrm{Evolve}\big(M_t, \Sigma_t;\ \tau_t,\ \mathbb{1}[\hat{a}_t = a_t^\star]\big),$$

其中 $\tau_t$ 为该样本的完整求解轨迹（规划、技能选择、工具调用、中间证据与最终答案）。测试阶段冻结 $(M, \Sigma)$，智能体直接检索并消费进化后的知识。

**验证式知识准入原则。** SpatialMem 对失败驱动路径中任何由诊断触发的结构性写入施加如下约束：

$$\mathrm{persist}\big(\Delta\Sigma\big) \iff \exists\, \pi' :\ \mathrm{Rerun}(s, \pi').\mathrm{answer} = a^\star(s), \tag{1}$$

即：反思产生的修正方案 $\pi'$ 必须在**同一样本** $s$ 上重新执行、且产出答案与真值匹配，相应的知识更新 $\Delta\Sigma$（新技能或新教训）才被允许持久化。式 (1) 是五路径自进化中的准入规则，而非单样本推理时的感知校验；它将"这条经验是否有效"从模型的主观判断变成一次可复现的实验，其完整机制在 §3.4 展开。

**架构总览。** 如 [Figure 2] 所示，SpatialMem 自底向上由三部分组成。

*工具层。* 空间感知能力来自 11 个确定性 Python 工具，由两个感知模型支撑：Depth Anything 3 [Lin et al.] 提供单目度量深度、相机位姿与稠密点云；SAM3 [Carion et al.] 提供文本提示的实例分割与跨帧跟踪。工具按依赖链组织：`depth_estimation` 是几乎所有下游工具的前驱；`object_segmentation` 的 2D 掩码经 `instance_3d_localization` 反投影与跨帧聚类得到全局 3D 实例；在此之上，`distance_computation`、`direction_computation`、`instance_counting` 与 `object_size_computation` 执行距离、方位、计数与物体尺寸等数值几何计算；`scene_size_computation`、`bev_generation` 与 `novel_view_synthesis` 提供场景尺度、鸟瞰渲染与新视角合成等辅助证据。典型的最短工具链为 `depth → seg → 3d_loc → {dist | dir | count | object_size}`。VLM 从不接触工具代码，只读取其自然语言描述。完整工具规格见附录。

*角色层。* 四个 LLM 角色驱动单样本推理。**Planner** 为全局决策者，综合问题、图像、Memory 上下文与技能检索结果，一次性输出任务类别、目标物体与所选技能（或备用工具计划）。**Reasoner** 仅在无技能可用时启用，在原始工具循环中逐步细化调用。**Checker** 是被动式视觉校验者：它不提出新计划、不调用几何工具，而是在工具产生候选实例后查看带框帧或原始帧，判断候选是否对应问题目标、计数实例是否需要剔除/合并/拆分/补漏，并将确认后的目标绑定传递给后续距离、方位、尺寸与最终定位绘制。**Reflector** 在每轮末尾评估证据充分性并决定继续或终止、训练时对答错样本产出诊断与修正方案、并基于全部工具证据与 Memory 先验抽取最终答案。

Checker 的作用不同于式 (1) 的知识准入验证。前者发生在单样本推理内部，解决"工具计算的是不是目标物体"；后者发生在训练时进化之后，决定"由失败反思产生的经验能否进入长期知识库"。这种时间尺度与职责分离，使 SpatialMem 既能在当前样本中稳定目标定位，又能在跨样本学习中控制知识污染。

*知识层。* 单样本主循环为：装配 Memory 上下文与技能检索结果 → Planner 规划 → 技能执行或原始工具循环 → 必要时由 Checker 校验目标绑定与实例修正 → Reflector 终结产出答案。训练模式在此之后触发进化机制（§3.3），失败驱动的结构性写入受式 (1) 约束（§3.4）；测试模式跳过进化，仅消费知识。

### 3.2 双知识表征：陈述性 Memory 与过程性 SKILL

借用认知科学中长时记忆的经典二分法，SpatialMem 将跨样本知识组织为**陈述性**（declarative，"知道是什么"的事实先验，被动查询）与**过程性**（procedural，"知道怎么做"的可执行流程，主动调用）两类载体。二者共享同一学习过程产生的轨迹作为数据源，但存储形式与消费路径截然不同。这一差异源于两类知识的本质属性：事实先验必须支持数值查询与统计更新，操作流程必须支持直接执行。

**陈述性 Memory。** Memory 存储三类内容：（i）物体尺寸先验：每个物体类别在宽/高/深三个维度上的在线统计量（均值与方差，Welford 算法增量更新），来源于历次成功轨迹中 3D 定位工具的输出；（ii）场景尺度先验：按场景类型索引的房间尺寸与地板面积统计；（iii）疑难样本记录：验证失败的样本及其轨迹摘要，作为未来技能设计的候选素材。为防止异常观测污染先验，新条目先进入候选池，观测次数达到预热阈值后才晋升入主池。Memory 的消费是被动的：匹配的先验被拼装为文本上下文注入提示词，主要服务于 Reflector 的答案合理性校验，例如当计算出的物体尺寸远超先验分布时降低置信度并触发复查。

**过程性 SKILL。** 每个技能遵循 Agent Skills 文件夹规范 [Anthropic]，包含三个文件：`SKILL.md`（结构化文档，含使用时机、参数规格、工具序列与已知陷阱四个可进化段落及调用统计元数据）、`execute.py`（签名固定的可执行函数，封装完整的工具调用逻辑）、`trajectories.jsonl`（追加式调用日志）。技能的消费是主动的：Planner 选中某技能后，框架直接动态加载并执行其 `execute.py`，工具调用序列、参数解析与异常处理全部走确定性代码路径，VLM 只承担一次"选技能、填参数"的决策。

**可执行表征在空间工具链上的价值。** 将技能表示为文本或 JSON 注入提示词时，"如何执行"仍需 LLM 在每次复用时重新翻译成具体调用，因此同一技能的两次复用可能产生不同的工具顺序与参数，甚至幻觉出不存在的工具名。空间工具链具有严格的依赖结构与数值参数敏感性（深度必须先于反投影、聚类阈值直接决定实例数），对这类方差尤其脆弱。可执行表征从机制上消除了这层方差：技能一旦通过质量校验入库，每次执行严格复现同一代码路径。代码形式还使技能可接受程序化的三重质量关卡（语法解析、入口函数检查、运行时导入校验），无法通过校验的候选在入库前即被拒绝。

**两类知识的协作。** Memory 与 SKILL 在推理循环中于四处协作（[Figure 2] 中虚线标注）：Planner 同时消费两者做联合决策；Planner 推断的场景类型反向精化 Memory 检索；一次成功执行的工具输出被双向分流，数值证据更新先验，调用轨迹追加进技能日志；Memory 中累积的未覆盖类别轨迹则成为主动技能生成的原料（§3.3）。

### 3.3 五种自进化机制

训练时，每个样本求解完成后，进化模块依据"答案是否正确 × 是否使用了技能"分派处理路径，这一 2×2 组合天然划分出四条**反应式**路径（[Figure 3]），辅以一条**主动式**引导通道。

**Path A（答对 × 用了技能）：强化与适用边界精化。** 技能的成功统计被更新，轨迹摘要追加入其日志；每累积若干次成功，LLM 综合近期轨迹按需重写该技能的"使用时机"段落，使适用边界随经验收敛。同时，本次工具输出中的物体尺寸与场景尺度被提取更新 Memory 先验。

**Path B（答对 × 未用技能）：从成功中发现新技能。** 一次不依赖技能、包含至少两次有效工具调用的成功求解，意味着可能存在尚未被覆盖的解题模式。LLM 在展示该类别下全部已有技能的前提下三选一：跳过（已有覆盖）、更新（追加适用描述）、或创建（产出完整的文档与可执行代码）。新技能须通过三重程序化校验后进入待晋升池。

**Path C（答错 × 用了技能）：从验证过的失败中蒸馏教训。** Reflector 对比真值与预测产出诊断与修正方案，修正方案在同一样本上重跑验证（§3.4）；仅当验证通过，LLM 才对比失败与成功两次执行、蒸馏出一条"触发条件 → 修正动作"形式的教训，追加进技能的"已知陷阱"段落。验证失败的样本记入疑难库，技能不受任何写入。

**Path D（答错 × 未用技能）：从验证过的修复中发现新技能。** 复用 Path C 的验证流程；若修正方案验证通过，则以修正后的成功轨迹为输入走 Path B 的蒸馏流程，产出新技能。

**Bootstrap：类别级主动引导。** 反应式路径的技能发现依赖"恰好出现一次可蒸馏的轨迹"，对系统性缺乏覆盖的任务类别收敛缓慢。为此，每当某样本无技能可用，其轨迹被记入对应类别的缓冲区；当某类别累积足够多的无技能轨迹时，LLM 一次性读取其中的成功与失败案例、对比分析后主动设计一个覆盖该类别的新技能。反应式与主动式构成互补的双通道发现：前者敏捷响应单点机会，后者兜底系统性盲区；反应式产出的技能会使同类别的主动引导自动跳过，避免重复。

所有新蒸馏的技能先进入待晋升池，与主库一同被检索（附带自动蒸馏标签），累积足够成功次数后整体晋升。技能的可靠性由真实使用中的成功率与失败修复的重跑验证共同背书，而非由生成它的 LLM 自我担保。

### 3.4 验证式知识准入

**动机。** 失败样本蕴含最有价值的学习信号，但失败诊断也是幻觉风险最高的环节：LLM 面对"预测 0.9 米、真值 2.2 米"的差异，可以流畅地编造出任何听起来合理的归因（分割错了目标、深度在反光面失效、聚类合并了两把椅子），而其中多数与真实错误原因无关。若将这些诊断不加验证地固化为知识，无效甚至有害的"教训"就会以经验之名持续误导后续推理，且随训练规模累积。

**机制。** SpatialMem 对失败诊断施加式 (1) 的准入约束，形成"反思、重跑验证、蒸馏"三步闭环。给定答错的样本 $s$：（i）反思：Reflector 接收问题、原计划、预测、真值与完整工具证据，产出诊断文本与一份具体到工具与参数的修正方案 $\pi'$；诊断不允许停留在文字层面，必须落实为可重新执行的计划。（ii）重跑验证：框架按照 $\pi'$ 在同一样本 $s$ 上完整重新执行工具流程，并将产出答案与真值比对。（iii）蒸馏：仅当验证通过，该诊断才被认定为"真实可修复的失败模式"，进而蒸馏为持久知识：Path C 中为技能追加教训，Path D 中蒸馏新技能。验证失败时不做任何猜测性写入，样本落入疑难库。值得注意的是，验证式知识准入检验的是修正计划是否能够修复原始失败；通过验证后的轨迹才会被进一步蒸馏为 lesson 或 SKILL，并在后续复用中受可执行 SKILL 的确定性执行约束。

**与既有准入机制的关系。** 现有机制大多依赖模型自身的判断：反思直接注入上下文而不经任何检验 [Reflexion]；一致性校验检查候选更新是否与既往成功案例一致 [Skill-3D]，能过滤明显噪声，但错误归因完全可以与既往案例表面一致。这些机制下总结出的知识未经真实执行的验证，不一定具有可行性，可能无效甚至引入污染。式 (1) 把单条知识的有效性判断交给环境本身：修正方案要么在真实样本上跑通并给出正确答案，要么被拒绝。在感知噪声使"偶然蒙对"高发的空间推理域，验证式知识准入是必需而非锦上添花。

**代价与收益。** 每次触发验证需要额外一次完整重跑（约增加 30% 的训练时计算量），且仅发生在训练阶段。作为回报，技能库中每条教训都对应一次在真实样本上确证有效的修复记录，知识库的增长与其可靠性保持同步，§4.3 将量化这一收益。

### 3.5 手工先验与进化产出的分界

训练开始前，系统仅包含 11 个工具的实现与描述、4 个角色的提示词模板、以及 5 个手写的种子技能：`measure_distance`（绝对/相对距离）、`count_objects`（物体计数）、`judge_direction`（相对方位）、`measure_object_size`（物体尺寸）与 `measure_scene_size`（房间尺寸）。这些种子只覆盖最基础、最稳定的空间工具链；其适用边界、已知陷阱、后续新技能、以及 Memory 的全部数值先验仍由训练时进化产生。未被种子充分覆盖的任务模式（例如外观顺序、路径规划或更复杂的组合推理）完全依靠 Path B/D 与 Bootstrap 发现。训练产出包括新技能及其晋升、既有技能的段落更新、Checker 相关经验、以及 Memory 的全部数值先验与记录。这一分界在 §4.6 的消融中被逐项量化。实现细节（超参、白名单校验、提示词全文）见附录。

---

## 4 Experiments

> 本节当前为**实验设计与预测结论**，所有数字以【占位】标记，待实验完成后填入。

### 4.1 实验设置

**基准。** 主评测集为 VSIBench [Yang et al.]（约 5,130 条样本、10 类空间问题）；泛化评测覆盖 MMSI-Bench、ViewSpatial-Bench、EmbSpatial-Bench、MindCube 与 All-Angles-Bench。训练集使用 VSI-Train-10k，与全部评测集样本不重叠。当前实现以每类最多【占位】个训练样本进行知识积累，训练完成后冻结 checkpoint 中的 SKILL 与 Memory 进行测试。

**基线。** 四组：（i）直接 VLM 推理（GPT 系列、Gemini 系列、Qwen3-VL 系列等）；（ii）智能体式方法，包括 S-Agent、SpatialClaw 与 Skill-3D，其中 Skill-3D 为最直接的自进化技能对照；（iii）空间专用后训练 VLM（Cambrian-S、Spatial-MLLM、VST）；（iv）本方法及其消融变体。智能体式基线与本方法使用同一驱动 VLM 与相同帧采样设置。

**指标。** 对 VSIBench，选择题报告准确率（ACC），数值填空题报告官方 Mean Relative Accuracy（MRA），即在阈值 $\theta \in \{0.50,0.55,\ldots,0.95\}$ 上的平均相对精度。泛化基准遵循各自官方指标。除最终精度外，另报告技能命中率、平均工具调用轮数、Checker 修正率、每样本推理延迟与训练阶段验证开销。

### 4.2 主结果

**设计。** 在训练集上完成知识积累后冻结知识库，先在 VSIBench 上与全部基线对比精度，再在五个泛化基准上评估跨数据集迁移，并按任务类别细分。

**预测结论。** SpatialMem 在多数基准上超越无跨样本知识的 S-Agent 与 SpatialClaw，并在与 Skill-3D 的对比中取得一致优势。优势应集中在两处：感知噪声高发的类别（计数、绝对距离、物体尺寸），因为 Checker 降低了目标绑定错误，而验证式准入阻止了错误几何模式入库；以及种子未充分覆盖、依赖进化产出技能的类别，因为双通道发现保证了覆盖。相对后训练 VLM，预期以零训练成本达到可比或更优精度。

### 4.3 知识准入机制的对比分析（招牌实验）

**设计。** 在相同训练集与顺序下对比两种准入策略：（a）SpatialMem-Full：验证式知识准入（式 1）；（b）SpatialMem-NoVerify：移除原样本重跑验证，诊断直接蒸馏入库（对应现有方法中 LLM 自我判断与一致性校验式的准入）。比较三项：（i）无效知识比例：训练结束后对入库知识做事后审计，人工标注随机子集判断其可行性与因果有效性，辅以代理指标（某条知识被消费后同类样本正确率的变化）；（ii）训练动态：[Figure 4] 绘制滚动精度、库规模与无效条目数随训练样本数的演化；（iii）最终测试精度与准入开销。

**预测结论。** NoVerify 的库增长更快，但其中无效或有害知识的比例显著更高【占位】，滚动精度在训练中后期出现震荡或衰减；Full 版本入库的每条知识均经真实执行确证，曲线单调平滑上升，最终精度更高。该对照将量化验证环节的净价值，支撑贡献 3。

### 4.4 技能复用的执行一致性

**设计。** 选取被高频复用的技能，在其适用样本集上各复用 N 次，度量实际执行的工具调用序列相对标准流程的编辑距离与参数一致性。对照组为纯文本 SKILL 基线：将同一技能的文档以 JSON 形式注入提示词，由 LLM 现场生成工具调用。同时统计幻觉工具名出现率。

**预测结论。** 可执行 SKILL 的执行序列编辑距离恒为零、幻觉率为零；文本化基线的一致性显著更低【占位】、幻觉率非零【占位】，且不一致性在弱驱动模型上放大。该实验将表征差异转化为可度量的行为差异。

### 4.5 Checker 对目标绑定稳定性的影响

**设计。** 对比 SpatialMem-Full 与 SpatialMem-NoChecker：后者保留相同工具链、技能库与 Memory，但移除 Checker 对候选实例、计数修正与最终定位的视觉审计，直接消费 `instance_3d_localization` 的聚类结果。评估三项：（i）最终精度，尤其是计数、距离、方向与物体尺寸任务；（ii）目标绑定错误率：人工审计随机子集，标注工具计算是否落在问题目标上；（iii）Checker 操作统计，包括 spurious 删除、merge、split、missed 补充与 zero-track direct answer 的频率。

**预测结论。** NoChecker 在工具几何本身正确但目标绑定错误的样本上显著退化，尤其体现在相似实例密集、开放词汇歧义强的场景。Full 版本通过视觉审计把 VLM 的语义判别能力注入确定性工具链，使最终 localization 与数值答案更稳定。该实验对应贡献 1，区别于 §4.3 的长期知识准入分析。

### 4.6 消融：手工先验与进化机制的贡献分解

**设计。** 8 组递进设置：(0) 无知识基线（仅原始工具循环）；(1) 仅 Memory；(2) 仅 5 个种子技能（进化全关）；(3)–(6) 在 (2) 之上分别单开 Path A / Path B+D / Path C / Bootstrap；(7) 全开。核心对比量为 (7)−(2)（自进化净收益）与 (2)−(0)（手工先验贡献），按任务类别细分。

**预测结论。** (2)−(0) 集中在种子覆盖的基础工具链类别；(7)−(2) 在种子未充分覆盖或需要更细适用边界的类别上尤其显著。各路径预期互补：Path A 精化边界（提升命中后成功率）、Path B/D 与 Bootstrap 扩大覆盖（提升命中率）、Path C 降低同类错误复发率。

### 4.7 效率与成本结构

**设计。** 对比三种范式的成本结构：SpatialMem（训练时一次性知识积累开销 + 每样本推理延迟，按帧数分档）、Skill-3D（SFT 与 RL 的一次性训练成本 + 推理延迟）、直接 VLM。特别报告知识更新的边际成本，即修正或新增一条知识的开销。

**预测结论。** SpatialMem 推理延迟高于直接 VLM 但与 S-Agent 同量级；验证开销仅存在于训练阶段。在"发现错误知识后修复"的持续运营场景下，外置知识库的边际成本（文件级增删）比权重蒸馏（一轮重训练）低若干数量级。

### 4.8 驱动模型可迁移性与分布外泛化

**设计。** （i）以不同规模 VLM 驱动同一框架，并互换知识库（模型 A 积累的库驱动模型 B 推理）；（ii）在与训练集分布差异最大的 EmbSpatial 与 MindCube 上，对比 SpatialMem 与权重蒸馏方法的精度衰减。

**预测结论。** 知识库跨模型迁移仅带来轻微损失，证明知识以模型无关的形式沉淀；分布外设置下权重蒸馏模型衰减更大，而外置知识库中未失配的部分（如尺寸先验）依然有效。

---

## 5 Conclusion

本文将通用智能体领域的自进化技能库范式引入空间推理，并识别出其落地的两个关键障碍：单样本工具链容易在目标绑定处失效，跨样本总结出的知识若未经验证则可能无效甚至污染长期经验。我们提出 SpatialMem，以过程性可执行 SKILL 减少纯文本技能复用时的执行波动，以验证式知识准入在失败驱动路径中控制知识写入，并以陈述性 Memory 保存可统计更新的空间先验。整个过程不修改基础模型权重。我们希望这一工作表明：空间智能体的长期经验积累，必须同时具备可确定复用的过程知识与面向真实执行结果的准入验证机制。

## Limitations

本框架存在三点边界。其一，能力受底层工具集覆盖约束：对超出现有工具语义的任务（如路径规划），主动引导也未必能设计出有效技能。其二，Checker 与验证式知识准入都大幅压低但不能消除错误风险：Checker 仍可能在模糊或遮挡帧中误判目标，修正方案也可能因工具输出的随机性偶然通过验证，蒸馏出的教训仍可能是伪相关；提高多帧审计强度或重跑轮数可进一步压低该风险，代价是更高的训练与推理开销。其三，推理延迟持续存在于每次查询，且框架能力的天花板依赖驱动 VLM 正确消费技能文档与先验上下文的能力。

---

## References（占位清单）

- Dai et al. S-Agent: Spatial Tool-Use Elicits Reasoning for Spatial Intelligence. arXiv:2606.20515.
- Cho et al. SpatialClaw: Rethinking Action Interface for Agentic Spatial Reasoning. arXiv:2606.13673.
- Li et al. Skill-3D: Evolving Scene-Aware Skills for Agentic 3D Spatial Reasoning. arXiv:2606.07436.
- Mou et al. ViSRA: A Video-based Spatial Reasoning Agent for MLLMs. arXiv:2605.10106.
- AlloSpatial: Agentic Harness Framework for Spatial Reasoning. arXiv:2606.08952.
- Yang et al. VSI-Bench / Thinking in Space. CVPR 2025.
- Yang et al. Cambrian-S. arXiv:2511.04670.
- Wang et al. Voyager: An Open-Ended Embodied Agent with LLMs. arXiv:2305.16291.
- Anthropic. Agent Skills specification. 2025.
- Formal Skill: Programmable Runtime Skills for Efficient and Accurate LLM Agents. arXiv:2605.19604.
- Alzubi et al. EvoSkill: Automated Skill Discovery for Multi-Agent Systems. arXiv:2603.02766.
- Zhang et al. CoEvoSkills: Self-Evolving Agent Skills via Co-Evolutionary Verification. arXiv:2604.01687.
- Fang et al. Memp: Exploring Agent Procedural Memory. arXiv:2508.06433.
- Shinn et al. Reflexion. NeurIPS 2023.
- Zhao et al. ExpeL: LLM Agents Are Experiential Learners. AAAI 2024.
- Chen et al. AutoManual. NeurIPS 2024.
- SkillBrew: Multi-Objective Curation of Skill Banks. arXiv:2605.29440.
- SkillOps: Managing LLM Agent Skill Libraries. arXiv:2605.13716.
- Marsili et al. VADAR. arXiv:2502.06787.
- Lin et al. Depth Anything 3. arXiv:2511.10647.
- Carion et al. SAM 3. arXiv:2511.16719.
- （基准与后训练 VLM 引用待补齐。）

---

## Appendix 规划（目录）

- **A. 工具规格全表**：11 个工具的输入/输出/依赖/典型用法，含受限贪心聚类伪代码与 Checker 后处理接口。
- **B. 提示词全文**：Planner、Reasoner、Checker、Reflector 及五种蒸馏操作的完整提示词。
- **C. 实现细节**：进化超参表、工具白名单校验、待晋升池的三重程序化校验、JSON 结构化输出与重试策略。
- **D. SKILL 文件夹示例**：一个种子技能与一个进化产出技能的完整三文件对照。
- **E. 知识污染审计协议**：§4.3 人工标注标准与代理指标定义。
- **F. 定性案例**：验证闭环拦截伪相关教训的完整轨迹、Bootstrap 生成新技能的端到端过程。
