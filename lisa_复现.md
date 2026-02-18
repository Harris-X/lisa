为了复现论文《Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning Attack》，你需要遵循一个严格的“预训练-对齐-微调-评估”流程。

以下是完整的复现流程说明及参数配置表。

### 3. 复现验证流程详解

整个复现过程分为三个主要阶段：**准备阶段**、**第一阶段（安全对齐）和第二阶段（用户微调/Lisa防御）**，最后是**性能评估**。

#### **第一步：模型与数据准备 (Preparation)**

1. **选取基础模型**：
    
    - 推荐使用 **Llama2-7B** 作为基础模型，这是论文主实验中使用的核心模型 。
        
        +2
        
    - 备选模型：Opt-2.7B 或 Mistral-7B 。
        
        +1
        
2. **准备数据集**：
    
    - **对齐数据 (Alignment Data)**：使用 **BeaverTails** 数据集中的安全样本（Safe samples）。这用于第一阶段的对齐训练，以及在第二阶段 Lisa 算法中作为“锚点”数据。
        
        +1
        
    - **用户任务数据 (User Data - Benign)**：选择一个下游任务数据集，例如 **SST2** (情感分析)、**AGNEWS** (新闻分类) 或 **GSM8K** (数学推理) 。
        
    - **有害数据 (Harmful Data)**：从 BeaverTails 中选取有害样本（Unsafe samples）。
        
3. **构建攻击数据集 (Simulating the Attack)**：
    
    - 创建一个混合数据集用于第二阶段微调。
        
    - **混合策略**：总样本数 $n=5000$。其中 $90\%$ 是用户任务数据，$10\%$ ($p=0.1$) 是有害数据 。
        
    - **目的**：模拟用户在微调时无意或恶意混入有害数据的情况。
        

#### **第二步：第一阶段 - 初始安全对齐 (Stage 1: Alignment)**

在进行 Lisa 防御之前，必须先有一个已经“对齐”过的模型。

1. **训练目标**：让预训练模型学会安全回复。
    
2. **训练方法**：使用 BeaverTails 的安全数据（10,000条样本）对 Llama2-7B 进行 **SFT (Supervised Fine-Tuning)** 。
    
3. **技术细节**：使用 LoRA (Rank=8) 进行参数高效微调 。
    
4. **产出**：获得一个“已对齐模型 (Aligned Model)”，记为 $w_{aligned}$。
    

#### **第三步：第二阶段 - 用户微调与 Lisa 防御 (Stage 2: User Fine-tuning with Lisa)**

这是复现的核心。在这一步，我们模拟用户拿着混有毒数据的混合数据集对 $w_{aligned}$ 进行微调，并使用 Lisa 算法进行防御。

1. **加载模型**：加载上一阶段得到的 $w_{aligned}$。
    
2. **初始化 LoRA**：固定预训练参数，训练一个新的 LoRA Adapter 。
    
3. **执行 Lisa 算法 (双态优化 + 近端约束)**： 不要直接混合数据训练，而是开启一个循环，交替执行以下两个状态 ：
    
    - **状态 1 (Alignment State)**：
        
        - **数据**：使用 BeaverTails 对齐数据。
            
        - **步数**：训练 $K_1 = 100$ 步。
            
        - **损失函数**：$L = L_{SFT}(D_{align}) + \frac{\rho}{2}||w - w_{prev\_state}||^2$。即：最小化对齐损失，同时通过近端项（Proximal term）强迫参数不要偏离上一个状态太远。
            
    - **状态 2 (User Fine-tuning State)**：
        
        - **数据**：使用混有毒数据的用户数据集 (SST2 + Harmful)。
            
        - **步数**：训练 $K_2 = 900$ 步（模拟非对称计算，把更多算力给用户任务）。
            
        - **损失函数**：$L = L_{SFT}(D_{user}) + \frac{\rho}{2}||w - w_{prev\_state}||^2$。
            
    - **循环**：重复上述交替过程直到达到设定的 Epoch 数。
        

#### **第四步：性能验证与评估 (Evaluation)**

复现成功与否取决于是否同时在“安全性”和“任务性能”上达标。

1. **验证指标 1：有害分数 (Harmful Score, HS)**
    
    - **对应微调说明**：验证 Lisa 是否成功抵御了 $10\%$ 有毒数据的注入。
        
    - **操作**：使用 BeaverTails 的测试集（未见过的恶意指令，如1000条），输入给微调后的模型。
        
    - **判定**：使用 BeaverTails 提供的 **Moderation Model** 检测模型输出。HS 是被标记为“Unsafe”的回复比例。
        
    - **预期结果**：Lisa 的 HS 应显著低于普通 SFT（例如从 ~50% 降至 ~35%）。
        
2. **验证指标 2：微调准确率 (Finetune Accuracy, FA)**
    
    - **对应微调说明**：验证 Lisa 的“近端约束”是否导致模型无法学习用户任务（过犹不及）。
        
    - **操作**：使用用户任务的测试集（如 SST2 Test Set）。
        
    - **判定**：计算 Top-1 准确率（对于分类任务）或正确率（对于 GSM8K）。
        
    - **预期结果**：Lisa 的 FA 应与普通 SFT 持平或仅有微小下降（例如 SST2 保持在 ~95%）。
        

---

### 4. 模型训练各阶段参数表

下表总结了复现该论文所需的详细超参数。

|**阶段 (Stage)**|**参数名称 (Parameter)**|**数值 (Value)**|**说明/上下文**|**出处 (Source)**|
|---|---|---|---|---|
|**通用设置**|**Hardware**|NVIDIA H100|单卡工作站||
||**LoRA Rank**|8|用于两个阶段的 Adapter||
||**LoRA Implementation**|Double-LoRA|分别用于对齐和微调的两个独立Adapter||
|**阶段 1: 安全对齐**|**Learning Rate**|1e-3|较大，为了快速学习对齐知识||
|(Alignment Stage)|**Batch Size**|5|||
||**Epochs**|30|保证充分收敛||
||**Dataset Size**|10,000|采样自 BeaverTails||
||**Optimizer**|AdamW|(推断自微调阶段设定)||
|**阶段 2: 用户微调**|**Learning Rate**|1e-5|较小，避免破坏已有知识||
|(User Fine-tuning)|**Batch Size**|5|||
||**Optimizer**|AdamW|||
||**Epochs**|20 (SST2/AGNEWS)<br><br>  <br><br>50 (GSM8K)|根据任务难度调整||
||**Dataset Size (n)**|5,000|默认总样本数||
||**Poison Ratio (p)**|0.1 (10%)|有害数据占比||
|**Lisa 算法特定参数**|**Step Allocation ($K_1$)**|100 步|投入在对齐状态的步数||
|(Lisa Specifics)|**Step Allocation ($K_2$)**|900 步|投入在微调状态的步数||
||**Proximal Intensity ($\rho$)**|1|近端项的惩罚系数||

**注意：**

- **非对称计算**：请特别注意 $K_1$ (100) 和 $K_2$ (900) 的差异，这是论文强调的“非对称计算”场景，也是导致普通方法失效、Lisa 生效的关键设置 。
    
    +1
    
- **近端项 ($\rho$)**：在代码实现 loss function 时，确保加入 $\frac{\rho}{2}||w_{current} - w_{checkpoint}||^2$ 。