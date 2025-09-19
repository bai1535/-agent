from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType  # 新增 prepare_model_for_kbit_training
import torch
import logging
import os
import random
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # ✅ 禁用并行化

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 模型路径配置
    model_path = "/root/autodl-tmp/medical/models/Qwen2.5-7B"
    output_dir = './qlora_results'  # 修改输出目录名称
    os.makedirs(output_dir, exist_ok=True)
    
    # ===================== QLoRA核心修改：4-bit量化配置 =====================
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,              # 启用4-bit量化
        bnb_4bit_quant_type="nf4",      # 使用NF4量化类型
        bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时使用bfloat16
        bnb_4bit_use_double_quant=True  # 启用双重量化减少内存
    )
    
    # 加载学生模型（应用4-bit量化）
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,  # 添加量化配置
            device_map="auto",
            trust_remote_code=True,
            use_cache=False
        )
        logger.info("学生模型加载成功（4-bit量化）")
    except Exception as e:
        logger.error(f"学生模型加载失败: {str(e)}")
        raise
    
    # ===================== 准备模型用于k-bit训练 =====================
    model = prepare_model_for_kbit_training(model)
    
    # 改进的LoRA配置（保持不变）
    lora_config = LoraConfig(
        r=16,  # 增加秩
        lora_alpha=32,  # 增加alpha
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 增加更多目标模块
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="lora_only",  # 添加偏置项
        inference_mode=False
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tokenizer（保持不变）
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        logger.info("Tokenizer加载成功")
    except Exception as e:
        logger.error(f"Tokenizer加载失败: {str(e)}")
        raise
    
    # 改进的训练参数（启用梯度检查点）
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  
        per_device_train_batch_size=8,  
        gradient_accumulation_steps=4,  
        logging_steps=10,
        save_strategy='steps',
        save_steps=100,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        learning_rate=1e-4,  
        weight_decay=0.01,  
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        dataloader_num_workers=4,  
        report_to='tensorboard',
        remove_unused_columns=True,
        optim="paged_adamw_8bit",  # 使用分页优化器防止内存峰值
        gradient_checkpointing=True,  # 启用梯度检查点（QLoRA关键）
        logging_dir='./logs',
        max_grad_norm=1.0,  
        evaluation_strategy="steps",  
        eval_steps=100,  
        load_best_model_at_end=True,  
        metric_for_best_model="loss",  
        greater_is_better=False,  
        group_by_length=True,  
        prediction_loss_only=True,  
        dataloader_pin_memory=True,  
        dataloader_prefetch_factor=2,  
        auto_find_batch_size=True,  
    )
    
    # 数据准备（保持不变）
    try:
        from dataset import SFTDataset
        
        data_collator = DefaultDataCollator()
        dataset = SFTDataset(
            '/root/autodl-tmp/medical/data/CMtMedQA.json',
            tokenizer=tokenizer,
            max_seq_len=256
        )
        logger.info(f"数据集加载成功，样本数: {len(dataset)}")
        
        # 数据集分割
        from sklearn.model_selection import train_test_split
        train_idx, eval_idx = train_test_split(
            range(len(dataset)), 
            test_size=0.1, 
            random_state=42
        )
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        eval_dataset = torch.utils.data.Subset(dataset, eval_idx)
        logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}")
        
    except Exception as e:
        logger.error(f"数据集加载失败: {str(e)}")
        raise

    # 添加损失监控回调（保持不变）
    from transformers import TrainerCallback
    
    class LossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if "loss" in logs:
                logger.info(f"步骤 {state.global_step}: 损失 = {logs['loss']:.4f}, 学习率 = {logs.get('learning_rate', 0):.2e}")
            if "eval_loss" in logs:
                logger.info(f"评估步骤 {state.global_step}: 验证损失 = {logs['eval_loss']:.4f}")
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LossCallback()]  # 添加回调
    )
    
    # 开始训练
    try:
        logger.info("开始QLoRA微调训练...")
        
        # 训练前验证
        logger.info("训练前验证...")
        eval_results = trainer.evaluate()
        logger.info(f"初始验证损失: {eval_results['eval_loss']:.4f}")
        
        # 训练
        trainer.train()
        
        # 训练后评估
        logger.info("训练后评估...")
        eval_results = trainer.evaluate()
        logger.info(f"最终验证损失: {eval_results['eval_loss']:.4f}")
        
        logger.info("训练完成，保存模型...")
        
        # 保存LoRA适配器（仅保存适配器权重）
        model.save_pretrained(output_dir)
        logger.info(f"QLoRA适配器保存至: {output_dir}")
        
        # 注意：QLoRA不需要保存完整量化模型，通常只保存适配器
        
    except Exception as e:
        logger.error(f"训练失败: {str(e)}")
        raise