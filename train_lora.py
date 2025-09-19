from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import torch
import logging
import os
import random
import numpy as np
import os
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
    output_dir = './lora_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载学生模型
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False
        )
        logger.info("学生模型加载成功")
    except Exception as e:
        logger.error(f"学生模型加载失败: {str(e)}")
        raise
    
    # 改进的LoRA配置
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
    model.enable_input_require_grads()
    model.train()
    model.print_trainable_parameters()
    
    # Tokenizer
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
    
    # 改进的训练参数
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # 减少epoch数，先观察效果
        per_device_train_batch_size=8,  # 增加批次大小
        gradient_accumulation_steps=4,  # 减少梯度累积步数
        logging_steps=10,
        save_strategy='steps',
        save_steps=100,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        learning_rate=1e-4,  # 大幅增加学习率
        weight_decay=0.01,  # 添加权重衰减
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        dataloader_num_workers=4,  # 增加数据加载工作线程
        report_to='tensorboard',
        remove_unused_columns=True,
        optim="adamw_torch",
        gradient_checkpointing=False,  # 禁用梯度检查点，可能引起数值不稳定
        logging_dir='./logs',
        max_grad_norm=1.0,  # 增加梯度裁剪阈值
        evaluation_strategy="steps",  # 添加评估策略
        eval_steps=100,  # 每100步评估一次
        load_best_model_at_end=True,  # 结束时加载最佳模型
        metric_for_best_model="loss",  # 使用损失作为评估指标
        greater_is_better=False,  # 损失越小越好
        group_by_length=True,  # 按长度分组，提高效率
        prediction_loss_only=True,  # 只计算预测损失
        dataloader_pin_memory=True,  # 固定内存，提高速度
        dataloader_prefetch_factor=2,  # 预取因子
        auto_find_batch_size=True,  # 自动寻找最佳批次大小
    )
    
    # 数据准备
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

    # 添加损失监控回调
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
        logger.info("开始LoRA微调训练...")
        
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
        
        # 保存LoRA适配器
        model.save_pretrained(output_dir)
        logger.info(f"LoRA适配器保存至: {output_dir}")
        
        # 保存完整模型（可选）
        full_model_path = os.path.join(output_dir, "full_model")
        model.save_pretrained(full_model_path)
        tokenizer.save_pretrained(full_model_path)
        logger.info(f"完整模型保存至: {full_model_path}")
        
    except Exception as e:
        logger.error(f"训练失败: {str(e)}")
        raise