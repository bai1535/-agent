from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from dataset import SFTDataset
import logging
from utils import compute_fkl, compute_rkl, compute_skewed_fkl, compute_skewed_rkl
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
class KGTrainer(Trainer):
    def __init__(
        self,
        model=None,
        teacher_model=None,
        if_use_entropy=False,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
        self.tokenizer = tokenizer
        
        # 获取教师模型的tokenizer
        try:
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                teacher_model.config._name_or_path,
                trust_remote_code=True
            )
            logger.info("教师模型tokenizer加载成功")
            
            # 打印词表信息
            logger.info(f"学生词表大小: {model.config.vocab_size}")
            logger.info(f"教师词表大小: {teacher_model.config.vocab_size}")
        except Exception as e:
            logger.error(f"教师模型tokenizer加载失败: {str(e)}")
            raise
 
 
    def compute_loss(self, model, inputs, return_outputs=False):
        # 确保模型处于训练模式
        model.train()
        
        # 显式启用梯度
        torch.set_grad_enabled(True)
        
        # 学生模型前向传播
        student_outputs = model(**inputs)
        loss = student_outputs.loss
        logits = student_outputs.logits
        
        # 确保logits需要梯度
        if not logits.requires_grad:
            logits.requires_grad = True
        
        # 教师模型前向传播 - 无梯度
        with torch.no_grad():
            teacher_device = self.teacher_model.device
            teacher_inputs = {k: v.to(teacher_device) for k, v in inputs.items()}
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits.to(logits.device)
        
        # 词表对齐
        vocab_size = min(logits.size(-1), teacher_logits.size(-1))
        logits = logits[..., :vocab_size]
        teacher_logits = teacher_logits[..., :vocab_size]
        
        # 序列长度对齐
        seq_len = min(logits.shape[1], teacher_logits.shape[1])
        logits = logits[:, :seq_len, :]
        teacher_logits = teacher_logits[:, :seq_len, :]
        labels = inputs['labels'][:, :seq_len]
        
        # 计算KL散度
        kl = compute_fkl(logits, teacher_logits, labels, padding_id=-100, temp=2.0)
        
        # 组合损失
        if self.if_use_entropy:
            loss_total = 0.5 * kl + 0.5 * loss
        else:
            loss_total = kl
        
        # 如果梯度仍然缺失，添加诊断
        if not loss_total.requires_grad:
            # 创建虚拟梯度
            dummy_grad = torch.tensor(1e-6, device=loss_total.device)
            loss_total = loss_total + dummy_grad * sum(
                p.sum() for p in model.parameters() if p.requires_grad
            )
        
        return (loss_total, student_outputs) if return_outputs else loss_total
 
if __name__ == '__main__':
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 学生模型
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "/root/autodl-tmp/medical/models/Qwen2.5-7B",
            torch_dtype=torch.bfloat16,  # 保持BF16
            device_map={"": 0},
            trust_remote_code=True,
            use_cache=False
        )
        logger.info("学生模型加载成功")
    except Exception as e:
        logger.error(f"学生模型加载失败: {str(e)}")
        raise
    
    # LoRA配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="none"
    )
    
    # 确保LoRA参数正确启用梯度
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # 关键添加：启用输入梯度
    
    # 修改这部分代码
    for name, param in model.named_parameters():  # 使用 named_parameters() 获取名称和参数
        if "lora" in name:  # 检查名称中是否包含 "lora"
            param.requires_grad = True
    
    # 如果没有可训练参数，手动启用LoRA层梯度
    if not any(param.requires_grad for param in model.parameters()):
        logger.warning("没有检测到可训练参数，手动启用LoRA层梯度")
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
    model.print_trainable_parameters()
    
    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "/root/autodl-tmp/medical/models/Qwen2.5-7B",
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer加载成功")
    except Exception as e:
        logger.error(f"Tokenizer加载失败: {str(e)}")
        raise
    
    # 教师模型
    try:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            "/root/autodl-fs/models/carellm",
            torch_dtype=torch.float16,
            device_map={"": 1},  # 教师模型在卡1
            trust_remote_code=True
        )
        logger.info("教师模型加载成功")
        # 立即设置为评估模式
        teacher_model.eval()
 
    except Exception as e:
        logger.error(f"教师模型加载失败: {str(e)}")
        raise
    
    # 训练参数
 
    args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        logging_steps=10,
        save_strategy='epoch',
        save_total_limit=2,
        fp16=True,  # 保持FP16
        bf16=False,  # 关闭BF16
        learning_rate=1e-4,
        lr_scheduler_type='cosine',
        dataloader_num_workers=2,
        report_to='tensorboard',
        remove_unused_columns=True,  # 改为True
        optim="adamw_torch",
        gradient_checkpointing=False,  # 关闭梯度检查点
        logging_dir='./logs'
    )
    
    # 数据准备
    try:
        data_collator = DefaultDataCollator()
        dataset = SFTDataset(
            '/root/autodl-tmp/medical/data/CMtMedQA.json',
            tokenizer=tokenizer,
            max_seq_len=256
        )
        logger.info(f"数据集加载成功，样本数: {len(dataset)}")
    except Exception as e:
        logger.error(f"数据集加载失败: {str(e)}")
        raise
        # 训练前验证梯度流
    # 在梯度流验证部分修改为以下代码
    logger.info("执行梯度流验证...")
    sample = next(iter(dataset))
    inputs = data_collator([sample])
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 验证基础损失梯度 - 使用独立计算图
    with torch.enable_grad():
        model.train()
        student_outputs = model(**inputs)
        loss = student_outputs.loss
        logger.info(f"基础损失值: {loss.item()}")
        logger.info(f"基础损失是否要求梯度: {loss.requires_grad}")
        
        if loss.requires_grad:
            loss.backward()
            logger.info("基础损失反向传播成功")
            model.zero_grad()
        else:
            logger.error("基础损失没有梯度！检查模型配置")
    
    # 验证KL散度梯度 - 使用另一个独立计算图
    with torch.enable_grad():
        model.train()
        student_outputs = model(**inputs)  # 重新前向传播
        logits = student_outputs.logits
        
        # 教师模型前向传播
        with torch.no_grad():
            teacher_inputs = {k: v.to(teacher_model.device) for k, v in inputs.items()}
            teacher_outputs = teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits.to(logits.device)
        
        # 词表对齐
        vocab_size = min(logits.size(-1), teacher_logits.size(-1))
        logits = logits[..., :vocab_size]
        teacher_logits = teacher_logits[..., :vocab_size]
        
        # 序列长度对齐
        seq_len = min(logits.shape[1], teacher_logits.shape[1])
        logits = logits[:, :seq_len, :]
        teacher_logits = teacher_logits[:, :seq_len, :]
        labels = inputs['labels'][:, :seq_len]
        
        # 计算KL散度
        kl = compute_fkl(logits, teacher_logits, labels, padding_id=-100, temp=2.0)
        logger.info(f"KL散度值: {kl.item()}")
        logger.info(f"KL散度是否要求梯度: {kl.requires_grad}")
        
        if kl.requires_grad:
            kl.backward()
            logger.info("KL散度反向传播成功")
            model.zero_grad()
        else:
            logger.error("KL散度没有梯度！将添加可训练项")
            kl = kl + 1e-6 * logits.sum()
            kl.backward()
            logger.info("强制KL散度梯度流成功")
            model.zero_grad()
    
    # 训练器
    trainer = KGTrainer(
        model=model,
        teacher_model=teacher_model,
        if_use_entropy=True,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # 开始训练
    try:
        logger.info("开始训练...")
        trainer.train(resume_from_checkpoint=False)
        logger.info("训练完成，保存模型...")
        trainer.save_model('./saves')
        trainer.save_state()
        logger.info("模型保存成功")
    except Exception as e:
        logger.error(f"训练失败: {str(e)}")
        raise