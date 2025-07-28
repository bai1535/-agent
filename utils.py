import torch
 
# 计算前向kl散度
def compute_fkl(logits, teacher_logits, target, padding_id, reduction="sum", temp=1.0):
    # 确保学生模型部分保留梯度
    logits = logits / temp
    student_log_probs = torch.log_softmax(logits, -1)
    
    # 教师模型部分不需要梯度
    with torch.no_grad():
        teacher_logits = teacher_logits / temp
        teacher_probs = torch.softmax(teacher_logits, -1)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1)
    
    # 计算KL散度 - 确保使用学生模型的输出
    kl = teacher_probs * (teacher_log_probs - student_log_probs)
    kl = kl.sum(-1)
    
    if reduction == "sum":
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill(pad_mask, 0.0)
        kl = kl.sum()
    
    return kl
 
# 同样修改其他KL函数，移除原地操作(masked_fill_改为masked_fill)
# 计算反向kl散度
def compute_rkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id,
        reduction="sum", 
        temp = 1.0
    ):
        logits = logits / temp
        teacher_logits = teacher_logits / temp
 
        probs = torch.softmax(logits, -1)
        log_probs = torch.log_softmax(logits, -1)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1)
        kl = (probs * (log_probs - teacher_log_probs))
        kl = kl.sum(-1)
        if reduction == "sum":
            pad_mask = target.eq(padding_id)
            kl = kl.masked_fill_(pad_mask, 0.0)
            kl = kl.sum()
        return kl
 
# 计算偏向前kl散度
def compute_skewed_fkl(
        logits, 
        teacher_logits, 
        target, 
        padding_id, 
        reduction="sum", 
        temp = 1.0,
        skew_lambda = 0.1
    ):
        logits = logits / temp
        teacher_logits = teacher_logits / temp
 
        probs = torch.softmax(logits, -1)
        teacher_probs = torch.softmax(teacher_logits, -1)
        mixed_probs = skew_lambda * teacher_probs + (1 - skew_lambda) * probs
        mixed_log_probs = torch.log(mixed_probs)
        teacher_log_probs = torch.log_softmax(teacher_logits, -1)
        kl = (teacher_probs * (teacher_log_probs - mixed_log_probs))
        kl = kl.sum(-1)
        if reduction == "sum":
            pad_mask = target.eq(padding_id)
            kl = kl.masked_fill_(pad_mask, 0.0)
            kl = kl.sum()
 
            
        return kl
# 计算偏向反kl散度    
def compute_skewed_rkl(
    logits, 
    teacher_logits, 
    target,
    padding_id,
    reduction="sum", 
    temp = 1.0,
    skew_lambda = 0.1
):
    logits = logits / temp
    teacher_logits = teacher_logits / temp
    
    probs = torch.softmax(logits, -1)
    teacher_probs = torch.softmax(teacher_logits, -1)
    mixed_probs = (1 - skew_lambda) * teacher_probs + skew_lambda * probs
    mixed_log_probs = torch.log(mixed_probs)
    log_probs = torch.log_softmax(logits, -1)
    kl = (probs * (log_probs - mixed_log_probs))
    kl = kl.sum(-1)
    
    if reduction == "sum":
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill_(pad_mask, 0.0)
        kl = kl.sum()
 
 
    return kl