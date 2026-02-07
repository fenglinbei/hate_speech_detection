import math

def allocate_class_num(i, weights_dict, reverse: bool = False):
    # 初始化字典用于存储初始分配值和小数部分
    initial_allocation = {}
    fractions = []
    total_integer = 0

    if reverse:
        weights_dict = {k: 1/v for k, v in weights_dict.items()}
        total_weight = sum(weights_dict.values())
        weights_dict = {k: (v / total_weight) * 100 for k, v in weights_dict.items()}
    
    # 遍历权重字典，计算每个类别的理论值、整数部分和小数部分
    for key, weight in weights_dict.items():
        theory_value = i * weight / 100.0
        integer_part = math.floor(theory_value)
        fraction = theory_value - integer_part
        
        initial_allocation[key] = integer_part
        fractions.append((key, fraction))
        total_integer += integer_part
    
    # 计算剩余量（需要分配的额外单位数）
    remaining = i - total_integer
    
    # 根据小数部分降序排序（小数部分相同则按键名字母顺序升序）
    fractions.sort(key=lambda x: (-x[1], x[0]))
    print(fractions)
    print(remaining)
    
    # 将剩余量分配给小数部分最大的前 remaining 个类别
    for idx in range(remaining):
        if idx >= len(fractions):
            idx = idx % len(fractions)
        key = fractions[idx][0]
        initial_allocation[key] += 1
    
    return initial_allocation

if __name__ == "__main__":
    weights = {
        "A": 19,
        "B": 24,
        "C": 8,
        "D": 15,
        "E": 6,
    }
    total = 10
    allocation = allocate_class_num(total, weights)
    print(f"Allocation for total {total} with weights {weights}: {allocation}")
    
    allocation_reverse = allocate_class_num(total, weights, reverse=True)
    print(f"Reverse Allocation for total {total} with weights {weights}: {allocation_reverse}")