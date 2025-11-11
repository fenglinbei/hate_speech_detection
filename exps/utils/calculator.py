class F1Calculator:
    """F1分数计算工具类"""
    
    @staticmethod
    def calculate_f1(precision, recall):
        """计算F1分数（F1是精确率和召回率的调和平均数）[6](@ref)"""
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def calculate_precision(tp, fp):
        """计算精确率：TP / (TP + FP) [7](@ref)"""
        if tp + fp == 0:
            return 0
        return tp / (tp + fp)
    
    @staticmethod
    def calculate_recall(tp, fn):
        """计算召回率：TP / (TP + FN) [7](@ref)"""
        if tp + fn == 0:
            return 0
        return tp / (tp + fn)
    
    @staticmethod
    def f1_from_confusion_matrix(tp, fp, fn):
        """从混淆矩阵的基本元素计算F1分数[6](@ref)"""
        precision = F1Calculator.calculate_precision(tp, fp)
        recall = F1Calculator.calculate_recall(tp, fn)
        return F1Calculator.calculate_f1(precision, recall)
    
    