import hanlp

class HanLP:

    def __init__(self):
        self.tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        # self.tok.dict_combine = True
        self.tok.config.force_compact = True

    def cut(self, text):
        """
        Simulates the HanLP cut method for tokenizing text.
        """
        return self.tok(text)

if __name__ == "__main__":
    # Example usage
    hanlp = HanLP()
    text = "这可咋整。反对种族歧视的没有几个（至少我没看见几个）替受害的中国人说话的。倒是一帮替黑乐色洗白的。"
    tokens = hanlp.cut(text)
    print(tokens)
    print("Tokens:", [str(token) for token in tokens])