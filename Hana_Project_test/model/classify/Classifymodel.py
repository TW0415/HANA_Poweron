import torch
from transformers import AutoTokenizer

# 의도 분류 모델 모듈
class ClassifyModel:
    def __init__(self, model_name,tokenizer):

        # 의도 클래스 별 레이블
        self.labels = {0: "수익권과 수익증권", 1: "투자신탁재산", 2: "판매 및 환매", 3: "판매 및 환매"}

        # 의도 분류 모델 불러오기
        self.model = torch.load(model_name, map_location=torch.device("cpu"))
        
        self.tokenizer = tokenizer


    # 의도 클래스 예측
    def predict_class(self, query):
        
        tokenizer = self.tokenizer
        inputs = tokenizer(
        [query],
        max_length=50,
        padding="max_length",
        truncation=True,
        )
        
        with torch.no_grad():
            outputs = self.model(**{k: torch.tensor(v) for k, v in inputs.items()})
            prob = outputs.logits.softmax(dim=1)

        
        predict_class = prob.argmax(1)
        
        return list(map(self.labels.get, prob.argmax(1).tolist()))[0]
    
if __name__ == '__main__':
    main()
