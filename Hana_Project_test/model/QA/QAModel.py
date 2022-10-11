import torch
import json
from utils.DataBase import Database
from transformers import AutoTokenizer
from tqdm import tqdm


# QA 모듈

def predict(question, context, model, tokenizer):
        
    truncated_query = tokenizer.encode(
        question,
        add_special_tokens=False,
        truncation=True,
        max_length=32
    )

    inputs = tokenizer.encode_plus(
        text=truncated_query,
        text_pair=context,
        truncation="only_second",
        padding="max_length",
        max_length=128,
        return_token_type_ids=True,
    )

    with torch.no_grad():

        outputs = model(**{k: torch.tensor([v]) for k, v in inputs.items()})

        start_pred = outputs.start_logits.argmax(dim=-1).item()

        end_pred = outputs.end_logits.argmax(dim=-1).item()

        pred_text = tokenizer.decode(inputs['input_ids'][start_pred:end_pred+1]).replace('[CLS]','').replace('[SEP]','.').replace('[UNK]','')

        start_probabilities = torch.nn.functional.softmax(outputs.start_logits, dim=-1)[0]

        end_probabilities = torch.nn.functional.softmax(outputs.end_logits, dim=-1)[0]

        scores = start_probabilities[:, None] * end_probabilities[None, :]

        scores = torch.triu(scores)

        max_index = scores.argmax().item()
        start_index = max_index // scores.shape[1]
        end_index = max_index % scores.shape[1]

        score = scores[start_index, end_index].item()

        if pred_text == '':
            score = 0
            pred_text = "약관서에서 해당 내용을 찾을 수 없습니다."

        return score, pred_text


class QAmodel:
    
    def __init__(self, model_name, DB_values, tokenizer):
        self.DB_values = DB_values
        
        self.model = torch.load(model_name, map_location=torch.device("cpu"))
        
        self.tokenizer = tokenizer
    
    # 질의응답
    def QA_search(self, question):
        
        dataset = self.DB_values
        max_value = 0 #초기값 설정
        pred_text = "약관서에서 해당 내용을 찾을 수 없습니다." #초기값 설정
        score = 0
        content = ''
        fund = ''
        classes =''
        
        for data in tqdm(dataset):
            
            score, pred_text_i = predict(question, data['content'], self.model, self.tokenizer)
                
        
            if score > max_value:
                
                max_value = score
                
                pred_text = pred_text_i
                
                content = data['content']
                
                fund = data['fund']
                
                classes = data['class_raw']
            
        result = json.dumps({"Answer": pred_text,
                             "Prob": score,
                             "content": content,
                             "fund": fund,
                             "class": classes}, 
                            ensure_ascii=False, 
                            indent="\t")
        
        return result
    
if __name__ == '__main__':
    main()
