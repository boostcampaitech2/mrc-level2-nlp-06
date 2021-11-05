from datasets import load_from_disk

#little_datasets = datasets.filter(lambda e, i: i<5, with_indices=True)
import pandas as pd

augment_dict= {
}

def augmentData(datasets, type):
    why_df = pd.read_csv("./backtranslation/why_v1.csv", delimiter=',', engine='python', encoding='CP949')
    how_df = pd.read_csv("./backtranslation/how_csv_v1.csv", delimiter=',', engine='python')
    def check_none(str):
        if str=='none' or str=='None':
            return False
        else :
            return True
    if type%2==1: #1과 3일때만
        for why_data in why_df.iloc:
            origin_data = why_data['origin']
            augment_dict[origin_data]=[]    
            for index, data in enumerate(why_data) : 
                if check_none(data) and index!=0 and data!=origin_data:
                    augment_dict[origin_data].append(data)
    if type>=2:
        for how_data in how_df.iloc:
            origin_data = how_data.name[0]
            augment_dict[origin_data]=[]    
            for index, data in enumerate(how_data.name) : 
                if check_none(data) and index!=0 and  data!=origin_data:
                    augment_dict[origin_data].append(data)
            if check_none(how_data[0]):
                augment_dict[origin_data].append(how_data[0])

    
    def augment_data(examples):     
        global augment_dict
        data = {}         
        for key in examples.keys() :
            data[key]=[]
        augment_len = len(examples['question'])
        for index in range(augment_len):
            len_augment=1
            question = examples['question'][index]
            if question in augment_dict.keys() : #만약 해당 질의가 how나 why dict안에 있으면
                #여기서 각 question들을 넣는 식으로
                augment_question= augment_dict[question] 
                augment_question.append(question)
                data['question']+=augment_question
                len_augment=len(augment_question)         
                for key in examples.keys():
                    if key=='question':
                        continue
                    else :                          
                        temp=examples[key][index]
                        copied_data = [temp for i in range(len_augment)]               
                        data[key]+=copied_data
            else :
                for key in examples.keys() :               
                    data[key]+=[examples[key][index]]
        return data
    augmented_dataset = datasets.map(augment_data, batched=True, remove_columns=datasets.column_names)
    return augmented_dataset