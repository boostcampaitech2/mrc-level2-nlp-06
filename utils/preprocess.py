
def prepare_datasets_with_setting(tokenizer, datasets, training_args, data_args, max_seq_length):        
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names    
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]
    pad_on_right = tokenizer.padding_side == "right"
    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    
    def prepare_train_features(examples):
            # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
            # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                #return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
                padding="max_length" if data_args.pad_to_max_length else False,
            )

            # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
            # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
            offset_mapping = tokenized_examples.pop("offset_mapping")

            # 데이터셋에 "start position", "enc position" label을 부여합니다.
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []

            for i, offsets in enumerate(offset_mapping):
                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index

                # sequence id를 설정합니다 (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)

                # 하나의 example이 여러개의 span을 가질 수 있습니다.
                sample_index = sample_mapping[i]
                answers = examples[answer_column_name][sample_index]

                # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
                if len(answers["answer_start"]) == 0:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # text에서 정답의 Start/end character index
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])

                    # text에서 current span의 Start token index
                    token_start_index = 0
                    while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                        token_start_index += 1

                    # text에서 current span의 End token index
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                        token_end_index -= 1

                    # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
                    if not (
                        offsets[token_start_index][0] <= start_char
                        and offsets[token_end_index][1] >= end_char
                    ):
                        tokenized_examples["start_positions"].append(cls_index)
                        tokenized_examples["end_positions"].append(cls_index)
                    else:
                        # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                        # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                        while (
                            token_start_index < len(offsets)
                            and offsets[token_start_index][0] <= start_char
                        ):
                            token_start_index += 1
                        tokenized_examples["start_positions"].append(token_start_index - 1)
                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized_examples["end_positions"].append(token_end_index + 1)

            return tokenized_examples
    
    def prepare_validation_features(examples):   
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            #return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        ################ prepare train feature에서 가지고 온 코드 ################
        # 아래 코드의 역할: answer의 start position, end position을 뽑아서 eval input에게 함께 던져주기 위함이다.
        # train의 중간 중간에 evaluation loss을 계산하기 위해서 이 부분이 필요합니다. 
        # 나중에 적절한 시기가 되면 prepare train feature 함수의 동일한 부분과 함께 리팩토링을 수행하면 좋을 것 같습니다 :)

        # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
        # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
        offset_mapping = tokenized_examples["offset_mapping"]
        # 요 밑으로 함수화가 가능할 것 같아요 :)
        # prepare train feature에서는 offset_mapping = tokenized_examples.pop("offset_mapping") 이렇게 되어 있음.
        # 데이터셋에 "start position", "enc position" label을 부여합니다.
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index

            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # text에서 정답의 Start/end character index
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # text에서 current span의 Start token index
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # text에서 current span의 End token index
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                    # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
        ################ prepare train feature에서 가지고 온 코드 끝 ################
        return tokenized_examples
    dataset_dict = {}
    if training_args.do_train:        
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset =datasets["train"]
        column_names = train_dataset.column_names
        # dataset에서 train feature를 생성합니다.
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        dataset_dict['train'] = train_dataset
    # Validation preprocessing  
    if training_args.do_eval:        
        eval_dataset = datasets["validation"]
        column_names = eval_dataset.column_names    
        # Validation Feature 생성
        print("Do EVAL")
        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        dataset_dict['validation'] =eval_dataset

    return dataset_dict, answer_column_name

