from utils.utils_qa import postprocess_qa_predictions
from transformers import EvalPrediction
def post_processing_fuction_with_setting(data_args, datasets, answer_column_name):
    def post_processing_function(examples, features, predictions, training_args):
            # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
            predictions = postprocess_qa_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                max_answer_length=data_args.max_answer_length,
                output_dir=training_args.output_dir,
            )
            # Metric을 구할 수 있도록 Format을 맞춰줍니다.
            formatted_predictions = [
                # v[0] = (predicted_text, score)
                # predictions: 'id' : [(predicted_text, score), (predicted_text, score), ...]
                {"id": k, "prediction_text":v[0][0], "prediction_cands" : v } for k, v in predictions.items()
            ]
            if training_args.do_predict:
                return formatted_predictions

            elif training_args.do_eval:
                references = [
                    {"id": ex["id"], "answers": ex[answer_column_name]}
                    for ex in datasets
                ]
                return EvalPrediction(
                    predictions=formatted_predictions, label_ids=references
                )
    return post_processing_function