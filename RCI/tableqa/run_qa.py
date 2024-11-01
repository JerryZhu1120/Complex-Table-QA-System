import sys
import json
import time

sys.path.append("../")
from datasets.construct_seq import RowConvert, ColumnConvert, Config
from util.args_help import fill_from_args
from tableqa.rci_system import RCISystem, TableQAOptions

start_time = time.time()
with open("../../table_data/test_tables_labeled.json", "r") as f:
    json_data = json.load(f)
# with open("../../table_data/test_data.json", "r") as f:
#     json_data = json.load(f)
print(len(json_data))
with open("../../IM-TQA/data/test_questions copy.json", "r") as f:
    test_questions = json.load(f)
with open("../../IM-TQA/data/dev_questions.json", "r") as f:
    dev_questions = json.load(f)
questions=test_questions+dev_questions
with open("../../IM-TQA/data/lookup/test_lookup.jsonl", "r") as f:
    test_table_data = [json.loads(line.strip()) for line in f]
with open("../../IM-TQA/data/lookup/dev_lookup.jsonl", "r") as f:
    dev_table_data = [json.loads(line.strip()) for line in f]
table_data = test_table_data + dev_table_data
# with open("../../IM-TQA/data/row_col_reps/test_row.jsonl", "r") as f:
#     test_rows = [json.loads(line.strip()) for line in f]
# with open("../../IM-TQA/data/row_col_reps/test_col.jsonl", "r") as f:
#     test_cols = [json.loads(line.strip()) for line in f]
json_data = json_data[-364:-100]
opts1 = TableQAOptions()
fill_from_args(opts1)
rci = RCISystem(opts1)
opts2 = Config()
row = RowConvert(opts2)
col = ColumnConvert(opts2)
correct_count = 0
total_count = 0
error_answer = []
vertical_count=0
vertical_correct_count=0
horizontal_count=0
horizontal_correct_count=0
complex_count=0
complex_correct_count=0
hierarchical_count=0
hierarchical_correct_count=0
for each in json_data:
    row_reps = []
    col_reps = []
    for w in questions:
        if w["table_id"] == each["table_id"]:
            total_count += 1
            question = w["chinese_question"]
            for x in table_data:
                if x["id"] == w["question_id"]:
                    table = x
                    break
            if table["table_type"] == "vertical":
                vertical_count+=1
            elif table["table_type"] == "horizontal":
                horizontal_count+=1
            elif table["table_type"] == "complex":
                complex_count+=1
            elif table["table_type"] == "hierarchical":
                hierarchical_count+=1
            row_reps = row.convert(table)[1]
            col_reps = col.convert(table)[1]
            answer = rci.get_answers_with_reps(
                question, table["table_value"], row_reps, col_reps
            )
            cell_ID_1 = each["cell_ID_matrix"][answer[0]["row_ndx"]][
                answer[0]["col_ndx"]
            ]
            cell_ID_2 = each["cell_ID_matrix"][answer[0]["row_ndx"]][
                answer[0]["col_ndx"]
            ]
            if cell_ID_1 in w["answer_cell_list"] or cell_ID_2 in w["answer_cell_list"]:
                correct_count += 1
                if table["table_type"] == "vertical":
                    vertical_correct_count+=1
                elif table["table_type"] == "horizontal":
                    horizontal_correct_count+=1
                elif table["table_type"] == "complex":
                    complex_correct_count+=1
                elif table["table_type"] == "hierarchical":
                    hierarchical_correct_count+=1
            else:
                error_answer.append(
                    {
                        "question_id": w["question_id"],
                        "question": question,
                        "table": table["table_value"],
                        "answer": answer,
                        "correct_answer_cell_IDs": w["answer_cell_list"],
                        "correct_answers": [table["table_value"][table["target_rows"][i]][table["target_columns"][i]] for i in range(len(table["target_rows"]))]
                    }
                )
            # time.sleep(0.1)
end_time = time.time()
print(end_time - start_time, "s")
print(correct_count / total_count, correct_count, total_count)
print("vertical:",vertical_correct_count / vertical_count, vertical_correct_count, vertical_count)
print("horizontal:",horizontal_correct_count / horizontal_count, horizontal_correct_count, horizontal_count)
print("complex:",complex_correct_count / complex_count, complex_correct_count, complex_count)
print("hierarchical:",hierarchical_correct_count / hierarchical_count, hierarchical_correct_count, hierarchical_count)
with open("test/RGCN-RCI/predict_label/error_answer_1.json", "w") as f:
    json.dump(error_answer, f, indent=4, ensure_ascii=False)
