import sys
import json
import time

sys.path.append("../")
from util.args_help import fill_from_args
from tableqa.rci_system import RCISystem, TableQAOptions

start_time = time.time()
opts = TableQAOptions()
fill_from_args(opts)
rci = RCISystem(opts)
with open("../../table_data/test_data.json", "r") as f:
    json_data = json.load(f)
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
json_data = json_data[:-100]

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
                if x["table_id"] == each["table_id"]:
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
            header = table["table_value"][0]
            rows = table["table_value"][1:]
            answer = rci.get_answers(question, header, rows)
            cell_ID_1 = each["cell_ID_matrix"][answer[0]["row_ndx"] + 1][
                answer[0]["col_ndx"]
            ]
            cell_ID_2 = each["cell_ID_matrix"][answer[1]["row_ndx"] + 1][
                answer[1]["col_ndx"]
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
                        "correct_answer_cell_ID": w["answer_cell_list"],
                        "correct_answer": [table["table_value"][target_rows[i]][target_columns[i]] for i in range(len(target_rows))]
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
with open("test/RCI/bert/error_answer_2.json", "w") as f:
    json.dump(error_answer, f, indent=4, ensure_ascii=False)
# and separately for rows and columns
# print(rci.get_answer_columns("Who won the race in 2021?", header, rows))
# print(rci.get_answer_rows("Who won the race in 2021?", header, rows))
