import json

# 输入 xtuner 格式数据路径
input_file = "/Users/yee/vscode/DailyCode/demo/EmotionalDialogueModel /data/style_chat_data_20250707_214748.json"
# 输出 llamafactory 格式路径
output_file = "./data/train_data.json"

with open(input_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

converted = []

for item in raw_data:
    instruction = item.get("user", "").strip()
    style = item.get("style", "").strip()
    response = item.get("assistant", "").strip()

    # 如果有风格字段，将其拼接在输出开头
    if style:
        output = f"{style}\n{response}"
    else:
        output = response

    converted.append({
        "instruction": instruction,
        "input": "",
        "output": output
    })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

print(f"✅ 转换完成，共 {len(converted)} 条，输出文件：{output_file}")