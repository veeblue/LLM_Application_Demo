import json

# 输入 xtuner 格式数据路径
input_file = "./data/style_chat_data_20250707_214748.json"
# 输出 llamafactory 格式路径
output_file = "./data/train_data_history.json"

with open(input_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

converted = []

for item in raw_data:
    instruction = item.get("user", "").strip()
    style = item.get("style", "").strip()
    output = item.get("assistant", "").strip()
    # 拼接风格
    if style:
        output = f"{style}\n{output}"
    # 处理 history
    history = []
    if "history" in item and isinstance(item["history"], list):
        for turn in item["history"]:
            if isinstance(turn, dict):
                u = turn.get("user", "").strip()
                a = turn.get("assistant", "").strip()
            elif isinstance(turn, list) and len(turn) == 2:
                u, a = turn
            else:
                continue
            history.append([u, a])
    converted.append({
        "instruction": instruction,
        "input": "",
        "output": output,
        "history": history
    })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

print(f"✅ 转换完成，共 {len(converted)} 条，输出文件：{output_file}")