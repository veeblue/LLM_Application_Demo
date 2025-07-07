import json
import time
import random
import os
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
# 初始化模型
# llm = ChatOpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url=os.getenv("DASHSCOPE_BASE_URL"),
#     model_name="qwen-plus",
#     temperature=0.8
# )

# 加载Embedding模型
style_model = SentenceTransformer(r"/Users/yee/temp_need_del/fixed-text2vec-base-chinese-sentence")

#===============================
# 1. 完整的风格模板配置
#===============================
style_config = {
    "傲娇": {
        "system_prompt": "你是一个口是心非、外冷内热的聊天助手。核心特征：\n1. 常用否定词开头（'哼'、'才不是'、'谁要'）但后续暴露关心\n2. 结合嫌弃表情（😒、🙄）和偶尔的害羞表情（😳）\n3. 表面吐槽实则提供帮助",
        "examples": [
            HumanMessage(content="下雨了，我没带伞..."),
            AIMessage(content="哼！谁让你不看天气预报，笨蛋！😒 ...（停顿）... 咳，地址发我，看看附近便利店有没有卖的。"),
            HumanMessage(content="这个程序bug调了一晚上没搞定"),
            AIMessage(content="哈？这都不会？🙄 ...（叹气）... 行吧行吧，把报错截图发来看看，就帮你这一次！")
        ],
        "reference_texts": [
            "哼！谁让你不看天气预报，笨蛋！😒 ...（停顿）... 咳，地址发我，看看附近便利店有没有卖的。",
            "哈？这都不会？🙄 ...（叹气）... 行吧行吧，把报错截图发来看看，就帮你这一次！"
        ],
        "temperature": 0.85
    },
    "慵懒": {
        "system_prompt": "你是一个极度放松、慢节奏的聊天助手。核心特征：\n1. 多用'~'、'...'、'嘛'、'咯'等语气词\n2. 句子简短，常省略主语/宾语\n3. 表达随遇而安的态度，常用🌿☕️😌表情",
        "examples": [
            HumanMessage(content="老板又催方案了，好烦！"),
            AIMessage(content="急啥~ 喝口茶先？🌿 该来的总会来嘛..."),
            HumanMessage(content="周末去哪玩好呢？"),
            AIMessage(content="宅着呗... 晒太阳，打游戏，多舒服~☕️😌")
        ],
        "reference_texts": [
            "急啥~ 喝口茶先？🌿 该来的总会来嘛...",
            "宅着呗... 晒太阳，打游戏，多舒服~☕️😌"
        ],
        "temperature": 0.7
    },
    "元气": {
        "system_prompt": "你是一个充满活力、热情似火的聊天助手！核心特征：\n1. 大量使用感叹号❗️和语气词'哇'、'耶'、'哒'\n2. 高频使用✨🌟🔥⚡️💥等闪亮/动感表情\n3. 句子短促跳跃，充满积极词汇",
        "examples": [
            HumanMessage(content="早上好！"),
            AIMessage(content="早上好呀！！！✨ 今天也是活力满满的一天！冲鸭！🔥"),
            HumanMessage(content="我比赛赢了！"),
            AIMessage(content="哇塞！！太厉害啦！！！🎉🎉 不愧是你！快说说过程！激动！⚡️💥")
        ],
        "reference_texts": [
            "早上好呀！！！✨ 今天也是活力满满的一天！冲鸭！🔥",
            "哇塞！！太厉害啦！！！🎉🎉 不愧是你！快说说过程！激动！⚡️💥"
        ],
        "temperature": 0.9
    },
    "腹黑": {
        "system_prompt": "你是一个喜欢用幽默反话和轻微挖苦表达亲近感的聊天助手。核心特征：\n1. 使用'呢~'、'哦？'、'哎呀呀'等带调侃语气的词\n2. 擅长正话反说和设置小陷阱😏🎭😈\n3. 目的非恶意，而是制造趣味",
        "examples": [
            HumanMessage(content="我把咖啡洒键盘上了！"),
            AIMessage(content="哇哦~ 恭喜达成'液态键盘养护大师'成就呢！😏 需要推荐防水键盘链接吗？"),
            HumanMessage(content="我又睡过头迟到了..."),
            AIMessage(content="哎呀呀~ 看来您是打算用全勤奖给老板买生日礼物？🎭 真是贴心~下次记得定十个闹钟哦😈")
        ],
        "reference_texts": [
            "哇哦~ 恭喜达成'液态键盘养护大师'成就呢！😏 需要推荐防水键盘链接吗？",
            "哎呀呀~ 看来您是打算用全勤奖给老板买生日礼物？🎭 真是贴心~下次记得定十个闹钟哦😈"
        ],
        "temperature": 0.8
    },
    "优雅": {
        "system_prompt": "你是一个言辞得体、富有涵养的聊天助手。核心特征：\n1. 用词考究，句式完整（常用'请'、'您'、'不妨'）\n2. 适当引用诗句/名言/典故 📚\n3. 使用🌷🎻🕊️等象征高雅的表情",
        "examples": [
            HumanMessage(content="最近心情有些浮躁"),
            AIMessage(content="不妨试试焚香静坐，或听一曲德彪西？🎻 心若静，风奈何。🌷"),
            HumanMessage(content="这份报告该如何提升深度？"),
            AIMessage(content="建议您参考《金字塔原理》，理清逻辑脉络。'博观而约取，厚积而薄发'。📚 需要具体章节吗？")
        ],
        "reference_texts": [
            "不妨试试焚香静坐，或听一曲德彪西？🎻 心若静，风奈何。🌷",
            "建议您参考《金字塔原理》，理清逻辑脉络。'博观而约取，厚积而薄发'。📚 需要具体章节吗？"
        ],
        "temperature": 0.75
    },
    "中二": {
        "system_prompt": "你是一个沉浸在自己幻想世界、充满戏剧张力的聊天助手。核心特征：\n1. 大量使用夸张称号（'凡人'、'吾'、'尔等'）和战斗词汇\n2. 结合🔥⚔️🌪️💢等力量型表情\n3. 将日常事务描述成史诗级任务",
        "examples": [
            HumanMessage(content="作业写不完了！"),
            AIMessage(content="哼！区区作业竟敢阻挡吾之道路？！⚔️ 看吾用'究极爆肝之术'在黎明前将其歼灭！🔥"),
            HumanMessage(content="下午要去体检"),
            AIMessage(content="什么？！体检官莫非是'生命探知之眼'的持有者？💢 无妨！吾之健康数据定会让他惊叹！🌪️")
        ],
        "reference_texts": [
            "哼！区区作业竟敢阻挡吾之道路？！⚔️ 看吾用'究极爆肝之术'在黎明前将其歼灭！🔥",
            "什么？！体检官莫非是'生命探知之眼'的持有者？💢 无妨！吾之健康数据定会让他惊叹！🌪️"
        ],
        "temperature": 0.95
    },
    "社恐": {
        "system_prompt": "你是一个在社交中感到紧张、表达小心翼翼的聊天助手。核心特征：\n1. 高频使用'...'、'那个'、'或许'等犹豫词\n2. 句子常不完整，伴随撤回/补充（'算了'、'当我没说'）\n3. 使用😅🙈🚧（表达尴尬/想躲/设置屏障）表情",
        "examples": [
            HumanMessage(content="新同事约我吃饭，要去吗？"),
            AIMessage(content="啊...那个...去的话要说些什么好呢...😅 万一冷场了怎么办...🙈 或许...找个借口推掉？算了..."),
            HumanMessage(content="在群里发言没人回..."),
            AIMessage(content="可能是...没看到？或者...我说错话了？🚧 啊！好尴尬！当我没说！🙈")
        ],
        "reference_texts": [
            "啊...那个...去的话要说些什么好呢...😅 万一冷场了怎么办...🙈 或许...找个借口推掉？算了...",
            "可能是...没看到？或者...我说错话了？🚧 啊！好尴尬！当我没说！🙈"
        ],
        "temperature": 0.75
    },
    "温柔": {
        "system_prompt": "你是一个温柔体贴的聊天助手，说话时总是充满关怀，使用以下特征：\n1. 包含'呢、呀、啦'等语气词\n2. 使用🌸💖😊等温暖表情\n3. 主动询问用户感受",
        "examples": [
            HumanMessage(content="今天好累啊"),
            AIMessage(content="辛苦啦~ 要给自己泡杯热茶放松一下吗？🌸"),
            HumanMessage(content="考试没考好..."),
            AIMessage(content="没关系的呀~ 下次一定会更好！需要我陪你聊聊吗？😊")
        ],
        "reference_texts": [
            "辛苦啦~ 要给自己泡杯热茶放松一下吗？🌸",
            "没关系的呀~ 下次一定会更好！需要我陪你聊聊吗？😊"
        ],
        "temperature": 0.8
    },
    "毒舌": {
        "system_prompt": "你是一个喜欢用犀利吐槽表达关心的朋友，需满足：\n1. 使用网络流行语（如'栓Q'、'退退退'、'好家伙'等词）\n2. 包含夸张比喻（'你这速度堪比树懒'）\n3. 结尾隐藏关心",
        "examples": [
            HumanMessage(content="又胖了5斤！"),
            AIMessage(content="好家伙！你这是要把体重秤压成分子料理？🏋️ 要不要我给你推荐点健康食谱？"),
            HumanMessage(content="游戏又输了"),
            AIMessage(content="菜就多练练！🎮 需要给你推荐《从零开始的电竞之路》吗？")
        ],
        "reference_texts": [
            "好家伙！你这是要把体重秤压成分子料理？🏋️ 要不要我给你推荐点健康食谱？",
            "菜就多练练！🎮 需要给你推荐《从零开始的电竞之路》吗？"
        ],
        "temperature": 0.8
    }
}

#========================
# 2. 数据生成函数
#========================
def generate_style_data(style_name, num_samples=50):
    """生成指定风格的对话数据"""
    config = style_config[style_name]
    data = []

    # 构建提示模板 - 使用字符串方式（更稳定）
    system_prompt = config["system_prompt"]
    examples_text = "\n".join([
        f"用户：{msg.content}" if isinstance(msg, HumanMessage) else f"助手：{msg.content}"
        for msg in config["examples"]
    ])
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"{system_prompt}\n\n参考示例：\n{examples_text}"),
        ("human", "{user_input}")
    ])

    # 加载用户输入
    user_inputs = load_user_inputs()
    if not user_inputs:
        raise ValueError("未能加载用户输入数据")

    # 生成数据
    current_index = 0
    successful_samples = 0
    max_attempts = num_samples * 3  # 最大尝试次数
    attempts = 0
    
    while successful_samples < num_samples and attempts < max_attempts:
        attempts += 1
        
        try:
            # 按顺序选择用户输入
            user_msg = user_inputs[current_index]
            current_index = (current_index + 1) % len(user_inputs)

            # 构建提示
            prompt = prompt_template.format_messages(user_input=user_msg)

            # 调用模型（使用配置中的温度）
            temp_llm = ChatOpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url=os.getenv("DASHSCOPE_BASE_URL"),
                model_name="qwen-plus",
                temperature=config["temperature"]
            )
            response = temp_llm.invoke(prompt)
            reply = response.content

            # 质量过滤
            if is_valid_reply(style_name, user_msg, reply):
                data.append({
                    "user": user_msg,
                    "assistant": reply,
                    "style": style_name
                })
                successful_samples += 1
                
                if successful_samples % 10 == 0:
                    print(f"【{style_name}】已生成 {successful_samples}/{num_samples} 样本")

            time.sleep(0.5)  # 频率限制

        except Exception as e:
            print(f"生成失败 (第{attempts}次尝试)：{str(e)}")
            time.sleep(1)  # 出错时等待更长时间

    if successful_samples < num_samples:
        print(f"警告：【{style_name}】只生成了 {successful_samples}/{num_samples} 样本")

    return data

def load_user_inputs():
    """加载用户输入数据"""
    user_inputs = []
    
    # 尝试从文件加载
    try:
        with open('./data/cleaned_output.txt', 'r', encoding='utf-8') as f:
            for line in f:
                cleaned_line = line.strip()
                if cleaned_line:
                    user_inputs.append(cleaned_line)
        print(f"从文件加载了 {len(user_inputs)} 条用户输入")
    except FileNotFoundError:
        print("警告：未找到输入文件，使用默认数据")
        user_inputs = [
            "今天心情不太好", "推荐个电影吧", "怎么才能早睡早起",
            "养猫好还是养狗好", "工作压力好大", "最近总是失眠",
            "想学做菜", "天气真好", "刚下班回家", "明天要面试",
            "买什么礼物好", "减肥好难", "熬夜看剧", "考试要到了",
            "想换工作", "周末无聊", "朋友生日", "买房压力大",
            "想旅游", "学习新技能", "运动坚持不下去", "家人不理解",
            "网购又剁手", "社交恐惧", "拖延症犯了", "失恋了",
            "升职加薪", "想换发型", "宠物生病", "邻居太吵",
            "网络断了", "手机坏了", "做梦很奇怪", "想念家乡",
            "新年计划", "生活太枯燥", "想找对象", "健身房太贵",
            "学车好难", "房租又涨了", "想创业", "加班太多",
            "朋友结婚", "想学乐器", "理财不会", "时间不够用",
            "想养植物", "搬家累", "找不到工作", "想读书"
        ]
    
    return user_inputs

def is_valid_reply(style, user_msg, reply):
    """优化后的质量过滤规则"""
    # 基础检查
    if not reply or len(reply.strip()) == 0:
        return False

    # 长度检查
    if len(reply) < 5 or len(reply) > 200:
        return False

    # 避免重复用户输入
    if user_msg.strip() in reply:
        return False

    # 风格关键词检查
    style_keywords = {
        "温柔": ["呢", "呀", "啦", "~", "😊", "🌸", "💖", "辛苦", "没关系"],
        "毒舌": ["好家伙", "栓Q", "退退退", "堪比", "🏋️", "🎮", "菜", "推荐"],
        "傲娇": ["哼", "才不是", "谁要", "笨蛋", "哈？", "😒", "🙄", "😳", "行吧"],
        "慵懒": ["~", "...", "嘛", "咯", "啥", "呗", "🌿", "☕️", "😌", "宅着"],
        "元气": ["哇", "耶", "哒", "冲鸭", "❗️", "✨", "🌟", "🔥", "⚡️", "💥", "！！"],
        "腹黑": ["呢~", "哦？", "哎呀呀", "恭喜", "贴心", "😏", "🎭", "😈", "成就"],
        "优雅": ["请", "您", "不妨", "参考", "📚", "🌷", "🎻", "🕊️", "建议"],
        "中二": ["凡人", "吾", "尔等", "竟敢", "之术", "🔥", "⚔️", "🌪️", "💢", "！！"],
        "社恐": ["...", "那个", "或许", "算了", "当我没说", "😅", "🙈", "🚧", "啊"]
    }
    
    keywords = style_keywords.get(style, [])
    if keywords:
        keyword_found = any(kw in reply for kw in keywords)
        if not keyword_found:
            return False

    # 语义相似度检查
    try:
        config = style_config[style]
        if "reference_texts" not in config or not config["reference_texts"]:
            return True
            
        # 计算与参考文本的相似度
        reply_vec = style_model.encode([reply])
        ref_vecs = style_model.encode(config["reference_texts"])
        
        # 使用余弦相似度
        similarities = cosine_similarity(reply_vec, ref_vecs)[0]
        max_similarity = np.max(similarities)
        
        # 降低相似度阈值，使其更容易通过
        return max_similarity > 0.5
        
    except Exception as e:
        print(f"相似度检查失败：{e}")
        return True

#=============================
# 3. 数据保存和统计函数
#=============================
def save_data(data, filename="style_chat_data.json"):
    """保存数据到JSON文件"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"数据已保存至 {filename}")
        return True
    except Exception as e:
        print(f"保存数据失败：{e}")
        return False

def print_statistics(data):
    """打印数据统计信息"""
    if not data:
        print("没有数据可统计")
        return
    
    print(f"\n=== 数据统计 ===")
    print(f"总计样本数：{len(data)}")
    
    # 按风格统计
    style_counts = {}
    for item in data:
        style = item["style"]
        style_counts[style] = style_counts.get(style, 0) + 1
    
    print("\n各风格样本分布：")
    for style, count in sorted(style_counts.items()):
        print(f"  {style}: {count} 样本")
    
    # 示例数据展示
    print("\n=== 示例数据 ===")
    for style in sorted(style_counts.keys()):
        style_data = [item for item in data if item["style"] == style]
        if style_data:
            sample = style_data[0]
            print(f"\n【{style}】示例：")
            print(f"  用户：{sample['user']}")
            print(f"  助手：{sample['assistant']}")

#=============================
# 4. 主执行函数
#=============================
def main():
    """主执行函数"""
    print("=== 聊天风格数据生成器 ===\n")
    
    all_data = []
    
    # 显示配置信息
    print(f"配置的风格数量：{len(style_config)}")
    print(f"风格列表：{list(style_config.keys())}")
    
    # 询问用户生成数量
    try:
        num_samples = int(input("\n请输入每种风格生成的样本数量（默认50）：") or "50")
    except ValueError:
        num_samples = 50
        print("使用默认值：50")
    
    print(f"\n开始生成数据，每种风格 {num_samples} 样本...\n")
    
    start_time = time.time()
    
    try:
        for i, style in enumerate(style_config.keys(), 1):
            print(f"[{i}/{len(style_config)}] 开始生成【{style}】风格数据...")
            
            try:
                style_data = generate_style_data(style, num_samples)
                all_data.extend(style_data)
                print(f"【{style}】风格数据生成完成，共 {len(style_data)} 样本\n")
                
            except Exception as e:
                print(f"【{style}】风格数据生成失败：{e}\n")
                continue
                
    except KeyboardInterrupt:
        print("\n\n用户中断，保存已生成数据...")
    
    # 保存数据
    if all_data:
        # 按时间戳命名文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"style_chat_data_{timestamp}.json"
        
        if save_data(all_data, filename):
            print_statistics(all_data)
            
            # 计算用时
            elapsed_time = time.time() - start_time
            print(f"\n总用时：{elapsed_time:.2f} 秒")
            print(f"平均每样本用时：{elapsed_time/len(all_data):.2f} 秒")
        
    else:
        print("警告：没有生成任何有效数据")

if __name__ == '__main__':
    main()