import requests
from bs4 import BeautifulSoup
import re
import json
import os

# 目标网页
urls = [
    ("中华人民共和国劳动法", "https://www.mem.gov.cn/fw/flfgbz/201902/t20190213_231788.shtml"),
    ("中华人民共和国劳动合同法", "https://www.gjxfj.gov.cn/gjxfj/xxgk/fgwj/flfg/webinfo/2016/03/1460585589931971.htm")
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def extract_legal_clauses(law_name, url):
    response = requests.get(url, headers=headers)
    response.encoding = response.apparent_encoding
    soup = BeautifulSoup(response.text, "html.parser")

    legal_clauses = {}
    current_clause = None
    current_content = []

    # 直接从soup.stripped_strings中提取条款
    for text in soup.stripped_strings:
        # 跳过章节标题
        if re.match(r'^第[一二三四五六七八九十百零]+章', text):
            continue
        
        # 跳过页脚信息
        if any(keyword in text for keyword in ['版权所有', '技术支持', '京ICP备', '京公网安备', 'Produced By', '地址：', '电话：', '扫一扫', '责任编辑', '相关链接', '网站地图', '联系我们', '主办单位', '网站标识码']):
            continue
        
        # 检查是否是条款开头
        match = re.match(r'^第([一二三四五六七八九十百零]+)条', text)
        if match:
            # 保存上一条
            if current_clause and current_content:
                legal_clauses[current_clause] = ' '.join(current_content).strip()
            
            # 新条款
            clause_number = match.group(1)
            current_clause = f"{law_name} 第{clause_number}条"
            # 当前内容为本段去掉条款头部
            content = text[text.find('条')+1:].strip()
            current_content = [content] if content else []
        else:
            # 追加到当前条款内容
            if current_clause:
                current_content.append(text)

    # 保存最后一条
    if current_clause and current_content:
        legal_clauses[current_clause] = ' '.join(current_content).strip()
    
    return legal_clauses

# 确保data目录存在
os.makedirs('data', exist_ok=True)

# 转换为列表格式：每个法律作为一个列表项
all_results = []
for law_name, url in urls:
    print(f"正在抓取: {law_name} ...")
    clauses = extract_legal_clauses(law_name, url)
    
    # 将整个法律作为一个列表项
    all_results.append(clauses)
    
    print(f"{law_name} 共提取 {len(clauses)} 条法律条款。\n")

# 保存到data目录
with open('data/all_legal_clauses.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

print(f"全部抓取完成。共 {len(all_results)} 部法律，已保存到 data/all_legal_clauses.json 文件。")
