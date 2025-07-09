from langchain.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOpenAI
import requests
import os
import json
from config import Config

serpapi_api_key = os.getenv("SERPAPI_API_KEY")

@tool
def test():
    '''Test tool'''
    return "Hello"

@tool
def search_tool(query:str):
    '''当需要搜索的时候，使用这个工具''' 
    serpapi = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    result = serpapi.run(query)
    print("实时信息：", result)
    return result

@tool
def get_info_from_local_db(query:str):
    '''只有当回答本年运势的时候，才使用这个工具'''
    client = Chroma(
        collection_name="local_documents",
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory="./local_chroma_db"
    )
    retriever = client.as_retriever(search_type="mmr")
    result = retriever.get_relevant_documents(query)
    return result


@tool
def bazi_analysis(query: str, api_key: str = None):
    '''只有做八字排盘的时候才使用这个工具，需要姓名和出生年月日，缺一不可！'''
    url = "https://api.yuanfenju.com/index.php/v1/Bazi/cesuan"
    
    # 获取API密钥
    if not api_key:
        api_key = os.getenv("YUANFENJU_API_KEY")
    
    try:
        print(f"开始八字分析，查询内容: {query}")
        
        # 修改提示模板，要求返回JSON格式（转义大括号）
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            你是一个参数提取助手，根据用户输入提取八字分析所需的参数。
            请严格按照以下JSON格式返回，不要添加任何其他内容：
            
            {{{{
                "api_key": "{api_key}",
                "name": "提取的姓名",
                "sex": "1",
                "type": "1", 
                "year": "出生年份",
                "month": "出生月份",
                "day": "出生日期",
                "hours": "出生小时",
                "minute": "0",
            }}}}
            
            参数说明：
            - sex: 1=男，0=女
            - type: 固定为"1"
            - zhen: 1=使用真太阳时，0=不使用
            - 时间格式都是字符串
            - 如果用户没有提供某些信息，请合理推测或使用默认值
            
            用户输入：{{input}}
            """),
            ("human", "{input}")
        ])
        
        # 创建LLM链
        llm = ChatOpenAI(
            model=Config.CHAT_MODEL,
            openai_api_key=Config.DEEPSEEK_API_KEY,
            openai_api_base=Config.DEEPSEEK_API_BASE,
            temperature=0,
        )
        
        chain = prompt | llm
        
        # 调用链获取参数
        response = chain.invoke({"input": query})
        print(f"LLM响应: {response.content}")
        
        # 解析JSON响应
        try:
            # 清理响应内容，提取JSON部分
            content = response.content.strip()
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            
            data = json.loads(content)
            print(f"解析后的参数: {data}")
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始响应: {response.content}")
            return "参数解析失败，请确保提供完整的姓名和出生年月日时分信息"
        
        # 验证必需参数
        required_fields = ['name', 'year', 'month', 'day', 'hours', 'minute']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            return f"缺少必需参数: {', '.join(missing_fields)}"
        
        # 发送API请求
        print("发送API请求...")
        print(f"请求数据: {data}")
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        result = requests.post(url, data=data, headers=headers, timeout=30)
        print(f"API响应状态码: {result.status_code}")
        print(f"API响应内容: {result.text}")
        
        if result.status_code == 200:
            try:
                json_result = result.json()
                print(f"解析后的JSON结果: {json_result}")
                
                # 检查API返回的错误
                if 'code' in json_result and json_result['code'] != 200:
                    error_msg = json_result.get('msg', '未知错误')
                    return f"API调用失败: {error_msg}"
                
                # 检查数据结构
                if 'data' in json_result and 'bazi_info' in json_result['data']:
                    bazi_info = json_result['data']['bazi_info']
                    if 'bazi' in bazi_info:
                        return f"八字为: {bazi_info['bazi']}"
                    else:
                        return f"八字分析完成，但未找到八字信息。返回数据: {json_result}"
                else:
                    return f"API返回数据格式异常: {json_result}"
                    
            except json.JSONDecodeError as e:
                print(f"API响应JSON解析错误: {e}")
                return f"API响应解析失败: {result.text}"
        else:
            return f"API请求失败，状态码: {result.status_code}，响应: {result.text}"
    
    except Exception as e:
        print(f"函数执行异常: {e}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        return f"系统处理出错: {str(e)}"