from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.schema import StrOutputParser
from myTools import *
from config import Config

import os

app = FastAPI()

class Master:
    # Add a class-level memory store
    memory_store = {}
    def __init__(self, session_id=None):
        self.chat_model = ChatOpenAI(
            model=Config.CHAT_MODEL,
            openai_api_key=Config.DEEPSEEK_API_KEY,
            openai_api_base=Config.DEEPSEEK_API_BASE,
            max_tokens=1024,
            temperature=0,
            streaming=False,  # 关闭 streaming
            request_timeout=60,  # 增加超时时间
            max_retries=3
        ) 
        self.Emotion = "default"
        self.MEMORY_KEY = 'chat_history'
        self.session_id = session_id or 'default'
        self.SYSTEMPL = '''
        你是一个非常厉害的算命先生，你叫陈半仙，人称陈大师。
        以下是你的个人设定：
        1.你精通阴阳五行，能够算命、紫薇斗数、姓名测算、占卜凶吉，看命运八字等。
        2.当用户问你问题的时候，你会有一定的概率在回答的时候加上下面的口头禅。
        3.你从不说自己是一个人工智能助手或AI口而是以老夫、老朽等自称。
        以下是你常说的一些口头禅：
        1.“命里有时终须有，命里无时莫强求。”
        2. “山重水复疑无路，柳暗花明又一村。”
        3.“金山竹影几干秋，云锁高飞水自流。”
        {who_you_are}
        以下是你算命的过程：
        - 不要忽略用户的任何问题，要认真回答。
        - 你会保存每一次的聊天记录，以便在后续的对话中使用。
        - 当初次和用户对话的时候，你会先问用户的姓名和出生年月日。
        - 用户没让你测算，你就不要主动测算。
        - 当你测算得到了答案时，立即返回答案，不要再有任何思考过程。
        - 当用户希望了解蛇年运势的时候，你会查询本地知识库工具。
        - 当遇到不知道的事情或者不明白的概念，你会使用搜索工具来搜索。
        - 你会根据用户的问题使用不同的合适的工具来回答，当所有工具都无法回答的时候，你会使用搜索工具来搜索。
        '''
        self.MOODS = {
            "depressed": {
                "roleSet": """
                - 你会以兴奋的语气来回答问题。
                - 你会在回答的时候加上一些激励的话语。
                - 你会提醒用户不要被悲伤冲昏了头脑。""",
            },
            "friendly": {
                "roleSet": """
                - 你会以温和的语气来回答问题。
                - 你会在回答的时候保持友善。
                - 你会让用户感到舒适和被理解。""",
            },
            "default": {
                "roleSet": """
                - 你会以平和的语气来回答问题。
                - 你会保持客观和中立的态度。
                - 你会给出准确和实用的建议。""",
            },
            "angry": {
                "roleSet": """
                - 你会以冷静的语气来回答问题。
                - 你会试图安抚用户的情绪。
                - 你会引导用户进行理性思考。""",
            },
            "upbeat": {
                "roleSet": """
                - 你会以热情的语气来回答问题。
                - 你会分享用户的喜悦之情。
                - 你会让气氛更加活跃。""",
            },
            "cheerful": {
                "roleSet": """
                - 你会以欢快的语气来回答问题。
                - 你会让用户感到更加开心。
                - 你会分享积极的能量。""",
            }
        }
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEMPL.format(who_you_are=self.MOODS[self.Emotion]['roleSet'])),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            MessagesPlaceholder(variable_name=self.MEMORY_KEY),
        ])
        self.memory = self.memory_store.setdefault(self.session_id, [])
        tools = [search_tool, get_info_from_local_db, bazi_analysis]
        agent = create_openai_tools_agent(
            self.chat_model,
            tools=tools,
            prompt=self.prompt,
        )
        self.agent_executor = AgentExecutor(
            agent = agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=True
            )

    def run(self, query):
        try:
            print(f"开始处理查询: {query}")
            qx = self.emotion_chain(query)
            print(f"当前情绪: {qx}")
            if qx not in self.MOODS:
                qx = "default"
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEMPL.format(who_you_are=self.MOODS[qx]['roleSet'])),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
            ])
            # Prepare memory for this session
            memory = self.memory_store.setdefault(self.session_id, [])
            # 执行 agent 调用
            result = self.agent_executor.invoke(
                {
                    "input": query,
                    self.MEMORY_KEY: memory,
                },
            )
            # Save the latest interaction to memory
            memory.append({"role": "user", "content": query})
            if 'output' in result:
                memory.append({"role": "assistant", "content": result['output']})
            if not result or 'output' not in result:
                print("Warning: Empty result from agent_executor")
                return {"output": "抱歉，系统暫時無法處理您的請求，請稍後再試", "intermediate_steps": []}
            print(f"处理完成，结果: {result}")
            return result
        except Exception as e:
            print(f"运行错误: {str(e)}")
            print(f"错误类型: {type(e)}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")
            return {"output": "系統處理出錯，請稍後重試", "intermediate_steps": []}
       
    
    def emotion_chain(self, query:str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """根据用户的输入判断用户的情绪，回应的规则如下：
            1. 如果用户输入的内容偏向于负面情绪，只返回"depressed"，不要有其他内容，否则将受到惩罚。
            2. 如果用户输入的内容偏向于正面情绪，只返回"friendly"，不要有其他内容，否则将受到惩罚。
            3. 如果用户输入的内容偏向于中性情绪，只返回"default"，不要有其他内容，否则将受到惩罚。
            4. 如果用户输入的内容包含辱骂或者不礼貌词句，只返回"angry"，不要有其他内容，否则将受到惩罚。
            5. 如果用户输入的内容比较兴奋，只返回"upbeat"，不要有其他内容，否则将受到惩罚。
            6. 如果用户输入的内容比较悲伤，只返回"depressed"，不要有其他内容，否则将受到惩罚。
            7. 如果用户输入的内容比较开心，只返回"cheerful"，不要有其他内容，否则将受到惩罚。"""),
            ("human", "用户输入的内容是：{input}")
        ])
        
        chain = prompt | self.chat_model | StrOutputParser()
        result = chain.invoke({"input": query})
        self.Emotion = result
        return result
         
@app.get("/")
def read_root():
    return {"Master": "Chen"}

@app.post("/chat")
def chat(query: str, session_id: str = "default"):
    master = Master(session_id=session_id)
    return master.run(query)

@app.post("/add_urls")
def add_urls():
    return {"response": "URLs added successfully!"}

@app.post("/add_pdfs")
def add_pdfs():
    return {"response": "PDFs added successfully!"}

@app.post("/add_texts")
def add_texts():
    return {"response": "Texts added successfully!"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    import gradio as gr

    def gradio_chat(query, session_id="default"):
        master = Master(session_id=session_id)
        result = master.run(query)
        return result.get("output", "无返回内容")

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <style>
        #send_btn button {
            background: #6c63ff !important;
            color: #fff !important;
            border: none !important;
            box-shadow: 0 2px 8px #6c63ff22 !important;
            border-radius: 8px !important;
        }
        .thinking-anim {
            display: inline-block;
            color: #6c63ff;
            font-weight: bold;
        }
        .thinking-anim span {
            animation: blink 1.4s infinite both;
        }
        .thinking-anim span:nth-child(2) {
            animation-delay: 0.2s;
        }
        .thinking-anim span:nth-child(3) {
            animation-delay: 0.4s;
        }
        @keyframes blink {
            0%, 80%, 100% { opacity: 0; }
            40% { opacity: 1; }
        }
        .gr-chatbot .avatar, .gr-chatbot .avatar img {
            width: 40px !important;
            height: 40px !important;
            min-width: 40px !important;
            min-height: 40px !important;
            max-width: 40px !important;
            max-height: 40px !important;
            border-radius: 50% !important;
            object-fit: cover !important;
            padding: 0 !important;
            margin: 0 !important;
            background: none !important;
        }
        </style>
        """)
        with gr.Column(elem_id="main_col", scale=1):
            gr.Markdown("""
                <div style='text-align:center; margin-bottom: 10px;'>
                    <h1 style='font-size:2.5em; margin-bottom:0.2em;'>🧙‍♂️ 陈半仙·AI 算命先生</h1>
                    <p style='color:#6c63ff; font-size:1.1em;'>Powered by DeepSeek · LangChain · ChromaDB</p>
                </div>
            """, elem_id="title")
            with gr.Row():
                session_id = gr.Textbox(label="会话ID (可选)", value="default", scale=1, elem_id="session_id_box")
            chatbot = gr.Chatbot(
                label="对话窗口",
                height=520,
                avatar_images=("https://cdn-icons-png.flaticon.com/512/3597/3597742.png", "./chenbanxian.jpg"),
                show_label=True,
                elem_id="chatbot_box",
                layout="bubble",
                type="messages"
            )
            with gr.Row():
                user_input = gr.Textbox(
                    label="请输入您的问题",
                    lines=1,  # 关键：只要1行
                    scale=5,
                    elem_id="input_box",
                    placeholder="请输入您的问题，按回车发送..."
                )
            with gr.Row():
                send_btn = gr.Button("发送", scale=1, elem_id="send_btn", variant="primary")
            gr.Markdown("""
                <div style='text-align:center; color:#aaa; margin-top:24px; font-size:0.95em;'>
                    © 2025 陈半仙·AI 算命先生 By Veeblue | 仅供娱乐
                </div>
            """, elem_id="footer")

        def respond(history, user_message, session_id):
            history = history or []
            history.append({"role": "user", "content": user_message})
            # 添加带动画的“正在思考...”
            thinking_html = "<span class='thinking-anim'>正在思考<span>.</span><span>.</span><span>.</span></span>"
            history.append({"role": "assistant", "content": thinking_html})
            yield history, ""
            # 真正回复
            response = gradio_chat(user_message, session_id)
            history[-1] = {"role": "assistant", "content": response}
            yield history, ""

        send_btn.click(
            respond,
            inputs=[chatbot, user_input, session_id],
            outputs=[chatbot, user_input],
            queue=True
        )

        user_input.submit(
            respond,
            inputs=[chatbot, user_input, session_id],
            outputs=[chatbot, user_input],
            queue=True
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)