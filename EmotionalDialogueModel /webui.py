import gradio as gr
import requests
import json

# VLLM服务器配置
VLLM_URL = "http://localhost:8000/v1/chat/completions"  # 修改为你的VLLM服务地址

def chat_with_vllm(message, history, temperature=0.7, max_tokens=512):
    """
    与VLLM服务进行对话
    """
    try:
        # 构建消息历史
        messages = []
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        
        # 添加当前消息
        messages.append({"role": "user", "content": message})
        
        # 发送请求到VLLM
        payload = {
            "model": "/root/autodl-tmp/models/Qwen3-1___7B_checkpoint-500",  # 替换为你的模型名称
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        response = requests.post(VLLM_URL, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            reply = result['choices'][0]['message']['content']
            return reply
        else:
            return f"错误: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"连接错误: {str(e)}"

def clear_chat():
    """清空对话历史"""
    return [], ""

# 创建Gradio界面
with gr.Blocks(title="VLLM 对话测试", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# VLLM 对话测试界面")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="对话历史",
                height=500,
                show_copy_button=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="输入消息",
                    placeholder="在这里输入你的问题...",
                    lines=2,
                    max_lines=5
                )
                
            with gr.Row():
                send_btn = gr.Button("发送", variant="primary")
                clear_btn = gr.Button("清空对话")
        
        with gr.Column(scale=1):
            gr.Markdown("### 参数设置")
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature"
            )
            max_tokens = gr.Slider(
                minimum=50,
                maximum=2048,
                value=512,
                step=50,
                label="Max Tokens"
            )
            
            gr.Markdown("### 使用说明")
            gr.Markdown("""
            1. 确保VLLM服务正在运行
            2. 修改代码中的VLLM_URL和模型名称
            3. 调整Temperature和Max Tokens参数
            4. 在输入框中输入问题并点击发送
            """)
    
    # 事件绑定
    def respond(message, history, temp, max_tok):
        if not message.strip():
            return history, ""
        
        # 获取AI回复
        bot_message = chat_with_vllm(message, history, temp, max_tok)
        
        # 更新历史记录
        history.append((message, bot_message))
        return history, ""
    
    # 绑定事件
    send_btn.click(
        respond,
        inputs=[msg, chatbot, temperature, max_tokens],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        respond,
        inputs=[msg, chatbot, temperature, max_tokens],
        outputs=[chatbot, msg]
    )
    
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, msg]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )