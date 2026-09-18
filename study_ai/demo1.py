"""Gradio 入门 Demo

运行方式:
    python demo1.py
然后在浏览器打开 http://127.0.0.1:7860
"""

import gradio as gr


def greet(name, excitement):
    """最简单的处理函数: 接收输入, 返回输出"""
    if not name.strip():
        return "请输入名字~"
    suffix = "!" * int(excitement)
    return f"你好, {name}{suffix}"


with gr.Blocks(title="Gradio Demo1") as demo:
    gr.Markdown("# Gradio 第一个 Demo\n输入名字, 点 Submit 看结果.")
    with gr.Row():
        with gr.Column():
            name_input = gr.Textbox(label="你的名字", placeholder="例如: 小明")
            excitement_slider = gr.Slider(minimum=0, maximum=5, step=1, value=1, label="感叹号数量")
            submit_btn = gr.Button("打招呼", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="结果")

    # 绑定事件: 点击按钮 -> 调用 greet -> 输出到 output
    submit_btn.click(fn=greet, inputs=[name_input, excitement_slider], outputs=output)


if __name__ == "__main__":
    demo.launch()
