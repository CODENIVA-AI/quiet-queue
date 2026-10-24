"""Optional local web demo. Run: python web_demo.py; open the printed localhost URL."""
import json
from pathlib import Path
import gradio as gr
from quietqueue.core import triage, to_markdown

SEED = (Path(__file__).parent/'examples'/'sample_inbox.json').read_text(encoding='utf-8')


def process(raw: str, today: str):
    try:
        items = json.loads(raw)
        result = triage(items, today=today)
        return to_markdown(result), json.dumps(result, ensure_ascii=False, indent=2)
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        return f'Invalid input: {e}', '{}'

with gr.Blocks(title='CODENIVA AI — Quiet Queue') as demo:
    gr.Markdown('# CODENIVA AI — Quiet Queue 🦥\n**Slow & Steady:** let local open-source machine learning prepare your brief. Data stays on this computer.')
    with gr.Row():
        source = gr.Code(label='Paste a list of task objects (JSON)', value=SEED, language='json', lines=18)
        with gr.Column():
            today = gr.Textbox(value='2026-10-24', label='Reference date YYYY-MM-DD')
            go = gr.Button('Build my quiet queue', variant='primary')
            gr.Markdown('**Safety:** no account access, no sending, no deletion. Review every recommendation.')
    out = gr.Markdown(label='Daily brief')
    structured = gr.Code(label='Structured JSON', language='json')
    go.click(process, inputs=[source, today], outputs=[out, structured])

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', share=False, inbrowser=False)
