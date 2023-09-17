from flask import Flask, request, jsonify

from typing import Optional

import fire

from llama import Llama

max_new_tokens = 512

app = Flask(__name__)

deploy_model = 'llama-2-7b-chat'

generator = Llama.build(ckpt_dir=deploy_model +'/',
                        tokenizer_path='tokenizer.model',
                        max_seq_len=3584,
                        max_batch_size=1,
                    )

@app.route('/<path:method>', methods=['POST'])
def handle_api_call(method):
    print(f"method={method}")
    if method == "ChatCompletion.create" or method == "chat/completions":
        data = request.get_json()
        messages_oai = data.get("messages")
        model = data.get("model")
        temperature = 0.1 if "temperature" not in data.keys() else data.get("temperature")
        top_p = 1 if "top_p" not in data.keys() else data.get("top_p")

        if model == 'llama-2-7b-chat':
            resp = generator.chat_completion(
                [messages_oai],
                max_gen_len=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            print(resp[0]['generation'])
            choices = [{
                "index": 0,
                "message": {
                    "role": resp[0]['generation']['role'],
                    "content": resp[0]['generation']['content'],
                },
            }]
            
            usage = {}
        
        return jsonify({'choices': choices, 'usage': usage})
    else:
        return jsonify({'error': 'Invalid inputs.'}), 400

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
