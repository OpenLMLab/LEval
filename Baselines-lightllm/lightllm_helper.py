import subprocess
import time
import requests
import json

lightllm_headers = {'Content-Type': 'application/json'}


def lightllm_infer(input, do_sample=False, max_new_tokens=1024, ignore_eos=False, port=8000):
    url = f'http://localhost:{port}/generate'
    data = {
        'inputs': input,
        "parameters": {
            'do_sample': do_sample,
            'ignore_eos': ignore_eos,
            'max_new_tokens': max_new_tokens,
        }
    }
    response = requests.post(url, headers=lightllm_headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()['generated_text'][0]
    else:
        raise Exception('LightLLM Error:', response.status_code, response.text)
    

class LightLLMServer():
    def __init__(self, model_path, extra_args, port=8000):
        self.model_path = model_path
        self.extra_args = extra_args
        self._lightllm_proc = None
        self.port = port

    def __enter__(self):
        self._lightllm_proc = subprocess.Popen(["python", "-m", "lightllm.server.api_server", "--model_dir", self.model_path, "--port", str(self.port), *self.extra_args])
        # Wait for the server to start
        while True:
            try:
                url = f'http://localhost:{self.port}/generate'
                data = {
                    'inputs': "What is AI?", # Test generation
                    "parameters": {
                        'do_sample': False,
                        'ignore_eos': False,
                        'max_new_tokens': 5,
                    }
                }
                requests.post(url, headers=lightllm_headers, data=json.dumps(data))
                print("LightLLM Server Started")
                break
            except requests.exceptions.ConnectionError:
                print("Waiting for LightLLM to start...")
                time.sleep(5)
                continue
            except Exception as e:
                raise Exception("LightLLM Failed to start:", e)
        return self._lightllm_proc


    def __exit__(self, exc_type,exc_value, exc_traceback):
        if self._lightllm_proc:
            self._lightllm_proc.terminate()