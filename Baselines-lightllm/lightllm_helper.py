import subprocess
import time
import requests
import json

lightllm_headers = {'Content-Type': 'application/json'}

def start_lightllm_server(model_path, tp, max_input_len, max_len, port=8000):
    """
    Spawn a subprocess to start the lightllm server
    """
    lightllm_proc = subprocess.Popen(["python", "-m", "lightllm.server.api_server", "--model_dir", model_path, "--port", str(port), "--trust_remote_code", "--tp", str(tp), "--max_req_input_len", str(max_input_len-1000), "--max_req_total_len", str(max_len)])
    # Wait for the server to start
    while True:
        try:
            url = f'http://localhost:{port}/generate'
            data = {
                'inputs': "What is AI?", # Test generation
                "parameters": {
                    'do_sample': False,
                    'ignore_eos': False,
                    'max_new_tokens': 5,
                }
            }
            response = requests.post(url, headers=lightllm_headers, data=json.dumps(data))
            print("LightLLM Server Started")
            break
        except requests.exceptions.ConnectionError:
            print("Waiting for LightLLM to start...")
            time.sleep(5)
            continue
        except Exception as e:
            stop_lightllm_server(lightllm_proc) 
            raise Exception("LightLLM Failed to start:", e)
    return lightllm_proc

def stop_lightllm_server(lightllm_proc):
    lightllm_proc.send_signal(1) # FIXME: LightLLM not properly stopped

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