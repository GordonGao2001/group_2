from llama_cpp import Llama
import os

###############################
# 1 print models' existence
# 2 check & print input file status
# 3 check output function() ******
# 4 output
###############################

q_list = []

def file_reader(f_path):
    file = open(input_file, 'r')
    while True:
        line = file.readline()
        if not line:
            break
        try:
            if '\t' in line:
                q_list.append(line.split('\t')[1])
            else:
                raise ValueError('no <tab> found in current line!')
        except ValueError:
            print("wrong file format!!!")
            exit()
    print('input file ends')
    file.close()

# 1
model_loading_mode = input('model path is: default(input <1>) or  manual(enter full path)\n')
if model_loading_mode == '1':
    model_path = os.getcwd() + '/models/llama-2-7b-chat.Q4_K_M.gguf' # need to change this path in final version
else:
    if os.path.exists(model_loading_mode):
        print('model found, continue....(this can take a while)\n')
        model_path = model_loading_mode
    else:
        print('model path invalid, exiting')
        exit()

# 2
input_file = os.getcwd()+'/input.txt'
if os.path.exists(input_file):
    file_reader(input_file)
else:
    print(f'input file not found\n should be at {input_file}\n, exiting')
    exit()

# question = "What is the capital of Italy? "
# llm = Llama(model_path=model_path, verbose=False)
# print("Asking the question \"%s\" to %s (wait, it can take some time...)" % (question, model_path))
# output = llm(
#       question, # Prompt
#       max_tokens=32, # Generate up to 32 tokens
#       stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
#       echo=True # Echo the prompt back in the output
# )
# print("Here is the output")
# print(output['choices'])
