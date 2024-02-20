"""
Usage:
python3 openai_speculative.py
"""
from sglang import function, gen, set_default_backend, OpenAI
import sglang as sgl

class Rail:
    def __init__(self, system_prompt):
        """Structured generation for state control and optimization."""
        self.system_prompt = system_prompt
        self.logs = []
    
    def log(self, m):
        self.logs.append(m)

    def instruct(self, instruction, variables):
        prompt = self.system_prompt + "\n"
        prompt += "Log history:\n"
        for log in self.logs:
            prompt += "{0}\n".format(log)
        prompt += instruction + "\n"
        
        @function
        def gen_instruct(s):
            s += prompt
            for name in variables:
                s += "I will update {:s} from {:.4f} to ".format(name, variables[name]) + gen(name, stop=" ") + " because " + gen("{0}_reasoning".format(name), stop="\n\n")
        state = gen_instruct.run()
        return state


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()  # take environment variables from .env.

    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    set_default_backend(OpenAI("gpt-3.5-turbo-instruct"))


    hyperparameters = {
        "learning_rate": 0.0001,
        "batch_size": 64
    }

    r = Rail("Act as a 10x engineer, specifically an ML engineer. You are in charge of tuning hyperparameters for a model training on CIFAR10.")
    r.log("Training run result:\ntest accuracy:0.94\n\nConfig:\n{0}".format(str(hyperparameters)))
    update = r.instruct("Suggest new hyperparameters and explain your reasoning.", hyperparameters)
    print(update)
    for h in hyperparameters:
        print(h, update[h])
    