# llm-easy-questions
Test and visualise the evolution of LLMs ability to answer 'easy' questions.


Running:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=<your key>
python src/ask-easy-questions.py data/count-questions.csv gpt-4 gpt-5
```

> An OpenAI API Key is required, so there is some cost to run. You can provide any number of models as arguments to the code.

Visualise:

```
python src/visualise-results.py --outdir results/letter-counting/plots results/letter-counting/results-run-*
```