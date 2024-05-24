# Human-Agent Cooperation in Games under Incomplete Information through Natural Language Communication
- Shenghui Chen, Daniel Fried, Ufuk Topcu
- International Joint Conference on Artificial Intelligence (IJCAI), Human-Centred Artificial Intelligence track, 2024 

<div align="center" style="font-size: 24px; font-weight: bold;">
<!--   <a href="link-to-ijcai-paper">🔗 Paper</a> &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; -->
  <a href="https://arxiv.org/abs/2405.14173">📑 Paper (main+appendix) on Arxiv</a> &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
  <a href="https://shenghui-chen.com/publication/2024/03/22/shared_control_game/">📝 Blog</a>
</div>

## 📋 Abstract
🌟 Developing autonomous agents to strategize and cooperate with humans using natural language is challenging. Our testbed, Gnomes at Night 🎮, is a maze game where two players alternately control a token to achieve a common goal with incomplete information.

✍️ We introduce a *shared-control game*, where two players collectively control a *token* ♟️ in alternating turns to achieve a common objective under incomplete information. 
We formulate a policy synthesis problem for an autonomous agent in this game with a human as the other player.

🤖 To solve this problem, we propose a *communication-based approach* comprising a language module and a planning module. The language module translates natural language messages into and from a finite set of *flags*, a compact representation defined to capture player intents. The planning module leverages these flags to compute a policy using an *asymmetric information-set Monte Carlo tree search with flag exchange* (AISMCTS-F) algorithm we present. 

<div align="center" style="display: flex; justify-content: center; align-items: center; gap: 100px; margin-bottom: 20px;">
  <img src="images/gnomes_at_night.jpg" alt="Gnomes at Night" width="350" padding-right: 100px/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="images/cooperative_control_game.jpg" alt="Cooperative Control Game" width="350"/>
</div>

## 📚 Citation
If you use this work, please cite:

```bibtex
@inproceedings{chen2024sharedcontrol,
  author={Chen, Shenghui and Fried, Daniel and Topcu, Ufuk},
  booktitle={International Joint Conference on Artificial Intelligence (IJCAI), Human-Centred Artificial Intelligence track}, 
  title={Human-Agent Cooperation in Games under Incomplete Information through Natural Language Communication}, 
  year={2024},
}
```




## 📦 Installation
It is recommended to create a Python virtual environment first:

```bash
python -m venv venv
source venv/bin/activate
```

Then install the dependencies:

```bash
pip install -e .
```

To use the OpenAI API, you need to add your API key 🔑 in a `.env` file located in the root directory. Create the .env file with the following content:
```txt
OPENAI_API_KEY=[your openai api key]
```
This API key will be used in the `src/shared_control_language/aismcts_language_utils.py` file. Make sure to replace `[your openai api key]` with your actual OpenAI API key.


## 🚀 Usage
To test this algorithm in a 9x9 Gnomes at Night environment called `GnomesAtNightEnv9A` by interacting with an agent in the terminal, run the following command:

```bash
python scripts/test_aismcts_language.py --round 1 --explore 100 --render
```

The command line arguments include
- `--round`: Specifies the round number (in [1, 2, 3, 4, 5], default is 1).
- `--explore`: Sets the number of iterations (exploration constant) in MCTS (must be an integer, default is 100).
- `--render`: Enables rendering of the environment (use this flag only when rendering is desired).


At the end of gameplay, you should see logging output 📝 similar to the following:

```terminal
===============================================
GnomesAtNightEnv9A Play Summary
-----------------------------------------------
Round         1
MCTS Explore  100
Render        False
-----------------------------------------------
Total Reward  17.0
Total Steps   6
Side Info     {}
===============================================
```
