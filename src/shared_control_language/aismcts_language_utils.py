import numpy as np
import random
from copy import deepcopy
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load API key from .env file and initialize OpenAI client with it
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

"""
============================
game constants
1. direction_map: mapping of action to direction
2. flag_dict: mapping of flag to action
3. opposite_direction: mapping of action to opposite action
4. offsets: offsets for calculating mirror positions
============================
"""

direction_map = {
    0: "stay put",
    1: "move right",
    2: "move up",
    3: "move left",
    4: "move down",
}
flag_dict = {
    "STAY_PUT": 0,
    "RIGHT": 1,
    "UP": 2,
    "LEFT": 3,
    "DOWN": 4,
    "INQUIRY": "INQUIRY",
    "ACCEPT": "ACCEPT",
    "REJECT": "REJECT",
}
opposite_direction = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2}
offsets = {
    1: (1, 0),
    2: (0, -1),
    3: (-1, 0),
    4: (0, 1),
}  # offsets for calculating mirror positions


""" 
============================
modified ISMCTS functions and classes 
1. is_equivalent_node: check if two nodes are equivalent
2. Node: class for MCTS node
3. Tree: class for MCTS tree
============================
"""


def is_equivalent_node(node_a, node_b):
    return all(
        [
            node_a.player == node_b.player,
            np.array_equal(
                node_a.observation, node_b.observation
            ),  # important for stochastic game transitions
            node_a.action == node_b.action,
        ]
    )


class Node:
    def __init__(self, game, player, action=None, parent=None, c=np.sqrt(2)):
        self.parent = parent
        self.children = []

        # player information set
        self.player = player
        self.observation = game.unwrapped.get_player_observation(
            self.player
        )  # current player, token position, walls, and treasure info

        self.action = action  # the action that transitions the parent node to this node
        self.untried_actions = None

        self.c = c
        self.T = 0  # total rewards from MCTS exploration
        self.N = 0  # visit count

    @property
    def ucb(self):
        # Unexplored nodes have maximum values so we favour exploration
        if self.N == 0:
            return np.inf

        # parent node of the current node
        top_node = self
        if top_node.parent:
            top_node = top_node.parent

        # UCB formula
        return self.T / self.N + self.c * np.sqrt(np.log(top_node.N) / self.N)

    def backpropagate(self, current_T):
        self.N += 1
        self.T += current_T
        if self.parent:
            self.parent.backpropagate(self.T)


class Tree:
    def __init__(self, game, player):
        self.game = game
        self.player = player

        self.root_node = Node(deepcopy(game), player)
        self.current_node = self.root_node

        # during selection (self.is_on_rollout_policy=False), we select actions based on UCB and keep track of new nodes.
        # during rollout (self.is_on_rollout_policy=True), we select actions randomly and do NOT keep track of new nodes
        self.is_on_rollout_policy = False

    def select_best_action(self, flag=None, chat="", side_info={}, last_flag=None):
        # side_info:= knowledge of the other side = {token position: set of disallowed actions}

        messages = []

        print("token at", self.current_node.observation[1:3])

        # find the set of child node indices with the highest visit count
        N_scores = [child.N for child in self.current_node.children]
        node_indices = [
            index for index, value in enumerate(N_scores) if value == max(N_scores)
        ]
        action_to_nodeidx = {
            self.current_node.children[i].action: i for i in node_indices
        }

        if flag is None or flag == "ACCEPT":
            action, nodeidx = random.choice(list(action_to_nodeidx.items()))
        elif flag == "INQUIRY":
            action, nodeidx = random.choice(list(action_to_nodeidx.items()))
            messages.append(
                respond_inquiry(chat, self.game.unwrapped.get_infos(), action)
            )
        elif flag == "REJECT":
            if last_flag is not None and last_flag != 0:
                # update knowledge on the other side, else ignore rejection b/c noop is always valid
                current_token_pos = tuple(self.game.unwrapped.get_infos()["token"])
                if current_token_pos not in side_info.keys():
                    side_info[current_token_pos] = set([last_flag])
                else:
                    side_info[current_token_pos].add(last_flag)

                # mirror the disallowed action and add to side_info
                offset = offsets[last_flag]
                mirror_token_pos = (
                    current_token_pos[0] + offset[0],
                    current_token_pos[1] + offset[1],
                )
                mirror_flag = opposite_direction[last_flag]

                # update side_info with mirror_token_pos and mirror_flag
                if mirror_token_pos not in side_info.keys():
                    side_info[mirror_token_pos] = set([mirror_flag])
                else:
                    side_info[mirror_token_pos].add(mirror_flag)

            # regardless of whether side_info is updated, we select a random action
            action, nodeidx = random.choice(list(action_to_nodeidx.items()))
        elif flag in action_to_nodeidx.keys():
            # adopt flag
            print("flag adopted: ", direction_map[flag])
            action, nodeidx = flag, action_to_nodeidx[flag]
        else:
            # reject flag and explain
            print("flag not adopted: ", flag)

            if flag not in self.game.unwrapped.get_valid_actions(self.player):
                # not a valid action
                messages.append(
                    "I cannot {} because there is a wall in that direction.".format(
                        direction_map[flag]
                    )
                )
            else:
                # not the best action
                messages.append(
                    "Moving {} is not the best action to take in my computation.".format(
                        direction_map[flag]
                    )
                )

            action, nodeidx = random.choice(list(action_to_nodeidx.items()))

        """ flag output based on N scores of the grandchild nodes """
        child = self.current_node.children[nodeidx]
        print("child node token at", tuple(child.observation[1:3]))
        if tuple(child.observation[1:3]) in side_info.keys():
            # prune grandchildren based on side_info
            print("pruning grandchildren ...")
            grandchildren = [
                g
                for g in child.children
                if g.action not in side_info[tuple(child.observation[1:3])]
            ]
            # remove the child node from the side_info dictionary??
        else:
            # keep all grandchildren
            grandchildren = child.children

        # print (pruned) grandchild N scores
        if len(grandchildren) == 0:
            flag_output = None
        elif len(grandchildren) == 1:
            flag_output = grandchildren[0].action
        else:
            if opposite_direction[action] in [g.action for g in grandchildren]:
                action_to_N = {
                    g.action: g.N
                    for g in grandchildren
                    if g.action != opposite_direction[action]
                }
                maxN_actions = [
                    a for a, N in action_to_N.items() if N == max(action_to_N.values())
                ]
                flag_output = random.choice(maxN_actions)
            else:
                flag_output = grandchildren[
                    np.argmax([g.N for g in grandchildren])
                ].action

        # return action, flag, and messages
        return action, flag_output, side_info, messages

    def select_action(self, side_info):
        # compute current token position
        current_token_pos = tuple(self.game.unwrapped.get_infos()["token"])

        if (
            self.is_on_rollout_policy
        ):  # rollout: with valid_action with side_info in mind for human partner
            if self.player == 0:  # human player
                if current_token_pos in side_info.keys():
                    # print("rollout with pruned human actions")
                    return random.choice(
                        [
                            action
                            for action in self.game.unwrapped.get_inbound_actions()
                            if action not in side_info[current_token_pos]
                        ]
                    )
                else:
                    return random.choice(self.game.unwrapped.get_inbound_actions())
            elif self.player == 1:  # agent player
                return random.choice(self.game.unwrapped.get_valid_actions(self.player))
        else:  # try each action at least once, then select the action with the highest UCB
            if (
                self.current_node.untried_actions is None
                and len(self.current_node.children) == 0
            ):
                if self.player == 0:  # human player
                    if current_token_pos in side_info.keys():
                        self.current_node.untried_actions = [
                            action
                            for action in self.game.unwrapped.get_inbound_actions()
                            if action not in side_info[current_token_pos]
                        ]
                        # print("try each pruned human action: ", self.current_node.untried_actions)
                    else:
                        self.current_node.untried_actions = (
                            self.game.unwrapped.get_inbound_actions()
                        )
                elif self.player == 1:  # agent player
                    self.current_node.untried_actions = (
                        self.game.unwrapped.get_valid_actions(self.player)
                    )
                random.shuffle(self.current_node.untried_actions)
            if (
                self.current_node.untried_actions is not None
                and len(self.current_node.untried_actions) > 0
            ):
                return self.current_node.untried_actions.pop()
            return self.current_node.children[
                np.argmax([child.ucb for child in self.current_node.children])
            ].action

    def find_or_create_child(self, action):
        # if we are not on rollout policy, we need to keep track of new nodes
        if not self.is_on_rollout_policy:
            # find the node that corresponds to the action and set it as the current node
            new_node = Node(
                deepcopy(self.game),
                player=self.player,
                action=action,
                parent=self.current_node,
            )
            # for existing_node in [self.current_node, *self.current_node.children]:
            for existing_node in self.current_node.children:
                if is_equivalent_node(new_node, existing_node):
                    # print("existing node found: ", existing_node, existing_node.player, existing_node.action)
                    self.current_node = existing_node
                    return
            # else we create a new node
            # print("new node created: ", new_node, new_node.player, new_node.action, new_node.parent)
            self.current_node.children.append(new_node)
            self.current_node = new_node
            self.is_on_rollout_policy = True
            # print("starting rollout ...")


""" 
============================
planning module via AISMCTS-F algorithm
============================
"""


def aismcts_flag(
    root_state, flag, chat, side_info, last_flag, MCTS_POLICY_EXPLORE=100, player=None
):
    # Instantiate trees and reset_state
    reset_state = deepcopy(root_state)
    game = deepcopy(root_state)
    trees = [Tree(game, player) for player in range(game.unwrapped.n_agents)]

    for i in range(MCTS_POLICY_EXPLORE):
        # reset statistics
        done = False
        r = 0

        while not done:
            # EXPLORE
            action = trees[game.unwrapped.current_player].select_action(side_info)

            # game step: accumulates reward, transitions to next state, and switches player
            obs_n, reward, terminated, truncated, info = game.step(action)
            done = terminated or truncated
            r += reward

            # EXPAND if not found in the tree
            for tree in trees:
                tree.find_or_create_child(action)

        # BACKPROPAGATE
        for tree in trees:
            tree.current_node.backpropagate(r)

        # reset game, trees
        game = deepcopy(reset_state)
        for tree in trees:
            tree.game = game
            tree.current_node = tree.root_node
            tree.is_on_rollout_policy = False

    # return SelectBestAction -> action, flag, messages
    this_player = game.unwrapped.current_player if player is None else player
    return trees[this_player].select_best_action(
        flag=flag, chat=chat, side_info=side_info, last_flag=last_flag
    )


""" 
============================
language modules via GPT-3.5
============================
"""


def chat_to_flag(chat):
    # Check if chat is empty and return None immediately
    if chat.strip() == "":
        return None

    prompt = """
    Based on the conversation below, identify the key intention or action. Output MUST be strictly one of the following: [STAY_PUT, RIGHT, UP, LEFT, DOWN, INQUIRY, ACCEPT, REJECT]. 
    "STAY_PUT" should only be used if the chat explicitly asks to not move (e.g., "Can you stay put?"). 
    For responses indicating agreement or confirmation (e.g., "ok", "sure", "I did"), use "ACCEPT".
    For responses asking for information (e.g., "Where exactly is the hidden treasure located?"), use "INQUIRY".

    Conversation examples and expected flags:

    Conversation:
    "Right and then down. First move should be right."

    Flag:
    RIGHT
    ---
    Conversation:
    "Can you move left by one step?"

    Flag:
    LEFT
    ---
    Conversation:
    "Where exactly is the hidden treasure located?"

    Flag:
    INQUIRY
    ---
    Conversation:
    "Can you stay put?"

    Flag:
    STAY_PUT
    ---
    Conversation:
    "Ok."

    Flag:
    ACCEPT
    ---
    Conversation:
    "Sure."

    Flag:
    ACCEPT
    ---
    Conversation:
    "I did."

    Flag:
    ACCEPT
    ---
    Conversation:
    "I cannot, there is a wall in that direction."

    Flag:
    REJECT
    ---
    Conversation:
    {}

    Flag:

    """.format(chat)

    # Usage of the OpenAI API
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Analyze the conversation and predict the next desired move or question. The response must strictly be one of: [STAY_PUT, RIGHT, UP, LEFT, DOWN, INQUIRY, ACCEPT, REJECT]. Do not provide explanations or multiple options.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    output = completion.choices[0].message.content.split("\n")[-1]
    print("GPT output: ", output)

    # Find and store all matching actions
    matching_flags = [flag for flag in flag_dict.keys() if flag in output]
    if matching_flags:
        flag_key = random.choice(matching_flags)
        if len(matching_flags) > 1:
            print("I am taking action: ", flag_key)  # message
        flag_value = flag_dict[flag_key]
    else:
        flag_value = None

    return flag_value


def flag_to_chat(flag):
    if flag is not None:
        return ["Can you {}?".format(direction_map[flag])]


def respond_inquiry(chat, info, action):
    treasure_msg = (
        f"- Treasure position: {info['treasure']['pos'].tolist()}"
        if info["treasure"]["onWhichSide"] == 1
        else "- You cannot see the treasure position."
    )
    # Construct the prompt
    prompt = f"""You are playing a maze game where the goal is to reach the treasure. The maze's coordinate system starts at [0,0] in the top-left corner, with the x-axis increasing to the right and the y-axis increasing downwards. Current game state:
    - At token position: {info['token'].tolist()}, I take the move {action} now.
    - {treasure_msg}

    Respond to the inquiry: "{chat}"
    """

    # Usage of the OpenAI API
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Assist in a maze game called 'Gnomes at Night'. Use the provided coordinates to focus on the general direction instead of specific next steps. Remind the player of possible obstacles in the way. Limit the response to within 30 words.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    output = completion.choices[0].message.content
    # print("GPT output: ", output)

    # respond to chat with info given
    return output
