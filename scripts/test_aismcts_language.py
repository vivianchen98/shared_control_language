from shared_control_language.gnomesAtNightEnv_9A import GnomesAtNightEnv9A
from shared_control_language.aismcts_language_utils import (
    aismcts_flag,
    chat_to_flag,
    flag_to_chat,
)
from copy import deepcopy
import argparse


def print_summary(metadata, statistics):
    # both metadata and statistics are dictionaries
    # left_column_width = max of all the lengths of the labels in metadata and statistics
    left_column_width = max(
        max(map(len, metadata.keys())),
        max(map(len, statistics.keys())),
    )

    # print the summary with proper alignment and line in between metadata and statistics
    summary = (
        "===============================================\n"
        "GnomesAtNightEnv9A Play Summary\n"
        "-----------------------------------------------\n"
        + "\n".join(
            [
                f"{label.ljust(left_column_width)}  {value}"
                for label, value in metadata.items()
            ]
        )
        + "\n"
        "-----------------------------------------------\n"
        + "\n".join(
            [
                f"{label.ljust(left_column_width)}  {value}"
                for label, value in statistics.items()
            ]
        )
        + "\n"
        "===============================================\n"
    )
    print(summary)


def agent(env, chat, side_info, last_flag, MCTS_POLICY_EXPLORE=100):
    flag = chat_to_flag(chat)

    action, agent_flag, updated_side_info, msg = aismcts_flag(
        deepcopy(env),
        flag,
        chat,
        side_info,
        last_flag,
        MCTS_POLICY_EXPLORE=MCTS_POLICY_EXPLORE,
        player=1,
    )
    print("current side_info: ", updated_side_info)

    agent_msg = msg + flag_to_chat(agent_flag) if agent_flag is not None else msg

    return action, agent_msg, updated_side_info, agent_flag


def human(agent_msg):
    print("Agent messages: ", agent_msg)
    action = int(input("input action: "))
    chat = input("input chat: ")
    return action, chat


def main(ROUND, EXPLORE, RENDER):
    side_info = {}
    last_flag = None

    # reset statistics
    total_reward = 0
    steps = 0
    done = False

    # create environment and reset
    env = GnomesAtNightEnv9A(render_mode="rgb_array", round=ROUND)
    obs_n, info = env.reset()
    if RENDER is True:
        visualize_env = GnomesAtNightEnv9A(render_mode="human", round=ROUND)
        visualize_env.reset()

    # initialize flags and turn indicator
    one_chat = []

    # initialize turn indicator
    zero_turn = True if info["current_player"] == 0 else False

    while not done:
        print("--- token pos: {} ---".format(obs_n[0][1:3]))
        if zero_turn:
            action, zero_chat = human(one_chat)
            print()
        else:
            action, one_chat, side_info, last_flag = agent(
                deepcopy(env),
                zero_chat,
                side_info,
                last_flag,
                MCTS_POLICY_EXPLORE=EXPLORE,
            )
            print()

        # take step
        obs_n, reward, terminated, truncated, info = env.step(action)
        if RENDER is True:
            visualize_env.step(action)

        # update statistics
        total_reward += reward
        steps += 1
        done = terminated or truncated

        # render if requested
        if RENDER is True:
            visualize_env.render()

        # switch turns
        zero_turn = not zero_turn

    env.close()

    # print the summary with proper alignment
    print_summary(
        {
            "Round": ROUND,
            "MCTS Explore": EXPLORE,
            "Render": RENDER,
        },
        {
            "Total Reward": total_reward,
            "Total Steps": steps,
            "Side Info": side_info,
        },
    )


if __name__ == "__main__":
    # Define command line arguments
    parser = argparse.ArgumentParser(
        description="test AISMCTS-F with openAI LLM on `GnomesAtNightEnv9A` environment"
    )
    parser.add_argument("--round", type=int, required=False, default=1)
    parser.add_argument("--explore", type=int, default=100, required=False)
    parser.add_argument(
        "--render", action="store_true", default=False, help="Enable rendering"
    )
    args = parser.parse_args()

    kwargs = {
        "ROUND": args.round,
        "EXPLORE": args.explore,
        "RENDER": args.render,
    }

    main(**kwargs)
