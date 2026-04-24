import argparse
import sys
import pygame
import random

from rescue_bot.env import RescueBotEnv
from rescue_bot.renderer import Renderer
from rescue_bot.config import Action, FPS

def run_human(env: RescueBotEnv, renderer: Renderer):
    """Mode joueur humain — contrôle au clavier."""
    total_reward = 0.0
    last_event   = ""
    obs          = env.reset()
    renderer.draw(total_reward, last_event)

    while True:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:    action = Action.UP
                if event.key == pygame.K_DOWN:  action = Action.DOWN
                if event.key == pygame.K_LEFT:  action = Action.LEFT
                if event.key == pygame.K_RIGHT: action = Action.RIGHT
                if event.key == pygame.K_p:     action = Action.PICKUP
                if event.key == pygame.K_d:     action = Action.DROP
                if event.key == pygame.K_r:
                    obs = env.reset(); total_reward = 0.0; last_event = "reset"
                if event.key == pygame.K_q:
                    renderer.close(); sys.exit()

        if action is not None:
            obs, reward, done, info = env.step(action)
            total_reward += reward
            last_event    = info.get("event", "")
            if done:
                last_event = "DONE — appuie R pour rejouer"

        renderer.draw(total_reward, last_event)
        renderer.tick(30)

def run_random(env: RescueBotEnv, renderer: Renderer):
    """Mode politique aléatoire — pour tester l'environnement."""
    obs          = env.reset()
    total_reward = 0.0
    last_event   = ""
    episode      = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                renderer.close(); sys.exit()

        mask   = env.get_action_mask()
        valid  = [a for a in range(6) if mask[a]]
        action = random.choice(valid) if valid else 0

        obs, reward, done, info = env.step(action)
        total_reward += reward
        last_event    = info.get("event", "")

        renderer.draw(total_reward, last_event)
        renderer.tick(FPS)

        if done:
            episode += 1
            print(f"Épisode {episode:4d} | Récompense : {total_reward:7.1f} | Évacués : {env.rescued}")
            obs          = env.reset()
            total_reward = 0.0
            last_event   = "nouvel épisode"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true", help="Politique aléatoire")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    env      = RescueBotEnv(seed=args.seed)
    renderer = Renderer(env)

    print(f"Taille observation : {env.obs_size}")
    print(f"Nombre d'actions   : {env.n_actions}")

    if args.random:
        run_random(env, renderer)
    else:
        run_human(env, renderer)


if __name__ == "__main__":
    main()