import argparse
import sys
import os
import pygame
import random
import torch

from rescue_bot.env import RescueBotEnv
from rescue_bot.renderer import Renderer
from rescue_bot.config import Action, FPS, WIN_W, WIN_H
from rescue_bot.policy import PolicyNet

def show_menu(renderer: Renderer):
    """Affiche le menu principal et attend le choix du joueur."""
    # Polices d'écriture
    font_title = pygame.font.SysFont("monospace", 45, bold=True)
    font_desc = pygame.font.SysFont("monospace", 15)
    font_btn = pygame.font.SysFont("monospace", 20, bold=True)
    
    c_bg, c_title, c_text = (10, 25, 40), (94, 234, 212), (226, 232, 240)
    c_btn, c_btn_hover = (13, 148, 136), (20, 184, 166)
    
    btn_w, btn_h = 280, 50
    btn_manual = pygame.Rect(WIN_W // 2 - btn_w // 2, 280, btn_w, btn_h)
    btn_auto = pygame.Rect(WIN_W // 2 - btn_w // 2, 350, btn_w, btn_h)
    
    description = [
        "Bienvenue dans RescueBot !", "",
        "Une catastrophe a eu lieu. Votre mission est",
        "de secourir les survivants isolés (ronds jaunes).",
        "Évitez les zones d'incendie et déposez les",
        "blessés sur les zones d'évacuation (croix vertes).", "",
        "Faites vite, chaque déplacement vous coûte des points !"
    ]

    while True:
        renderer.screen.fill(c_bg) # Dessin sur le canevas fixe
        
        # --- Gestion du redimensionnement de la souris ---
        raw_mouse_pos = pygame.mouse.get_pos()
        win_w, win_h = renderer.window.get_size()
        scale_x, scale_y = win_w / WIN_W, win_h / WIN_H
        # Position virtuelle de la souris sur la surface originale
        mouse_pos = (raw_mouse_pos[0] / scale_x, raw_mouse_pos[1] / scale_y)
        
        # Textes
        title = font_title.render("RESCUE BOT", True, c_title)
        renderer.screen.blit(title, (WIN_W // 2 - title.get_width() // 2, 60))
        for i, line in enumerate(description):
            txt = font_desc.render(line, True, c_text)
            renderer.screen.blit(txt, (WIN_W // 2 - txt.get_width() // 2, 130 + i * 20))
            
        # Boutons
        for btn, text in [(btn_manual, "Jouer (Mode Manuel)"), (btn_auto, "Observer (Mode IA)")]:
            color = c_btn_hover if btn.collidepoint(mouse_pos) else c_btn
            pygame.draw.rect(renderer.screen, color, btn, border_radius=8)
            txt_surf = font_btn.render(text, True, c_bg)
            renderer.screen.blit(txt_surf, (btn.x + btn.width // 2 - txt_surf.get_width() // 2, 
                                            btn.y + btn.height // 2 - txt_surf.get_height() // 2))
            
        # Affichage à l'écran avec mise à l'échelle
        scaled_surface = pygame.transform.smoothscale(renderer.screen, (win_w, win_h))
        renderer.window.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_manual.collidepoint(mouse_pos):
                    return "manual"
                if btn_auto.collidepoint(mouse_pos):
                    return "auto"
                    
        renderer.clock.tick(30)

def run_human(env: RescueBotEnv, renderer: Renderer):
    """Mode joueur humain — contrôle au clavier."""
    total_reward = 0.0
    last_event   = ""
    obs          = env.reset()
    done         = False
    move_repeat_ms = 120
    next_move_at = 0
    renderer.draw(total_reward, last_event)

    while True:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close(); sys.exit()

            if renderer.is_restart_click(event):
                obs = env.reset()
                total_reward = 0.0
                last_event = "reset"
                done = False
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs = env.reset()
                    total_reward = 0.0
                    last_event = "reset"
                    done = False
                if event.key == pygame.K_q:
                    renderer.close(); sys.exit()

                if not done:
                    if event.key == pygame.K_UP:    action = Action.UP
                    if event.key == pygame.K_DOWN:  action = Action.DOWN
                    if event.key == pygame.K_LEFT:  action = Action.LEFT
                    if event.key == pygame.K_RIGHT: action = Action.RIGHT
                    if event.key == pygame.K_p:     action = Action.PICKUP
                    if event.key == pygame.K_d:     action = Action.DROP

        # Déplacement continu quand une flèche reste enfoncée
        if action is None and not done:
            keys = pygame.key.get_pressed()
            now = pygame.time.get_ticks()
            if now >= next_move_at:
                if keys[pygame.K_UP]:
                    action = Action.UP
                elif keys[pygame.K_DOWN]:
                    action = Action.DOWN
                elif keys[pygame.K_LEFT]:
                    action = Action.LEFT
                elif keys[pygame.K_RIGHT]:
                    action = Action.RIGHT

        if action is not None and not done:
            obs, reward, done, info = env.step(action)
            total_reward += reward
            last_event    = info.get("event", "")
            if action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
                next_move_at = pygame.time.get_ticks() + move_repeat_ms
            if done:
                last_event = "Partie terminee - clique Restart ou appuie R"

        renderer.draw(total_reward, last_event)
        renderer.tick(30)

def run_trained(env: RescueBotEnv, renderer: Renderer, model_path: str):
    """Mode agent entraîné — utilise le modèle PyTorch sauvegardé."""
    if not os.path.exists(model_path):
        print(f"⚠️ Erreur: Modèle '{model_path}' introuvable.")
        print("Avez-vous terminé l'entraînement et sauvegardé le modèle ?")
        print("Lancement du mode humain par défaut.")
        run_human(env, renderer)
        return

    # Chargement du réseau
    policy = PolicyNet(obs_size=env.obs_size, n_actions=env.n_actions)
    policy.load_state_dict(torch.load(model_path, weights_only=True))
    policy.eval()

    obs = env.reset()
    total_reward = 0.0
    last_event = "Modèle chargé"
    episode = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                renderer.close(); sys.exit()

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(env.get_action_mask(), dtype=torch.bool).unsqueeze(0)

        with torch.no_grad():
            dist = policy(obs_tensor, mask_tensor)
            action = dist.sample().item()

        obs, reward, done, info = env.step(action) # type: ignore
        total_reward += reward
        last_event = info.get("event", "")

        renderer.draw(total_reward, last_event)
        renderer.tick(FPS)

        if done:
            episode += 1
            print(f"Épisode {episode:4d} | Récompense : {total_reward:7.1f} | Évacués : {env.rescued}")
            obs = env.reset()
            total_reward = 0.0
            last_event = "nouvel épisode"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rescue_bot_policy.pth", help="Chemin du modèle IA")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Initialisation de l'environnement et du rendu
    env      = RescueBotEnv(seed=args.seed)
    renderer = Renderer(env)

    # Affichage du menu
    mode = show_menu(renderer)

    # Lancement selon le choix
    if mode == "manual":
        run_human(env, renderer)
    elif mode == "auto":
        run_trained(env, renderer, args.model)

if __name__ == "__main__":
    main()