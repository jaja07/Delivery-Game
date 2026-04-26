import argparse
import sys
import os
from pathlib import Path
import pygame
import random
import torch

from rescue_bot.env import RescueBotEnv
from rescue_bot.renderer import Renderer
from rescue_bot.config import Action, FPS, WIN_W, WIN_H
from rescue_bot.policy import PolicyNet


def resolve_model_path(model_arg: str) -> str:
    """Resolve model path from common execution locations."""
    candidate = Path(model_arg)
    if candidate.exists():
        return str(candidate)

    module_dir = Path(__file__).resolve().parent
    search_paths = [
        module_dir / model_arg,        # Running from outside `rescue_bot/`
        module_dir.parent / model_arg, # Running from project root
    ]
    for path in search_paths:
        if path.exists():
            return str(path)

    # Fallback to the original value so error handling keeps clear user input.
    return model_arg

def show_menu(renderer: Renderer):
    """Display the main menu and wait for the player's choice."""
    font_title = pygame.font.SysFont("monospace", 45, bold=True)
    font_desc = pygame.font.SysFont("monospace", 15)
    font_btn = pygame.font.SysFont("monospace", 20, bold=True)
    
    c_bg, c_title, c_text = (10, 25, 40), (94, 234, 212), (226, 232, 240)
    c_btn, c_btn_hover = (13, 148, 136), (20, 184, 166)
    
    btn_w, btn_h = 280, 50
    btn_manual = pygame.Rect(WIN_W // 2 - btn_w // 2, 280, btn_w, btn_h)
    btn_auto = pygame.Rect(WIN_W // 2 - btn_w // 2, 350, btn_w, btn_h)
    
    description = [
        "Welcome to RescueBot!", "",
        "A disaster has struck. Your mission is",
        "to rescue isolated survivors (yellow circles).",
        "Avoid fire zones and drop survivors",
        "at evacuation zones (green crosses).", "",
        "Move quickly, every step costs you points!"
    ]

    while True:
        renderer.screen.fill(c_bg) # Draw on the fixed internal canvas.
        
        # --- Mouse position handling under window resizing ---
        raw_mouse_pos = pygame.mouse.get_pos()
        win_w, win_h = renderer.window.get_size()
        scale_x, scale_y = win_w / WIN_W, win_h / WIN_H
        # Virtual mouse position on the original surface
        mouse_pos = (raw_mouse_pos[0] / scale_x, raw_mouse_pos[1] / scale_y)
        
        # Text
        title = font_title.render("RESCUE BOT", True, c_title)
        renderer.screen.blit(title, (WIN_W // 2 - title.get_width() // 2, 60))
        for i, line in enumerate(description):
            txt = font_desc.render(line, True, c_text)
            renderer.screen.blit(txt, (WIN_W // 2 - txt.get_width() // 2, 130 + i * 20))
            
        # Buttons
        for btn, text in [(btn_manual, "Play (Manual Mode)"), (btn_auto, "Watch (AI Mode)")]:
            color = c_btn_hover if btn.collidepoint(mouse_pos) else c_btn
            pygame.draw.rect(renderer.screen, color, btn, border_radius=8)
            txt_surf = font_btn.render(text, True, c_bg)
            renderer.screen.blit(txt_surf, (btn.x + btn.width // 2 - txt_surf.get_width() // 2, 
                                            btn.y + btn.height // 2 - txt_surf.get_height() // 2))
            
        # Display on screen with scaling
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
    """Human player mode using keyboard controls."""
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

        # Continuous movement while an arrow key is held down
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
                last_event = "Game over - click Restart or press R"

        renderer.draw(total_reward, last_event)
        renderer.tick(30)

def run_trained(env: RescueBotEnv, renderer: Renderer, model_path: str):
    """Trained-agent mode using a saved PyTorch model."""
    if not os.path.exists(model_path):
        print(f"Error: model '{model_path}' was not found.")
        print("Did you finish training and save the model?")
        print("Starting human mode by default.")
        run_human(env, renderer)
        return

    # Load the policy network
    policy = PolicyNet(obs_size=env.obs_size, n_actions=env.n_actions)
    policy.load_state_dict(torch.load(model_path, weights_only=True))
    policy.eval()

    obs = env.reset()
    total_reward = 0.0
    last_event = "Model loaded"
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
            print(f"Episode {episode:4d} | Reward: {total_reward:7.1f} | Evacuated: {env.rescued}")
            obs = env.reset()
            total_reward = 0.0
            last_event = "new episode"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rescue_bot_policy.pth", help="Path to the AI model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    resolved_model_path = resolve_model_path(args.model)

    # Initialize environment and renderer
    env      = RescueBotEnv(seed=args.seed)
    renderer = Renderer(env)

    # Show menu
    mode = show_menu(renderer)

    # Start selected mode
    if mode == "manual":
        run_human(env, renderer)
    elif mode == "auto":
        run_trained(env, renderer, resolved_model_path)

if __name__ == "__main__":
    main()