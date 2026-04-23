import pygame
import csv
import random
import time

# ── Setup ──────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Motor Imagery Paradigm")
font_large = pygame.font.SysFont("Arial", 80)
font_small = pygame.font.SysFont("Arial", 40)

BLACK  = (0, 0, 0)
WHITE  = (255, 255, 255)
YELLOW = (255, 220, 0)
GRAY   = (150, 150, 150)

cues       = ["OPEN HAND", "CLOSE HAND"]
num_trials = 10

# ── Helper: show text on screen ──
def show_text(text, color, duration):
    screen.fill(BLACK)
    label = font_large.render(text, True, color)
    rect  = label.get_rect(center=(400, 300))
    screen.blit(label, rect)
    pygame.display.flip()
    time.sleep(duration)

# ── CSV Setup ───
with open("trial_log.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["trial", "cue", "timestamp"])

    # ── Trial Loop ───
    for i in range(num_trials):
        cue = random.choice(cues)

        # 1. Fixation cross — 2 seconds
        show_text("+", WHITE, 2.0)

        # 2. Cue — 4 seconds
        show_text(cue, YELLOW, 4.0)
        writer.writerow([i + 1, cue, round(time.time(), 3)])

        # 3. Rest — 2 seconds
        show_text("REST", GRAY, 2.0)

        print(f"Trial {i+1}: {cue}")

# ── Done ──
show_text("DONE!", WHITE, 2.0)
pygame.quit()
print("Finished! Check trial_log.csv")