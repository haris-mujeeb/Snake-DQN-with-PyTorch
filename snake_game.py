import pygame
from base_snake_game import BaseSnakeGame

# Colors
WHITE, GREEN, RED, BLUE, BLACK = (255, 255, 255), (0, 255, 0), (255, 0, 0), (0, 128, 255), (0, 0, 0)

GRID_SIZE = 20

class SnakeGame(BaseSnakeGame):
    def __init__(self, grid_width=30, grid_height=30, fps=10):
        super().__init__(grid_width, grid_height)
        pygame.init()
        self.pygame = pygame
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.screen = pygame.display.set_mode((grid_width * GRID_SIZE, grid_height * GRID_SIZE))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
        self.fps = fps

    def render(self):
        pygame = self.pygame
        self.screen.fill(BLACK)
        grid = self.get_grid()

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell = grid[y, x]
                rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                if cell == 1:
                    pygame.draw.rect(self.screen, GREEN, rect)
                elif cell == 2:
                    pygame.draw.rect(self.screen, RED, rect)
                elif cell == 9:
                    pygame.draw.rect(self.screen, BLUE, rect)

        # Score display
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()

    def run_human_play(self):
        pygame = self.pygame
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and self.direction != (0, 1):
                        self.direction = (0, -1)
                    elif event.key == pygame.K_DOWN and self.direction != (0, -1):
                        self.direction = (0, 1)
                    elif event.key == pygame.K_LEFT and self.direction != (1, 0):
                        self.direction = (-1, 0)
                    elif event.key == pygame.K_RIGHT and self.direction != (-1, 0):
                        self.direction = (1, 0)
                    elif event.key == pygame.K_r and self.game_over:
                        self.reset()

            if not self.game_over:
                _, _, done, _ = self.step(self.direction)
                if done:
                    print(f"Game Over! Final Score: {self.score}")

            self.render()
            self.clock.tick(self.fps)

        pygame.quit()
