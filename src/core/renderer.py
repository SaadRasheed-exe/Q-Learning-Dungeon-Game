import pygame
import time
from .config import RenderingConfig as cfg

class Renderer:
    def __init__(self):

        self.texture_pack = {
            cfg.WALL: pygame.image.load(f"{cfg.ASSETPATH}/Wall.png"),
            cfg.WALKABLE: pygame.image.load(f"{cfg.ASSETPATH}/Walkable.png"),
            cfg.AGENT: pygame.image.load(f"{cfg.ASSETPATH}/Agent.png"),
            cfg.KEY: pygame.image.load(f"{cfg.ASSETPATH}/Key.png"),
            cfg.LAVA: pygame.image.load(f"{cfg.ASSETPATH}/Lava.png"),
            cfg.GOAL: pygame.image.load(f"{cfg.ASSETPATH}/Goal.png")
        }

        self.grid_size = cfg.GRID_SIZE
        self.cell_size = cfg.CELL_SIZE

        native_res = (self.grid_size[1] * self.cell_size, self.grid_size[0] * self.cell_size)
        window_size = native_res[0] * cfg.UPSCALE_FACTOR, native_res[1] * cfg.UPSCALE_FACTOR

        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        self.render_surface = pygame.Surface(native_res)
        pygame.display.set_caption("Dungeon Environment")
        self.clock = pygame.time.Clock()
        time.sleep(1)

    def draw_grid(self, grid):

        self.render_surface.fill((200, 200, 200))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # grid
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                cell_type = grid[i, j]

                if cell_type == cfg.AGENT:
                    self.render_surface.blit(
                        self.texture_pack[cfg.WALKABLE],
                        (j * self.cell_size, i * self.cell_size)
                    )

                self.render_surface.blit(
                    self.texture_pack[cell_type],
                    (j * self.cell_size, i * self.cell_size)
                )
            
        scaled_surface = pygame.transform.scale(self.render_surface, self.screen.get_size())
        self.screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(cfg.TICKS_PER_SECOND)