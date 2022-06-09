import pygame
import random
 
# Global constants

# What frame is the game on 
f = 0
 
# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
 
# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
 
 
class Player(pygame.sprite.Sprite):
    """
    This class represents the bar at the bottom that the player controls.
    """
 
    # -- Methods
    def __init__(self):
        """ Constructor function """
 
        # Call the parent's constructor
        super().__init__()
 
        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        width = 30
        height = 45
        self.image = pygame.image.load("ninja.png")
        self.image = pygame.transform.scale(self.image, [width, height])
        
 
        # Set a referance to the image rect.
        self.rect = self.image.get_rect()
 
        # Set speed vector of player
        self.change_x = 0
        self.change_y = 0
 
        # List of sprites we can bump against
        self.level = None
 
    def update(self):
        """ Move the player. """
        # Gravity
        self.calc_grav()
 
        # Move left/right
        self.rect.x += self.change_x
 
        # See if we hit anything
        block_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        for block in block_hit_list:
            # If we are moving right,
            # set our right side to the left side of the item we hit
            if self.change_x > 0:
                self.rect.right = block.rect.left
            elif self.change_x < 0:
                # Otherwise if we are moving left, do the opposite.
                self.rect.left = block.rect.right
 
        # Move up/down
        self.rect.y += self.change_y
 
        # Check and see if we hit anything
        block_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        for block in block_hit_list:
 
            # Reset our position based on the top/bottom of the object.
            if self.change_y > 0:
                self.rect.bottom = block.rect.top
            elif self.change_y < 0:
                self.rect.top = block.rect.bottom
 
            # Stop our vertical movement
            self.change_y = 0
 
            if isinstance(block, MovingPlatform):
                self.rect.x += block.change_x
 
    def calc_grav(self):
        """ Calculate effect of gravity. """
        if self.change_y == 0:
            self.change_y = 1
        else:
            self.change_y += .35
 
        # See if we are on the ground.
        if self.rect.y >= SCREEN_HEIGHT - self.rect.height and self.change_y >= 0:
            self.change_y = 0
            self.rect.y = SCREEN_HEIGHT - self.rect.height
 
    def jump(self):
        """ Called when user hits 'jump' button. """
 
        # move down a bit and see if there is a platform below us.
        # Move down 2 pixels because it doesn't work well if we only move down
        # 1 when working with a platform moving down.
        self.rect.y += 2
        platform_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        self.rect.y -= 2
 
        # If it is ok to jump, set our speed upwards
        if len(platform_hit_list) > 0 or self.rect.bottom >= SCREEN_HEIGHT:
            self.change_y = -10
 
    # Player-controlled movement:
    def go_left(self):
        """ Called when the user hits the left arrow. """
        self.change_x = -6
 
    def go_right(self):
        """ Called when the user hits the right arrow. """
        self.change_x = 6
 
    def stop(self):
        """ Called when the user lets off the keyboard. """
        self.change_x = 0
 
 
class Platform(pygame.sprite.Sprite):
    """ Platform the user can jump on """
 
    def __init__(self, width, height):
        """ Platform constructor. Assumes constructed with user passing in
            an array of 5 numbers like what's defined at the top of this code.
            """
        super().__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(GREEN)
 
        self.rect = self.image.get_rect()
 
 
class MovingPlatform(Platform):
    """ This is a fancier platform that can actually move. """
    change_x = 0
    change_y = 0
 
    player = None
 
    level = None
 
    def update(self):
        """ Move the platform.
            If the player is in the way, it will shove the player
            out of the way. This does NOT handle what happens if a
            platform shoves a player into another object. Make sure
            moving platforms have clearance to push the player around
            or add code to handle what happens if they don't. """
 
        # Move left/right
        self.rect.x += self.change_x
 
        # See if we hit the player
        hit = pygame.sprite.collide_rect(self, self.player)
        if hit:
            # We did hit the player. Shove the player around and
            # assume he/she won't hit anything else.
 
            # If we are moving right, set our right side
            # to the left side of the item we hit
            if self.change_x < 0:
                self.player.rect.right = self.rect.left
            else:
                # Otherwise if we are moving left, do the opposite.
                self.player.rect.left = self.rect.right
 
        # Move up/down
        self.rect.y += self.change_y
 
        # Check and see if we the player
        hit = pygame.sprite.collide_rect(self, self.player)
        if hit:
            # We did hit the player. Shove the player around and
            # assume he/she won't hit anything else.
 
            # Reset our position based on the top/bottom of the object.
            if self.change_y < 0:
                self.player.rect.bottom = self.rect.top
            else:
                self.player.rect.top = self.rect.bottom
 
 
class Level(object):
    """ This is a generic super-class used to define a level.
        Create a child class for each level with level-specific
        info. """
 
 
    def __init__(self, player):
        """ Constructor. Pass in a handle to player. Needed for when moving
            platforms collide with the player. """
        self.platform_list = pygame.sprite.Group()
        self.enemy_list = pygame.sprite.Group()
        self.player = player
         
        # Background image
        self.background = None
     
 
    # Update everythign on this level
    def update(self):
        """ Update everything in this level."""

        # Generate a new planks every X frames
        if(f % 300 == 0):
            planks_generated = 2
            for i in range(planks_generated):
                # Add a custom moving platform
                block = MovingPlatform(200, 20)
                block.rect.x = 30+300*i
                block.rect.y = 80
                block.change_y = 1
                block.player = self.player
                block.level = self
                self.platform_list.add(block)

        # Generate a new planks every X frames
        if(f % 300 == 150):
            planks_generated = 1
            for i in range(planks_generated):
                # Add a custom moving platform
                block = MovingPlatform(200, 20)
                block.rect.x = 200
                block.rect.y = 80
                block.change_y = 1
                block.player = self.player
                block.level = self
                self.platform_list.add(block)

        self.platform_list.update()
        self.enemy_list.update()

        # Check if the platform is in the main screen
        new_platform_list = pygame.sprite.Group()
        for platform in self.platform_list:
            if platform.rect.y < SCREEN_HEIGHT-100:
                new_platform_list.add(platform)

        self.platform_list = new_platform_list

        
 
    def draw(self, screen):
        """ Draw everything on this level. """
 
        # Draw the background
        screen.fill(BLUE)
        pygame.draw.rect(screen, BLACK, [0, 0, 600, 100], 0)
        pygame.draw.rect(screen, BLACK, [0, 500, 600, 100], 0)
        pygame.draw.rect(screen, RED, [0, 97, 600, 3], 0)
        pygame.draw.rect(screen, RED, [0, 500, 600, 3], 0)


 
        # Draw all the sprite lists that we have
        self.platform_list.draw(screen)
        self.enemy_list.draw(screen)
 
 
# Create platforms for the level
class Level_01(Level):
    """ Definition for level 1. """
 
    def __init__(self, player):
        """ Create level 1. """
 
        # Call the parent constructor
        Level.__init__(self, player)
 
        # Add a custom moving platform
        block = MovingPlatform(200, 20)
        block.rect.x = 200
        block.rect.y = 230
        block.change_y = 1
        block.player = self.player
        block.level = self
        self.platform_list.add(block)
 
 
class LegendaryNinja_v0():

    def __init__(self, render):

        self.render = render
        self.reward = 0
        pygame.init()

        # Create the player
        self.player = Player()
        # Create the level
        self.current_level = Level_01(self.player)   
        self.active_sprite_list = pygame.sprite.Group()
        self.player.level = self.current_level

        self.player.rect.x = 200
        self.player.rect.y = 230 - self.player.rect.height
        self.active_sprite_list.add(self.player)

        self.player_platform = self.current_level.platform_list.sprites()[0]
        # left, right, up, stop
        self.action_space = [0, 1, 2, 3]

        # Update observation space
        self.observation_space = [0 for i in range(18)]

        if(render):
            # ALL CODE TO DRAW SHOULD GO BELOW THIS COMMENT
            # Set the height and width of the screen
            size = [SCREEN_WIDTH, SCREEN_HEIGHT]
            self.screen = pygame.display.set_mode(size)
            pygame.display.set_caption("LegendaryNinja")
            self.current_level.draw(self.screen)
            self.active_sprite_list.draw(self.screen)
            # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT
        else:
            pygame.display.quit()
       

    
    def step(self, action):
        done = False
        
        self.reward = 0.1
        
        if action == self.action_space[0]:
            self.player.go_left()
        if action == self.action_space[1]:
            self.player.go_right()
        if action == self.action_space[2]:
            self.player.jump()
        if action == self.action_space[3]:
            pass

        # Update the player.
        self.active_sprite_list.update()

        # Update items in the level
        self.current_level.update()

        self.player.stop()

        if((self.player.rect.y > SCREEN_HEIGHT - 100 - self.player.rect.height)
        or (self.player.rect.y <  100)):
            done = True    

        
        p = self.current_level.platform_list.sprites()
        for i in range(len(p)):
            if(self.player.rect.x < p[i].rect.x+p[i].rect.width
            and self.player.rect.x > p[i].rect.x-self.player.rect.width
            and self.player.rect.y + self.player.rect.height <= p[i].rect.y
            and self.player.rect.y + self.player.rect.height >= p[i].rect.y-1):
                if(self.player_platform != p[i]):
                    print()
                    self.player_platform = p[i]
                    self.reward += 0.5

        if(self.render):
            # ALL CODE TO DRAW SHOULD GO BELOW THIS COMMENT
            self.current_level.draw(self.screen)
            self.active_sprite_list.draw(self.screen)

            font = pygame.font.Font('freesansbold.ttf', 16)
            text = font.render(str(int(self.reward)), True, WHITE, BLACK)
            textRect = text.get_rect()
            textRect.center = (32, 32)
            self.screen.blit(text, textRect)
            
            # Limit to 60 frames per second
            pygame.time.Clock().tick(60)

            # Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT

         
        # Update observation space
        self.observation_space = [0 for i in range(18)]
        self.observation_space[0] = self.player.rect.x
        self.observation_space[1] = self.player.rect.y
        self.observation_space[2] = self.player.change_y
        
        for i in range(len(p)):
            self.observation_space[3*i+3] = p[i].rect.x
            self.observation_space[3*i+4] = p[i].rect.x+p[i].rect.width
            self.observation_space[3*i+5] = p[i].rect.y
        
        # update the frame count
        global f
        f = f + 1
        
        return self.observation_space, self.reward, done


    def reset(self):
        global f
        f = 0
        self.reward = 0
         # Create the player
        self.player = Player()
        # Create the level
        self.current_level = Level_01(self.player)   
        self.active_sprite_list = pygame.sprite.Group()
        self.player.level = self.current_level
    
        self.player.rect.x = 200
        self.player.rect.y = 230 - self.player.rect.height
        self.active_sprite_list.add(self.player)


        # Update observation space
        self.observation_space = [0 for i in range(18)]
        self.observation_space[0] = self.player.rect.x
        self.observation_space[1] = self.player.rect.y
        self.observation_space[2] = self.player.change_y
        p = self.current_level.platform_list.sprites()
        
        for i in range(len(p)):
            self.observation_space[3*i+3] = p[i].rect.x
            self.observation_space[3*i+4] = p[i].rect.x+p[i].rect.width
            self.observation_space[3*i+5] = p[i].rect.y

        self.player_platform = self.current_level.platform_list.sprites()[0]

        if(self.render):
            # ALL CODE TO DRAW SHOULD GO BELOW THIS COMMENT
            self.current_level.draw(self.screen)
            self.active_sprite_list.draw(self.screen)
            # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT

        return self.observation_space

    def close(self):
        pygame.quit()
 
 