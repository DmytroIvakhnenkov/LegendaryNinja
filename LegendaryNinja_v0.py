import pygame
import random
 
# Global constants

# What frame is the game on plus 149
f = -1
 
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
        width = 40
        height = 60
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
        if(f % 150 == 0):
            planks_generated = random.randint(1,3)
            for i in range(planks_generated):
                # Add a custom moving platform
                platform_width = random.randint(40,(SCREEN_WIDTH/planks_generated)-self.player.rect.width*1.5)
                if(platform_width > SCREEN_WIDTH*0.4):
                    platform_width = SCREEN_WIDTH*0.4
                block = MovingPlatform(platform_width, 40)
                x1 = (SCREEN_WIDTH/planks_generated)*i
                x2 = ((SCREEN_WIDTH/planks_generated))*(i+1)-self.player.rect.width*1.5 - platform_width
                # need to make sure that x2 > platform_width + player_width
                block.rect.x = random.randint(x1,x2)
                block.rect.y = 59
                block.change_y = 1
                block.player = self.player
                block.level = self
                self.platform_list.add(block)

        self.platform_list.update()

        # Check if the platform is beyond the main screen
        new_platform_list = pygame.sprite.Group()
        for platform in self.platform_list:
            if platform.rect.y < SCREEN_HEIGHT-100:
                new_platform_list.add(platform)

        self.platform_list = new_platform_list

        self.enemy_list.update()
 
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
        block = MovingPlatform(70, 40)
        block.rect.x = 100
        block.rect.y = 230
        block.change_y = 1
        block.player = self.player
        block.level = self
        self.platform_list.add(block)

        # Add a custom moving platform
        block = MovingPlatform(70, 40)
        block.rect.x = 300
        block.rect.y = 230
        block.change_y = 1
        block.player = self.player
        block.level = self
        self.platform_list.add(block)
 
 
class LegendaryNinja_v0():

    def __init__(self):

        pygame.init()
    
        # Set the height and width of the screen
        size = [SCREEN_WIDTH, SCREEN_HEIGHT]
        self.screen = pygame.display.set_mode(size)
    
        pygame.display.set_caption("Platformer with moving platforms")
    
        # Create the player
        self.player = Player()
        # Create the level
        self.current_level = Level_01(self.player)   
        self.active_sprite_list = pygame.sprite.Group()
        self.player.level = self.current_level
    
        self.player.rect.x = 100
        self.player.rect.y = 230 - self.player.rect.height
        self.active_sprite_list.add(self.player)
        # left, right, up
        self.action_space = [0, 1, 2, 3]

        # Update observation space
        self.observation_space = [0 for i in range(29)]

    

    def step(self, action):
        done = False
        reward = 1
        global f
        f = f + 1
        if action == self.action_space[0]:
            self.player.go_left()
        if action == self.action_space[1]:
            self.player.go_right()
        if action == self.action_space[2]:
            self.player.jump()
        if action == self.action_space[3]:
            self.player.stop()

        # Update the player.
        self.active_sprite_list.update()

        # Update items in the level
        self.current_level.update()

        if((self.player.rect.y > SCREEN_HEIGHT - 100 - self.player.rect.height)
        or (self.player.rect.y <  100)):
            done = True

        # ALL CODE TO DRAW SHOULD GO BELOW THIS COMMENT
        #self.current_level.draw(self.screen)
        #self.active_sprite_list.draw(self.screen)

        # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT

        # Limit to 60 frames per second
        #pygame.time.Clock().tick(60)

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()
        
        # Update observation space
        self.observation_space = [0 for i in range(29)]
        self.observation_space[0] = self.player.rect.x/600
        self.observation_space[1] = self.player.rect.y/600
        p = self.current_level.platform_list.sprites()
        try:
            for i in range(len(p)):
                self.observation_space[3*i+2] = p[i].rect.width/600
                self.observation_space[3*i+3] = p[i].rect.x/600
                self.observation_space[3*i+4] = p[i].rect.y/600
        except:
            for i in range(9):
                self.observation_space[3*i+2] = p[i].rect.width/600
                self.observation_space[3*i+3] = p[i].rect.x/600
                self.observation_space[3*i+4] = p[i].rect.y/600

        return self.observation_space, reward, done


    def reset(self):
        global f
        f = -1
         # Create the player
        self.player = Player()
        # Create the level
        self.current_level = Level_01(self.player)   
        self.active_sprite_list = pygame.sprite.Group()
        self.player.level = self.current_level
    
        self.player.rect.x = 100
        self.player.rect.y = 230 - self.player.rect.height
        self.active_sprite_list.add(self.player)


        # Update observation space
        self.observation_space = [0 for i in range(29)]
        self.observation_space[0] = self.player.rect.x/600
        self.observation_space[1] = self.player.rect.y/600
        p = self.current_level.platform_list.sprites()
        
        for i in range(len(p)):
            self.observation_space[3*i+2] = p[i].rect.width/600
            self.observation_space[3*i+3] = p[i].rect.x/600
            self.observation_space[3*i+4] = p[i].rect.y/600

        return self.observation_space

    def close(self):
        pygame.quit()
 
 