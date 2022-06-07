        if(f % 300 == 150):
            planks_generated = 1
            for i in range(planks_generated):
                # Add a custom moving platform
                platform_width = random.randint(40,(SCREEN_WIDTH/planks_generated)-self.player.rect.width*3)
                if(platform_width > SCREEN_WIDTH*0.4):
                    platform_width = SCREEN_WIDTH*0.4
                block = MovingPlatform(platform_width, 10)
                x1 = (SCREEN_WIDTH/planks_generated)*i
                x2 = ((SCREEN_WIDTH/planks_generated))*(i+1)-self.player.rect.width*3 - platform_width
                # need to make sure that x2 > platform_width + player_width
                block.rect.x = random.randint(x1,x2)
                block.rect.y = 60
                block.change_y = 1
                block.player = self.player
                block.level = self
                self.platform_list.add(block)