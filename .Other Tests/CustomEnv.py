import sys
import numpy as np
import gymnasium as gym
import pygame

class CustomEnv(gym.Env):
    # ? Class Constructor
    def __init__(self, 
                grid_size = 9, 
                goal_coordinates = {
                    'Bar1' : np.array([3, 8]), 
                    'Bar2' : np.array([4, 8]), 
                    'Bar3' : np.array([5, 8])
                }, 
                random_initialization = False,
                sound = True) -> None:
        super().__init__()

        self.step_count = 0
        self.state = None
        self.done = False
        self.info = {}
        self.reward = 0
        self.cell_size = 50
        self.grid_size = grid_size
        self.goal = goal_coordinates
        self.random_initialization = random_initialization
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low = 0, high = 4, shape = (2,), dtype = np.int32)
        self.sound = sound

        self.danger_states = []
        
        if self.sound is True: 
            pygame.mixer.init(frequency = 22050, size = -16, channels = 2, buffer = 512)  # Initialize the mixer module.
            self.soundChannelsInitializer()
        
        self.bgObjectInitializer()
        
        self.soundEffectsInitializer()
    
    # ? Defining Sound Channels for audio
    def soundChannelsInitializer(self):
        self.channel_joy = pygame.mixer.Channel(0)
        self.channel_run = pygame.mixer.Channel(1)
        self.channel_whistle = pygame.mixer.Channel(2)
        self.channel_boo = pygame.mixer.Channel(3)
        self.channel_applause = pygame.mixer.Channel(4)
     
    # ? Initializing Sound Effects    
    def soundEffectsInitializer(self):
        self.sound_effect_joy = pygame.mixer.Sound("./resources/audio/JOY.mp3")  # Load a sound.
        #? Source: Sound Effect by freesound_community from Pixabay [https://pixabay.com/sound-effects/running-in-grass-6237/]
        #! License: Free (https://pixabay.com/service/license-summary/)
        #! Modification: None
        self.sound_effect_run = pygame.mixer.Sound("./resources/audio/RUNNING.mp3")  # Load a sound.
        #? Source: Sound Effect by freesound_community from Pixabay [https://pixabay.com/sound-effects/running-in-grass-6237/]
        #! License: Free (https://pixabay.com/service/license-summary/)
        #! Modification: None
        self.sound_effect_whistle = pygame.mixer.Sound("./resources/audio/WHISTLE.mp3")  # Load a sound.
        #? Source: Sound Effect by freesound_community from Pixabay [https://pixabay.com/sound-effects/referee-whistle-blow-gymnasium-6320/]
        #! License: Free (https://pixabay.com/service/license-summary/)
        #! Modification: None
        self.sound_effect_boo = pygame.mixer.Sound("./resources/audio/BOO.mp3")  # Load a sound.
        #? Source: Sound Effect by freesound_community from Pixabay [https://pixabay.com/sound-effects/boo-6377/]
        #! License: Free (https://pixabay.com/service/license-summary/)
        #! Modification: None
        self.sound_effect_applause = pygame.mixer.Sound("./resources/audio/APPLAUSE.mp3")  # Load a sound.
        #? Source: Sound Effect by freesound_community from Pixabay [https://pixabay.com/sound-effects/crowd-applause-236697/]
        #! License: Free (https://pixabay.com/service/license-summary/)
        #! Modification: None

    # ? Initializing Background Images 
    def bgObjectInitializer(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.cell_size*self.grid_size, self.cell_size*self.grid_size))
        pygame.display.set_caption('PADM_SS2025_PROJECT')
        self.background_image = pygame.image.load("./resources/img/FIELD.png").convert_alpha()
        #? Source: https://www.freepik.com/free-vector/soccer-field-background-with-scoreboard_2077950.htm?
        #! License: Free (https://www.freepik.com/legal/terms-of-use#nav-freepik-license)
        #! Modification: Cropped
        self.obstacle_image = pygame.image.load("./resources/img/DEFENDER.png").convert_alpha()
        #? Source: https://www.freepik.com/icon/soccer-player_3948791
        #! License: Free (https://www.freepik.com/legal/terms-of-use#nav-freepik-license)
        #! Modification: None
        self.agent_image = pygame.image.load("./resources/img/PLAYER.png").convert_alpha()
        #? Source: https://www.freepik.com/icon/football_1099672
        #! License: Free (https://www.freepik.com/legal/terms-of-use#nav-freepik-license)  
        #! Modification: None
     
    # ? Adding Danger States   
    def addDanger(self, coordinates, role):
        self.danger_states.append({
            'coordinates': coordinates, 
            'role': role
        })
    
    # ? Distance between Agent and Goal
    def distanceToGoal(self):
        # ! Euclidean Distance (Nearest) [sqrt( (x - x_i)^2 + (y - y_i)^2 )]
        return min([
            np.linalg.norm(self.state - goal_coord)
            for goal_coord in self.goal.values()
        ])

    # ? Resetting to initial state
    def reset(self):
        self.state = np.array([4, 0])
        self.done = False
        self.reward = 0
        self.step_count = 0

        self.info["Distance to goal"] = self.distanceToGoal()
        
        if self.sound:
            self.channel_whistle.play(self.sound_effect_whistle)

        return self.state, self.info

    # ? Agent's movement
    def step(self, action):
        if self.sound is True: self.sound_effect_run.play() 
        # Up: 0
        if action == 0 and self.state[0] > 0:
            self.state[0] -= 1

        # Down: 1
        if action == 1 and self.state[0] < self.grid_size - 1:
            self.state[0] += 1

        # Right: 2
        if action == 2 and self.state[1] < self.grid_size - 1:
            self.state[1] += 1

        # Left: 3
        if action == 3 and self.state[1] > 0:
            self.state[1] -= 1
        
        self.step_count += 1
        
        return self.checkTermination()

    # ? Check Termination and Rewards 
    def checkTermination(self):
        oldDistance = self.info["Distance to goal"]
        self.info["Distance to goal"] = self.distanceToGoal()
        
        # ! Position basis point
        # variablePoint = 0.1 if oldDistance > self.info["Distance to goal"] else (0 if oldDistance == self.info["Distance to goal"] else -0.2)
        
        
        # ! Goal:
        if np.array_equal(self.state, self.goal['Bar1']) or np.array_equal(self.state, self.goal['Bar2']) or np.array_equal(self.state, self.goal['Bar3']):
            if self.sound:
                self.channel_applause.play(self.sound_effect_applause)
            self.done = True
            # self.reward += 10 - (self.step_count / 100)
            self.reward += 10
        #  ! Danger:
        elif True in [np.array_equal(self.state, each_danger['coordinates']) for each_danger in self.danger_states]:
            if self.sound:
                self.channel_boo.play(self.sound_effect_boo)
            self.done = True
            # self.reward = abs(self.reward) * (-1) - 10 - (self.step_count / 100)
            self.reward -= 10 
        # else:
        #     self.done = False
        #     # self.reward += variablePoint - (self.step_count / 100)
        #     self.reward -= .1

        return self.state, self.done, self.reward, self.info

    # ? Environment Render
    def render(self):
        # Closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Background Color:
        self.screen.fill((11, 74, 1))

        # Field Grass
        self.background_image = pygame.transform.scale(
            self.background_image,
            (self.cell_size * self.grid_size, self.cell_size * self.grid_size)
        )
        
        self.screen.blit(self.background_image, (0, 0))  # Draw image background

        # # ! Code for grid lines:
        # for col in range(self.grid_size):
        #     for row in range(self.grid_size):
        #         grid = pygame.Rect(col*self.cell_size,
        #                            row*self.cell_size,
        #                            self.cell_size,
        #                            self.cell_size)
        #         pygame.draw.rect(self.screen,
        #                          (11, 74, 1),
        #                          grid,
        #                          1)
                
        # Opponent & GK:
        for each_danger in self.danger_states:
            temp_obstacle_image = self.obstacle_image
            if each_danger['role'] == "GK":
                self.obstacle_image = pygame.image.load("./resources/img/GOALKEEPER.png").convert_alpha()
                #? Source: https://www.freepik.com/icon/goalkeeper_3564495
                #! License: Free (https://www.freepik.com/legal/terms-of-use#nav-freepik-license)  
                #! Modification: None

            self.obstacle_image = pygame.transform.scale(
                self.obstacle_image,
                (self.cell_size, self.cell_size)
            )
            self.screen.blit(
                self.obstacle_image,
                (each_danger['coordinates'][1]*self.cell_size, each_danger['coordinates'][0]*self.cell_size)
            )
            self.obstacle_image = temp_obstacle_image
            
            
        # Agent:
        self.agent_image = pygame.transform.scale(
            self.agent_image,
            (self.cell_size, self.cell_size)
        )
        self.screen.blit(
            self.agent_image,
            (self.state[1]*self.cell_size, self.state[0]*self.cell_size)
        )
        

        pygame.time.wait(300)
        pygame.display.flip()

    # ? Close Function
    def close(self):
        pygame.quit()
        
# ? Instance Creator
def createEnv(goal_coordinates,
            danger_coordinates,
            random_initialization,
            sound):
    
    env = CustomEnv(goal_coordinates = goal_coordinates,
                random_initialization = random_initialization,
                sound = sound)

    if env.sound:
        env.channel_joy.play(env.sound_effect_joy)
    for danger in danger_coordinates:
        env.addDanger(coordinates = danger['coordinates'], role = danger['role'])

    return env