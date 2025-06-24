import sys
# import time
import numpy as np
import gymnasium as gym
import pygame

class CustomEnv(gym.Env):
    def __init__(self, 
                 grid_size = 9, 
                 goal_coordinates = {'Bar1' : np.array([3, 8]), 
                                     'Bar2' : np.array([4, 8]), 
                                     'Bar3' : np.array([5, 8])}, 
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
        self.observation_space = gym.spaces.Box(low=0, high=4, shape=(2,), dtype=np.int32)
        self.sound = sound

        self.danger_states = []
        
        if self.sound is True: 
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)  # Initialize the mixer module.
            self.soundChannelsInitializer()
        
        self.bgObjectInitializer()
        
        self.soundEffectsInitializer()
        
     
    def soundChannelsInitializer(self):
        self.channel_joy = pygame.mixer.Channel(0)
        self.channel_run = pygame.mixer.Channel(1)
        self.channel_whistle = pygame.mixer.Channel(2)
        self.channel_boo = pygame.mixer.Channel(3)
        self.channel_applause = pygame.mixer.Channel(4)
           
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
        
    def addDanger(self, coordinates, role):
        self.danger_states.append({'coordinates': coordinates, 'role': role})
        
    def distanceToGoal(self):
        # Euclidean Distance
        return min([
            np.linalg.norm(self.state - goal_coord)
            for goal_coord in self.goal.values()
        ])

    def reset(self):
        self.state = np.array([4, 0])
        self.done = False
        self.reward = 0
        self.step_count = 0

        self.info["Distance to goal"] = self.distanceToGoal()
        
        if self.sound:
            self.channel_whistle.play(self.sound_effect_whistle)

        return self.state, self.info

    def step(self, action):
        if self.sound is True: self.sound_effect_run.play() 
        # Up: 0
        if action==0 and self.state[0]>0:
            self.state[0]-=1

        # Down: 1
        if action==1 and self.state[0]<self.grid_size-1:
            self.state[0] += 1


        # Right: 2
        if action==2 and self.state[1]<self.grid_size-1:
            self.state[1]+=1

        # Left: 3
        if action==3 and self.state[1]>0:
            self.state[1]-=1
        
        self.step_count += 1
        
        return self.checkTermination()

    def checkTermination(self):
        # Info:
        oldDistance = self.info["Distance to goal"]
        self.info["Distance to goal"] = self.distanceToGoal()
        
        variablePoint = 0.1 if oldDistance > self.info["Distance to goal"] else (0 if oldDistance == self.info["Distance to goal"] else -0.2)
        
        
        # Goal:
        if np.array_equal(self.state, self.goal['Bar1']) or np.array_equal(self.state, self.goal['Bar2']) or np.array_equal(self.state, self.goal['Bar3']):
            if self.sound:
                self.channel_applause.play(self.sound_effect_applause)
            self.done = True
            self.reward += 10 - (self.step_count/100)
        # Danger:
        elif True in [np.array_equal(self.state, each_danger['coordinates']) for each_danger in self.danger_states]:
            if self.sound:
                self.channel_boo.play(self.sound_effect_boo)
            self.done = True
            self.reward = abs(self.reward) * (-1) - 10 - (self.step_count/100)
        else:
            self.done = False
            self.reward += variablePoint - (self.step_count/100)

        return self.state, self.done, self.reward, self.info

    def render(self):
        # Code for closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Background:
        # -----------
        self.screen.fill((11, 74, 1))

        self.background_image = pygame.transform.scale(
            self.background_image,
            (self.cell_size * self.grid_size, self.cell_size * self.grid_size)
        )
        
        self.screen.blit(self.background_image, (0, 0))  # Draw image background

        # # Draw gridlines:
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
                
        # Add danger states:
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
            
            
        # Draw agent:
        self.agent_image = pygame.transform.scale(
            self.agent_image,
            (self.cell_size, self.cell_size)
        )
        self.screen.blit(
            self.agent_image,
            (self.state[1]*self.cell_size, self.state[0]*self.cell_size)
        )
        

        pygame.time.wait(100)
        pygame.display.flip()

    def close(self):
        pygame.quit()
        
# Function 1: Create an instance of the environment
# -----------
def createEnv(goal_coordinates,
            danger_coordinates,
            random_initialization,
            sound):
    
    
    # Create the environment:
    # -----------------------
    env = CustomEnv(goal_coordinates=goal_coordinates,
                random_initialization=random_initialization,
                sound=sound)

    if env.sound:
                env.channel_joy.play(env.sound_effect_joy)
    for danger in danger_coordinates:
        env.addDanger(coordinates = danger['coordinates'], role = danger['role'])

    return env

# if __name__=="__main__":
#     pygame.mixer.init()  # Initialize the mixer module.
#     sound_effect_joy = pygame.mixer.Sound("./resources/audio/JOY.mp3")  # Load a sound.
#     #? Source: Sound Effect by freesound_community from Pixabay [https://pixabay.com/sound-effects/running-in-grass-6237/]
#     #! License: Free (https://pixabay.com/service/license-summary/)
#     #! Modification: None
#     sound_effect_joy.play()
    
#     for _ in range(100):
#         env = CustomEnv(grid_size=9)
#         # 1st Line Defenders
#         env.addDanger(coordinates=(3, 2), role="D")
#         env.addDanger(coordinates=(5, 2), role="D")
#         # 2nd Line Defenders
#         env.addDanger(coordinates=(1, 5), role="D")
#         env.addDanger(coordinates=(4, 4), role="D")
#         env.addDanger(coordinates=(7, 5), role="D")
#         #  Goal Keeper
#         env.addDanger(coordinates=(4, 6), role="GK")

#         state, info = env.reset()
#         env.render()

#         print("Initial_state: ", state, "Distance to goal: ", info["Distance to goal"])

#         for _ in range(15):
#             # action = int(input("Choose an action: "))
#             # next_step, done, reward, info = env.step(action)
            
#             next_step, done, reward, info = env.step(env.action_space.sample())

#             env.render()

#             print(f"Next-state: {next_step}, Done: {done}, Reward: {reward}, Distance to goal: {info['Distance to goal']}")

#             if done:
#                 env.close()
#                 break
            
#         env.close()
