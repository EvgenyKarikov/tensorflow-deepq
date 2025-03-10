from __future__ import division

import math
import time

import matplotlib.pyplot as plt
from itertools import count
from os.path import join, exists
from os import makedirs
from IPython.display import clear_output, display, HTML

import pygame

WHITE=(255,255,255)
RED=(255,0,0)
GREEN=(0,255,0)
BLUE=(0,0,255)
BLACK=(0,0,0)

def simulate(simulation,
             controller= None,
             fps=60,
             visualize_every=1,
             action_every=1,
             simulation_resolution=None,
             wait=False,
             disable_training=False,
             save_path=None):
    """Start the simulation. Performs three tasks

        - visualizes simulation in iPython notebook
        - advances simulator state
        - reports state to controller and chooses actions
          to be performed.

    Parameters
    -------
    simulation: tr_lr.simulation
        simulation that will be simulated ;-)
    controller: tr_lr.controller
        controller used
    fps: int
        frames per seconds
    visualize_every: int
        visualize every `visualize_every`-th frame.
    action_every: int
        take action every `action_every`-th frame
    simulation_resolution: float
        simulate at most 'simulation_resolution' seconds at a time.
        If None, the it is set to 1/FPS (default).
    wait: boolean
        whether to intentionally slow down the simulation
        to appear real time.
    disable_training: bool
        if true training_step is never called.
    save_path: str
        save svg visualization (only tl_rl.utils.svg
        supported for the moment)
    """

    # prepare path to save simulation images
    if save_path is not None:
        if not exists(save_path):
            makedirs(save_path)
    last_image = 0

    # calculate simulation times
    chunks_per_frame = 1
    chunk_length_s   = 1.0 / fps

    if simulation_resolution is not None:
        frame_length_s = 1.0 / fps
        chunks_per_frame = int(math.ceil(frame_length_s / simulation_resolution))
        chunks_per_frame = max(chunks_per_frame, 1)
        chunk_length_s = frame_length_s / chunks_per_frame

    # state transition bookkeeping
    last_observation = None
    last_action      = None

    simulation_started_time = time.time()

    # setup rendering handles for reuse
    if hasattr(simulation, 'setup_draw'): 
        simulation.setup_draw()

    for frame_no in count():
        for _ in range(chunks_per_frame):
            simulation.step(chunk_length_s)

        if frame_no % action_every == 0:
            new_observation = simulation.observe()
            reward          = simulation.collect_reward()
            # store last transition
            if last_observation is not None:
                controller.store(last_observation, last_action, reward, new_observation)

            # act
            new_action = controller.action(new_observation)
            simulation.perform_action(new_action)

            #train
            if not disable_training:
                controller.training_step()

            # update current state as last state.
            last_action = new_action
            last_observation = new_observation

        # adding 1 to make it less likely to happen at the same time as
        # action taking.
        if (frame_no + 1) % visualize_every == 0:
            fps_estimate = frame_no / (time.time() - simulation_started_time)

            # draw simulated environment all the rendering is handled within the simulation object 
            stats = ["fps = %.1f" % (fps_estimate, )]
            if hasattr(simulation, 'draw'): # render with the draw function
                simulation.draw(stats) 
            elif hasattr(simulation, 'to_html'): # in case some class only support svg rendering
                clear_output(wait=True)
                svg_html = simulation.to_html(stats)
                display(svg_html)

            if save_path is not None:
                img_path = join(save_path, "%d.svg" % (last_image,))
                with open(img_path, "w") as f:
                    svg_html.write_svg(f)
                last_image += 1

        time_should_have_passed = frame_no / fps
        time_passed = (time.time() - simulation_started_time)
        if wait and (time_should_have_passed > time_passed):
            time.sleep(time_should_have_passed - time_passed)

def simulate2(simulation,
             controller= None,
             fps=60,
             visualize_every=1,
             action_every=1,
             simulation_resolution=None,
             wait=False,
             disable_training=False,
             save_path=None):
    
    # calculate simulation times
    chunks_per_frame = 1
    chunk_length_s   = 1.0 / fps

    if simulation_resolution is not None:
        frame_length_s = 1.0 / fps
        chunks_per_frame = int(math.ceil(frame_length_s / simulation_resolution))
        chunks_per_frame = max(chunks_per_frame, 1)
        chunk_length_s = frame_length_s / chunks_per_frame

    last_observation = None
    last_action      = None

    pygame.init()
    pygame.font.init()
    pygame.mixer.init()


    screen_start_point = (10,10)
    screen_size = (2*screen_start_point[0]+simulation.size[0],2*screen_start_point[1]+simulation.size[1]+200)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Game")
    clock = pygame.time.Clock()

    running = True
    frame_no = 0
    while running:
        screen.fill(WHITE)
        # Держим цикл на правильной скорости        

        for _ in range(chunks_per_frame):
            simulation.step(chunk_length_s)
        
        if frame_no % action_every == 0:
            new_observation = simulation.observe()
            reward          = simulation.collect_reward()
            # store last transition
            if last_observation is not None:
                controller.store(last_observation, last_action, reward, new_observation)
            
            

            #act
            action = None
            if (controller==[]):
                for i in pygame.event.get():
                    if i.type == pygame.KEYDOWN:
                        if i.key == pygame.K_RIGHT:
                            action=0
                        elif i.key == pygame.K_DOWN:
                            action=1
                        elif i.key == pygame.K_LEFT:
                            action=2
                        elif i.key == pygame.K_UP:
                            action=3
            else:
                action = controller.action(new_observation)
            if (action!=None):
                simulation.perform_action(action)

            #train
            if not disable_training:
                controller.training_step()

            # update current state as last state.
            last_action = action
            last_observation = new_observation

        frame_no+=1
        
        #draw
        stats = []
        simulation.to_pygame_screen(screen,screen_start_point,[])

        clock.tick(fps)
        pygame.display.flip()
        # Ввод процесса (события)
        for event in pygame.event.get():
            # check for closing window
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
    pass