import os
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from tf_rl.simulate import simulate2
from tf_rl.controller import DiscreteDeepQ
from tf_rl.simulation import KarpathyGame
from tf_rl.models import MLP

import tensorflow as tf

tf = tf.compat.v1

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  use_gpu = False
  tf_device = tf.device('/cpu:0')
else:
  use_gpu = True
  tf_device = tf.device('/device:GPU:0')
print('use gpu:', use_gpu)

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)

current_settings = {
    'objects': [
        'friend',
        'enemy',
    ],
    'colors': {
        'hero':   'yellow',
        'friend': 'green',
        'enemy':  'red',
    },
    'object_reward': {
        'friend': 1,
        'enemy': -1,
    },
    'hero_bounces_off_walls': False,
    'world_size': (700, 500),
    'hero_initial_position': [400, 300],
    'hero_initial_speed':    [0,   0],
    "maximum_speed":         [50, 50],
    "object_radius": 10.0,
    "num_objects": {
        "friend": 25,
        "enemy":  25,
    },
    "num_observation_lines": 32,
    "observation_line_length": 120.,
    "tolerable_distance_to_wall": 50,
    "wall_distance_penalty": -1,
    "delta_v": 50
}

kgame = KarpathyGame(current_settings)

human_control = False

if human_control:
    # WSAD CONTROL (requires extra setup - check out README)
    keyb_mapping = {b"w": 3, b"d": 0, b"s": 1, b"a": 2, }
    #current_controller = HumanController(keyb_mapping)
else:
    # Tensorflow business - it is always good to reset a graph before creating a new controller.
    tf.reset_default_graph()
    session = tf.InteractiveSession()

    # This little guy will let us run tensorboard
    #      tensorboard --logdir [LOG_DIR]
    #journalist = tf.train.SummaryWriter(LOG_DIR)

    # Brain maps from observation to Q values for different actions.
    # Here it is a done using a multi layer perceptron with 2 hidden
    # layers
    brain = MLP([kgame.observation_size, ], [200, 200, kgame.num_actions],
                [tf.tanh, tf.tanh, tf.identity])

    # The optimizer to use. Here we use RMSProp as recommended
    # by the publication
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.9,beta2=0.999)

    # DiscreteDeepQ object
    current_controller = DiscreteDeepQ((kgame.observation_size,), kgame.num_actions, brain, optimizer, session,
                                       discount_rate=0.99, exploration_period=5000, max_experience=10000,
                                       store_every_nth=4, train_every_nth=4)

    session.run(tf.initialize_all_variables())
    session.run(current_controller.target_network_update)
    # graph was not available when journalist was created
    #journalist.add_graph(session.graph)

FPS = 30
ACTION_EVERY = 3

fast_mode = True
if fast_mode:
    WAIT, VISUALIZE_EVERY = False, 50
else:
    WAIT, VISUALIZE_EVERY = True, 1

try:
    with tf.device(tf_device):
        simulate2(simulation=kgame,
                  controller=current_controller,
                  fps=FPS,
                  visualize_every=VISUALIZE_EVERY,
                  action_every=ACTION_EVERY,
                  wait=WAIT,
                  disable_training=False,
                  simulation_resolution=0.001,
                  save_path=None)
except KeyboardInterrupt:
    print("Interrupted")

