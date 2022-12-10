# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import math
import random
import contest.util as util
import time 

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint, manhattanDistance

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=4):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

CAPSULE_EFFECT_DURATION = 40

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time=.1):
        super().__init__(index, time)
        if index % 2 == 1:
            self.is_red = False
            self.enemy_indices = [0,2]
        else: 
            self.is_red = True
            self.enemy_indices = [1,3]

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        #print(actions)
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        #print(values)
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

    def is_on_opponent_field(self, game_state):
        pos = game_state.get_agent_position(self.index)
        width = (game_state.get_red_food().width)
        if self.is_red and pos[0] >= width/2 or self.is_red == False and pos[0] <= width/2:
            return True
        return False

class BaseAgent(CaptureAgent):
    def __init__(self, index, time=.1):
        super().__init__(index, time)
        if index % 2 == 1:
            self.is_red = False
            self.enemy_indices = [0,2]
        else: 
            self.is_red = True
            self.enemy_indices = [1,3]
        
    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        #start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        #print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """            # Find the enemy defense agent

        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

    def is_on_opponent_field(self, game_state):
        pos = game_state.get_agent_position(self.index)
        width = (game_state.get_red_food().width)
        if self.is_red and pos[0] >= width/2 or self.is_red == False and pos[0] <= width/2:
            return True
        return False

class OffensiveReflexAgent(BaseAgent):
    
    def __init__(self, index, time=.1):
        super().__init__(index, time)
        self.probability_to_return = 0
        self.returning = False
        self.capsuleEffect = False
        self.capsuleEffectDuration = CAPSULE_EFFECT_DURATION
        self.capsule_positions = []
    
    def n_food_eaten(self, successor):
        '''
            Returns the amount of food that an agent (pacman) is currently carrying
        '''
        start_food = 20 
        score = successor.get_score() #+ 20  TODO - self.opponent_score();
        #if not self.is_red:
        #    score *= -1
        food_left = len(self.get_food(successor).as_list())
        n_food_eaten = start_food - score - food_left
        return n_food_eaten        
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        
        # Check if on enemy side
        pos = successor.get_agent_position(self.index)
        width = (successor.get_red_food().width)
        if self.is_red and pos[0] >= width/2 or not self.is_red and pos[0] <= width/2:
            # Get distance to nearest opponent
            enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            alternative = False
            if len(enemies) > 0:
                try:
                    closest_opponent = min([self.get_maze_distance(pos, enemy) for enemy in enemies if enemy.get_position() is not None])
                    features["distance_to_opponent"] = closest_opponent # * stevilo_pozrte_hrane
                except BaseException:
                    alternative = True
            if alternative == True:
                opponent_distances = successor.get_agent_distances()      
                closest_opponent = min([opponent_distances[i] for i in self.enemy_indices])
                try:
                    features["distance_to_opponent"] = 1/closest_opponent # * stevilo_pozrte_hrane
                except ZeroDivisionError:
                    features["distance_to_opponent"] = 0
        else:
            features["distance_to_opponent"] = 1/100
            
        # Also minimize distance to capsule
        capsule_positions = self.get_capsules(successor)
        if len(capsule_positions) > 0:
            features["distance_to_capsule"] = min([self.get_maze_distance(pos, capsule_pos) for capsule_pos in capsule_positions])
        else:
            features["distance_to_capsule"] = 0
        
        
        carried_food = self.n_food_eaten(successor)
        features["food_eaten"] = carried_food
        return features


    def get_weights(self, game_state, action):
        if self.is_on_opponent_field(game_state):
            distance_to_opponent_weight = 10
        else:
            distance_to_opponent_weight = -10

        if self.capsuleEffect:
            distance_to_opponent_weight = -100
        
        return {'successor_score': 100, 'distance_to_food': -1, "distance_to_opponent": distance_to_opponent_weight, "food_eaten": 100, "distance_to_capsule": -1}


    def set_capsule_effect(self, successor):
        position = successor.get_agent_position(self.index)
        if any(capsule_pos == position for capsule_pos in self.get_capsules(successor)):
            self.capsuleEffect = True
            #print("Ate capsule")
            
    def track_capsule_effect(self):
        if self.capsuleEffectDuration > 0:
            self.capsuleEffectDuration -= 1
            self.capsuleEffect = True
        else:
            self.capsuleEffectDuration = 0
            self.capsuleEffect = False
            
    def dist_from_ghost(self, game_state):
        '''
            Returns distance from nearest ghost
        '''
        pos = game_state.get_agent_position(self.index)
        
        # Computes distance to invaders we can see
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        chasers = [a for a in enemies if a.is_pacman == False and a.get_position() is not None]
        if len(chasers) > 0:
            dists = [self.get_maze_distance(pos, a.get_position()) for a in chasers]
            return min(dists)
        else:
            opponent_distances = game_state.get_agent_distances()      
            return min([opponent_distances[i] for i in self.enemy_indices])

    def run_from_ghost(self, actions, game_state):
        '''
        Makes pacman return as safe as possible
        '''
        
        best_dist = 9999
        current_dist_from_ghost = self.dist_from_ghost(game_state)
        best_action = None
        for action in actions:
            successor = self.get_successor(game_state, action)
            pos2 = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(self.start, pos2)
            
            dist_from_ghost = self.dist_from_ghost(successor)
                        
            # Run home and away from
            if dist < best_dist and dist_from_ghost >= current_dist_from_ghost:
                best_action = action
                best_dist = dist
            elif dist < best_dist:
                best_action = action
                best_dist = dist
                
        return best_action

    def get_probability_to_return(self, features):
        increment_per_food = .25
        if self.capsuleEffect:
            increment_per_food = .1
            self.returning = False
        
        probability_to_return = 0
        if features["food_eaten"] > 0:
            probability_to_return = increment_per_food * features["food_eaten"]
        if features["distance_to_opponent"] >= 1/5:
            probability_to_return += .2
        probability_to_return = min(probability_to_return, 1)
        return probability_to_return

    def get_capsules(self, game_state):
        if self.is_red:
            return game_state.get_blue_capsules()
        else:
            return game_state.get_red_capsules()
            

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        
        print("HAI")
        if len(self.capsule_positions) == 0:
            self.capsule_positions = self.get_capsules(game_state)
        
        if self.capsuleEffect:
            self.track_capsule_effect()
        actions = game_state.get_legal_actions(self.index)
        #print("Capsule: ", self.capsuleEffect, self.capsuleEffectDuration)
        
        # Get features
        features = self.get_features(game_state, "Stop");
        #print(features)

        
        if self.is_on_opponent_field(game_state) == False:
            features["food_eaten"] = 0
        
        # If pacman has not eaten, is more than 5 fields from opponent or ate the capsule, it should not run
        if self.n_food_eaten(game_state) == 0  or self.capsuleEffect == True or self.is_on_opponent_field(game_state) == False:
            self.probability_to_return = 0
            self.returning = False
            

        # You can profile your evaluation time by uncommenting these lines
        #start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        #print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))


        self.probability_to_return = self.get_probability_to_return(features)
        #print(self.probability_to_return, self.returning)    
     
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())
        
        # Only make this decision if we aren't already returning
        if self.returning == False and self.capsuleEffect == False:
            self.returning = random.choices([True, False], weights=[self.probability_to_return, 1-self.probability_to_return])

        # TODO - take opponent position in account
        if food_left <= 2 or self.probability_to_return == 1 or self.returning == True: 
            #print("returning")
            action = self.run_from_ghost(actions, game_state)
            successor = game_state.generate_successor(self.index, action)
            self.set_capsule_effect(successor)
            return action

        action = random.choice(best_actions)
        successor = game_state.generate_successor(self.index, action)
        self.set_capsule_effect(successor)
            
        return action
    
class DefensiveReflexAgent(ReflexCaptureAgent):

  
    def __init__(self, index, time=.1):
        super().__init__(index, time)
        self.field_to_go_to = None

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        #print(successor)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0
        
        print(self.get_game_data(game_state).agent_states[0])
        print(self.get_game_data(game_state).agent_states[1])
        print(self.get_game_data(game_state).agent_states[2])
        print(self.get_game_data(game_state).agent_states[3])
        print(".....................................................")
        print(self.get_game_data(game_state))
        print(self.get_field_to_go_to(successor))



        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        trespassers = [enemy for enemy in enemies if self.is_on_opponent_field(game_state)]
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            opponent_distances = successor.get_agent_distances()      
            features["invader_distance"] = min([opponent_distances[i] for i in self.enemy_indices])

        #print(features["invader_distance"])
        if len(invaders) == 0:
            features['guard_border'] = self.is_by_border(game_state, my_pos)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def is_by_border(self, game_state, my_pos):
        #pos = game_state.get_agent_position(self.index)
        width = (game_state.get_red_food().width)
        wid = width/6
        if self.is_red:
            vrednost = 2 - math.floor(my_pos[0]/wid)
            #print(vrednost, "bla")
        else:
            vrednost = math.floor(my_pos[0]/wid) - 3
            #print(vrednost)
        return vrednost

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2, 'guard_border': -3}

    def get_normalizing_term(self, game_state) -> float:
        width = game_state.get_red_food().width
        height = game_state.get_red_food().height
        return 1 / (width * height)
    
    def P(self, v):
        return .25
    
    def get_layout(self, game_state):
        return game_state.data.layout
    
    def get_game_data(self, game_state):
        return game_state.data

    def n_fields(self, game_state) -> int:
        width = game_state.get_red_food().width
        height = game_state.get_red_food().height
        n_fields = width * height - len(game_state.get_walls())
        return n_fields
    
    def get_enemy_indices(self, game_state):
        if self.is_red:
            return game_state.get_blue_team_indices()
        else:
            return game_state.get_red_team_indices();
        
    def get_probability(self, game_state, field) -> float:
        '''
            Returns a probability of pacman being on field in N+k steps
        '''
        game_data = self.get_game_data(game_state)
        n_fields = self.n_fields(game_state)
        w = game_data.layout.width
        h = game_data.layout.height
        

        #backtracks = [i for i in range(3)]
        #distance = 7
        #if len(enemy_distances) > 0:
        #    distance = min(enemy_distances)
        distance = manhattanDistance(game_state.get_agent_position(self.index), field)
        
        return (24 * w * h * (distance - 1)**2) / (n_fields * distance * (2 * distance**2 + 3 * distance - 1))
        
    def get_new_position(self, pos, move):
        pos = [pos[0], pos[1]]
        if move == "North":
            pos[0] += 1
        elif move == "South":
            pos[0] -= 1
        elif move == "West":
            pos[1] -= 1
        elif move == "East":
            pos[1] += 1
            
        return (pos[0], pos[1])
    
    def get_fields_in_range(self, game_state, enemy_positions):
        layout = self.get_game_data(game_state)
        fields = []
        for enemy in enemy_positions:
            k = manhattanDistance(game_state.get_agent_position(self.index), enemy)
            for i in range(k):
                # try a path
                moves = 0
                field_in_range = None
                for move in ["North", "South", "West", "East"]:
                    new_pos = self.get_new_position(enemy, move)

                    if game_state.has_wall(new_pos[0], new_pos[1]):
                        break
                    k += 1
                    field_in_range = new_pos
                    print("heh")
                fields.append(field_in_range)
        print("FIELDS", fields)
        return fields
                          
        
    def get_field_to_go_to(self, game_state):

        enemy_indices = self.get_enemy_indices(game_state)
        enemy_positions = [game_state.get_agent_position(i) for i in enemy_indices if game_state.get_agent_state(i).is_pacman and game_state.get_agent_position(i) is not None]
        fields = self.get_fields_in_range(game_state, enemy_positions)
        
        best_field, max_prob = 0, 0
        print("HEKKK")
        for field in fields:
            if field is not None:
                field_prob = self.get_probability(game_state, field)
                if field_prob > max_prob:
                    max_prob = field_prob
                    best_field = field
                    
        return best_field
